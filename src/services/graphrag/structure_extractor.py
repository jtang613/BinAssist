#!/usr/bin/env python3

import os
import threading
from dataclasses import dataclass
from typing import Callable, List, Optional

from .graph_store import GraphStore
from .models import GraphNode, GraphEdge, EdgeType
from .security_feature_extractor import SecurityFeatureExtractor
from ..binary_context_service import BinaryContextService, ViewLevel

try:
    import binaryninja
    log = binaryninja.log.Logger(0, "BinAssist")
except ImportError:
    class MockLog:
        @staticmethod
        def log_info(msg): print(f"[BinAssist] INFO: {msg}")
        @staticmethod
        def log_warn(msg): print(f"[BinAssist] WARN: {msg}")
        @staticmethod
        def log_error(msg): print(f"[BinAssist] ERROR: {msg}")
    log = MockLog()


# Number of worker threads for parallel extraction
DEFAULT_THREAD_COUNT = max(2, os.cpu_count() // 2 if os.cpu_count() else 2)


@dataclass
class ExtractionResult:
    """Result of structure extraction."""
    functions_extracted: int = 0
    edges_created: int = 0


class StructureExtractor:
    """Extracts function nodes and relationships into the GraphRAG store."""

    def __init__(self, binary_view, graph_store: GraphStore):
        self.binary_view = binary_view
        self.graph_store = graph_store
        self.security_extractor = SecurityFeatureExtractor(binary_view)
        self.context_service = BinaryContextService(binary_view)
        self._cancelled = False
        self._progress_lock = threading.Lock()

    def cancel(self):
        """Cancel the extraction process."""
        self._cancelled = True

    def extract_function(self, function, binary_hash: str) -> Optional[GraphNode]:
        if function is None or not binary_hash:
            return None

        address = int(function.start)
        node = self.graph_store.get_node_by_address(binary_hash, "FUNCTION", address)
        if node is None:
            node = GraphNode(
                binary_hash=binary_hash,
                node_type="FUNCTION",
                address=address,
                name=function.name,
            )
        else:
            node.name = function.name

        raw_code = self._get_raw_code(address)
        features = self.security_extractor.extract_features(function, raw_code)
        node.network_apis = sorted(features.network_apis)
        node.file_io_apis = sorted(features.file_io_apis)
        node.ip_addresses = sorted(features.ip_addresses)
        node.urls = sorted(features.urls)
        node.file_paths = sorted(features.file_paths)
        node.domains = sorted(features.domains)
        node.registry_keys = sorted(features.registry_keys)
        node.activity_profile = features.get_activity_profile()
        node.risk_level = features.get_risk_level()
        node.security_flags = features.generate_security_flags()
        if raw_code:
            node.raw_code = raw_code
        node.is_stale = True

        node = self.graph_store.upsert_node(node)

        self._extract_call_edges(function, binary_hash, node)
        return node

    def _extract_call_edges(self, function, binary_hash: str, node: GraphNode) -> int:
        """Extract call edges for a function. Returns count of edges created."""
        edges_created = 0

        # Extract outgoing calls (this function calls others)
        callees = getattr(function, "callees", None) or []
        for callee in callees:
            callee_addr = int(callee.start)
            callee_node = self.graph_store.get_node_by_address(binary_hash, "FUNCTION", callee_addr)
            if callee_node is None:
                callee_node = GraphNode(
                    binary_hash=binary_hash,
                    node_type="FUNCTION",
                    address=callee_addr,
                    name=callee.name,
                    is_stale=True,
                )
                callee_node = self.graph_store.upsert_node(callee_node)

            # Create CALLS edge using EdgeType enum
            self.graph_store.add_edge(GraphEdge(
                binary_hash=binary_hash,
                source_id=node.id,
                target_id=callee_node.id,
                edge_type=EdgeType.CALLS,
                weight=1.0,
            ))
            edges_created += 1

            # Check if callee has security risk flags
            if callee_node.security_flags and any(flag.endswith("_RISK") for flag in callee_node.security_flags):
                self.graph_store.add_edge(GraphEdge(
                    binary_hash=binary_hash,
                    source_id=node.id,
                    target_id=callee_node.id,
                    edge_type=EdgeType.CALLS_VULNERABLE,
                    weight=1.0,
                ))
                edges_created += 1
                if "CALLS_VULNERABLE_FUNCTION" not in node.security_flags:
                    node.security_flags.append("CALLS_VULNERABLE_FUNCTION")
                    self.graph_store.upsert_node(node)

        # Extract incoming calls (others call this function)
        # This ensures bidirectional caller/callee relationship
        callers = getattr(function, "callers", None) or []
        for caller in callers:
            caller_addr = int(caller.start)
            caller_node = self.graph_store.get_node_by_address(binary_hash, "FUNCTION", caller_addr)
            if caller_node is None:
                # Create placeholder node for caller if not already indexed
                caller_node = GraphNode(
                    binary_hash=binary_hash,
                    node_type="FUNCTION",
                    address=caller_addr,
                    name=caller.name,
                    is_stale=True,
                )
                caller_node = self.graph_store.upsert_node(caller_node)

            # Only create edge if it doesn't already exist (avoid duplicates)
            if not self.graph_store.has_edge(caller_node.id, node.id, EdgeType.CALLS.value):
                self.graph_store.add_edge(GraphEdge(
                    binary_hash=binary_hash,
                    source_id=caller_node.id,
                    target_id=node.id,
                    edge_type=EdgeType.CALLS,
                    weight=1.0,
                ))
                edges_created += 1

        return edges_created

    def extract_structure(
        self,
        binary_hash: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> ExtractionResult:
        """
        Phase 1: Fast structure-only extraction.

        Extracts function names, addresses, and call graph.
        No decompilation - just structure. This is fast (~15-30 seconds for 8000 functions).

        Args:
            binary_hash: Hash of the binary being analyzed
            progress_callback: Optional callback(current, total, message) for progress updates

        Returns:
            ExtractionResult with extraction statistics
        """
        self._cancelled = False
        result = ExtractionResult()
        functions = list(self.binary_view.functions)
        total = len(functions)

        if total == 0:
            return result

        log.log_info(f"Phase 1: Extracting structure for {total} functions")

        # Clean up any duplicate nodes/edges from previous bad reindexes
        dup_nodes = self.graph_store.deduplicate_nodes(binary_hash)
        dup_edges = self.graph_store.deduplicate_edges(binary_hash)
        if dup_nodes > 0 or dup_edges > 0:
            log.log_info(f"Removed {dup_nodes} duplicate nodes, {dup_edges} duplicate edges")

        # Pre-load existing node IDs to enable non-destructive reindexing
        existing_count = self.graph_store.preload_node_cache(binary_hash)
        if existing_count > 0:
            log.log_info(f"Loaded {existing_count} existing node IDs for reindex")

        if progress_callback:
            progress_callback(0, total, f"Phase 1: Extracting structure...")

        for i, func in enumerate(functions):
            if self._cancelled:
                break

            try:
                address = int(func.start)
                name = func.name

                # Create node - NO decompilation yet
                node = GraphNode(
                    binary_hash=binary_hash,
                    node_type="FUNCTION",
                    address=address,
                    name=name,
                    raw_code=None,  # Populated in Phase 2
                    is_stale=True,
                )
                node = self.graph_store.queue_node_for_batch(node)

                # Extract call edges (fast - no decompilation)
                for callee in (getattr(func, "callees", None) or []):
                    try:
                        callee_addr = int(callee.start)
                        callee_id = self.graph_store.get_cached_node_id(binary_hash, callee_addr)
                        if not callee_id:
                            callee_node = GraphNode(
                                binary_hash=binary_hash,
                                node_type="FUNCTION",
                                address=callee_addr,
                                name=callee.name,
                                is_stale=True,
                            )
                            callee_node = self.graph_store.queue_node_for_batch(callee_node)
                            callee_id = callee_node.id
                        self.graph_store.queue_edge_for_batch(
                            node.id, callee_id, EdgeType.CALLS, binary_hash
                        )
                    except Exception:
                        pass

                result.functions_extracted += 1

                # Progress every 10 functions (low cost, high UX benefit)
                if progress_callback and (i % 10 == 0 or i == total - 1):
                    progress_callback(i + 1, total, f"Phase 1: {i + 1}/{total} functions")

            except Exception as e:
                log.log_warn(f"Failed to extract {func.name}: {e}")

            if (i + 1) % 500 == 0:
                self.graph_store.flush_all_batches()

        self.graph_store.flush_all_batches()

        edges = self.graph_store.get_edges_by_types(binary_hash, [EdgeType.CALLS.value])
        result.edges_created = len(edges)

        log.log_info(f"Phase 1 complete: {result.functions_extracted} functions, {result.edges_created} edges")
        return result

    def enrich_all(
        self,
        binary_hash: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> int:
        """
        Phase 2: Enrich all stale nodes with decompiled code and security features.

        This is the slower phase that performs decompilation. Progress is reported
        every 10 functions to keep the UI responsive.

        Args:
            binary_hash: Hash of the binary being analyzed
            progress_callback: Optional callback(current, total, message) for progress updates

        Returns:
            Count of enriched functions
        """
        self._cancelled = False

        # Get all stale nodes that need enrichment
        stale_nodes = self.graph_store.get_stale_nodes(binary_hash)
        total = len(stale_nodes)

        if total == 0:
            return 0

        log.log_info(f"Phase 2: Enriching {total} functions with decompilation")

        if progress_callback:
            progress_callback(0, total, f"Phase 2: Decompiling...")

        enriched = 0
        for i, node in enumerate(stale_nodes):
            if self._cancelled:
                break

            try:
                func = self.binary_view.get_function_at(node.address)
                if not func:
                    continue

                # Decompile (expensive but necessary)
                raw_code = self._get_raw_code(node.address)
                if raw_code:
                    node.raw_code = raw_code

                # Extract security features
                features = self.security_extractor.extract_features(func, raw_code)
                node.network_apis = sorted(features.network_apis)
                node.file_io_apis = sorted(features.file_io_apis)
                node.ip_addresses = sorted(features.ip_addresses)
                node.urls = sorted(features.urls)
                node.file_paths = sorted(features.file_paths)
                node.domains = sorted(features.domains)
                node.registry_keys = sorted(features.registry_keys)
                node.activity_profile = features.get_activity_profile()
                node.risk_level = features.get_risk_level()
                node.security_flags = features.generate_security_flags()
                node.is_stale = False

                self.graph_store.upsert_node(node)
                enriched += 1

                # Progress every 10 functions (low cost, high UX benefit)
                if progress_callback and (i % 10 == 0 or i == total - 1):
                    progress_callback(i + 1, total, f"Phase 2: {i + 1}/{total} decompiled")

            except Exception as e:
                log.log_warn(f"Failed to enrich {node.name}: {e}")

        log.log_info(f"Phase 2 complete: {enriched} functions enriched")
        return enriched

    def extract_all(
        self,
        binary_hash: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> ExtractionResult:
        """
        Extract all functions and their relationships from the binary.

        Args:
            binary_hash: Hash of the binary being analyzed
            progress_callback: Optional callback(current, total, message) for progress updates

        Returns:
            ExtractionResult with extraction statistics
        """
        result = ExtractionResult()
        functions = list(self.binary_view.functions)
        total = len(functions)

        if progress_callback:
            progress_callback(0, total, f"Extracting {total} functions...")

        for i, func in enumerate(functions):
            try:
                node = self.extract_function(func, binary_hash)
                if node:
                    result.functions_extracted += 1

                if progress_callback and (i % 10 == 0 or i == total - 1):
                    progress_callback(i + 1, total, f"Extracted {func.name}")

            except Exception as e:
                log.log_warn(f"Failed to extract function {func.name}: {e}")

        # Count edges created
        edges = self.graph_store.get_edges_by_types(binary_hash, [EdgeType.CALLS.value])
        result.edges_created = len(edges)

        if progress_callback:
            progress_callback(total, total, "Extraction complete")

        log.log_info(f"Structure extraction complete: {result.functions_extracted} functions, {result.edges_created} edges")
        return result

    def extract_all_parallel(
        self,
        binary_hash: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        thread_count: int = DEFAULT_THREAD_COUNT
    ) -> ExtractionResult:
        """
        Two-phase extraction with progress on both phases.

        Phase 1: Fast structure extraction (names, addresses, call graph)
        Phase 2: Decompilation and security feature extraction

        Binary Ninja's Python API cannot parallelize decompilation like Ghidra's
        Java API. The total time is similar but progress is now visible throughout.

        Args:
            binary_hash: Hash of the binary being analyzed
            progress_callback: Optional callback(current, total, message) for progress updates
            thread_count: Number of worker threads (unused in two-phase approach)

        Returns:
            ExtractionResult with extraction statistics
        """
        # Phase 1: Structure (fast - ~15-30 seconds for 8000 functions)
        result = self.extract_structure(binary_hash, progress_callback)

        if self._cancelled:
            return result

        # Phase 2: Enrichment (slower but with progress - decompilation)
        self.enrich_all(binary_hash, progress_callback)

        return result

    def _get_raw_code(self, address: int) -> Optional[str]:
        if not address:
            return None
        for level in (ViewLevel.HLIL, ViewLevel.MLIL, ViewLevel.LLIL, ViewLevel.ASM):
            result = self.context_service.get_code_at_level(address, level, context_lines=0)
            if result.get("error"):
                continue
            lines = result.get("lines") or []
            parts = []
            for line in lines:
                if isinstance(line, dict):
                    content = line.get("content")
                else:
                    content = str(line)
                if content:
                    parts.append(content.rstrip())
            if parts:
                return "\n".join(parts).strip()
        return None
