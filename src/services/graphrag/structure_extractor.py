#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Callable, Optional

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

    def _get_raw_code(self, address: int) -> Optional[str]:
        if not address:
            return None
        for level in (ViewLevel.PSEUDO_C, ViewLevel.HLIL, ViewLevel.MLIL, ViewLevel.LLIL, ViewLevel.ASM):
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
