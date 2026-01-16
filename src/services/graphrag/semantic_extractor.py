#!/usr/bin/env python3

import asyncio
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

from .extraction_prompts import function_summary_prompt
from .security_feature_extractor import SecurityFeatureExtractor
from .graph_store import GraphStore
from .models import GraphNode
from ..binary_context_service import BinaryContextService, ViewLevel
from ..models.llm_models import ChatRequest, ChatMessage, MessageRole

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
    summarized: int
    errors: int
    elapsed_ms: int


class SemanticExtractor:
    DEFAULT_BATCH_SIZE = 3
    DEFAULT_DELAY_MS = 500

    def __init__(self, provider, graph_store: GraphStore, binary_view, binary_hash: str):
        self.provider = provider
        self.graph_store = graph_store
        self.binary_view = binary_view
        self.binary_hash = binary_hash
        self.batch_size = self.DEFAULT_BATCH_SIZE
        self.delay_between_batches = self.DEFAULT_DELAY_MS / 1000.0
        self.cancelled = False
        self._context_service = BinaryContextService(binary_view) if binary_view else None
        self._security_extractor = SecurityFeatureExtractor(binary_view) if binary_view else None

    def set_batch_config(self, batch_size: int, delay_ms: int) -> None:
        self.batch_size = max(1, batch_size)
        self.delay_between_batches = max(0, delay_ms) / 1000.0

    def cancel(self) -> None:
        self.cancelled = True

    async def summarize_stale_nodes(
        self,
        limit: int = 0,
        progress_callback: Optional[Callable[[int, int, int, int], None]] = None,
    ) -> ExtractionResult:
        start = time.time()
        summarized = 0
        errors = 0

        nodes = self.graph_store.get_stale_nodes(self.binary_hash, limit)
        total = len(nodes)
        if total == 0:
            return ExtractionResult(0, 0, 0)

        processed = 0
        for idx, node in enumerate(nodes):
            if self.cancelled:
                break
            success = await self._summarize_node(node)
            processed += 1
            if success:
                summarized += 1
            else:
                errors += 1
            if progress_callback:
                progress_callback(processed, total, summarized, errors)

            if self.delay_between_batches > 0 and (idx + 1) % self.batch_size == 0:
                await asyncio.sleep(self.delay_between_batches)

        elapsed_ms = int((time.time() - start) * 1000)
        return ExtractionResult(summarized, errors, elapsed_ms)

    async def _summarize_node(self, node: GraphNode) -> bool:
        if node is None or node.node_type != "FUNCTION":
            return False
        if node.user_edited:
            return False
        if not node.raw_code:
            node.raw_code = self._get_raw_code(node.address)
            if node.raw_code:
                self.graph_store.upsert_node(node)

        if not node.raw_code:
            log.log_warn(f"Skipping summary for {node.name or node.address}: missing raw code")
            return False

        self._refresh_security_features(node)

        callers = self._get_callers(node.id)
        callees = self._get_callees(node.id)
        prompt = function_summary_prompt(
            node.name or f"sub_{node.address:x}",
            node.raw_code,
            callers,
            callees,
        )

        response = await self._call_llm(prompt)
        if not response:
            return False

        node.llm_summary = response.strip()
        node.is_stale = False
        node.user_edited = False
        self.graph_store.upsert_node(node)
        return True

    def _refresh_security_features(self, node: GraphNode) -> None:
        if not self._security_extractor or not self.binary_view or node.address is None:
            return
        try:
            function = self.binary_view.get_function_at(node.address)
            if not function:
                funcs = self.binary_view.get_functions_containing(node.address)
                function = funcs[0] if funcs else None
            if not function:
                return

            features = self._security_extractor.extract_features(function, node.raw_code)

            node.network_apis = self._merge_list(node.network_apis, features.network_apis)
            node.file_io_apis = self._merge_list(node.file_io_apis, features.file_io_apis)
            node.ip_addresses = self._merge_list(node.ip_addresses, features.ip_addresses)
            node.urls = self._merge_list(node.urls, features.urls)
            node.file_paths = self._merge_list(node.file_paths, features.file_paths)
            node.domains = self._merge_list(node.domains, features.domains)
            node.registry_keys = self._merge_list(node.registry_keys, features.registry_keys)

            node.activity_profile = features.get_activity_profile()
            node.risk_level = features.get_risk_level()

            new_flags = set(features.generate_security_flags())
            node.security_flags = sorted(set(node.security_flags or []).union(new_flags))

            self.graph_store.upsert_node(node)
        except Exception as exc:
            log.log_warn(f"Failed to refresh security features: {exc}")

    @staticmethod
    def _merge_list(existing, new_values):
        merged = set(existing or [])
        merged.update(new_values or [])
        return sorted(merged)

    async def _call_llm(self, prompt: str) -> Optional[str]:
        try:
            messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
            request = ChatRequest(
                messages=messages,
                model=self.provider.model,
                stream=False,
                max_tokens=self.provider.max_tokens,
            )
            response = await self.provider.chat_completion(request)
            return response.content if response else None
        except Exception as exc:
            log.log_error(f"Semantic LLM call failed: {exc}")
            return None

    def _get_raw_code(self, address: Optional[int]) -> Optional[str]:
        if not address or not self._context_service:
            return None
        for level in (ViewLevel.PSEUDO_C, ViewLevel.HLIL, ViewLevel.MLIL, ViewLevel.LLIL, ViewLevel.ASM):
            result = self._context_service.get_code_at_level(address, level, context_lines=0)
            if result.get("error"):
                continue
            lines = result.get("lines") or []
            parts: List[str] = []
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

    def _get_callers(self, node_id: Optional[int]) -> List[str]:
        if not node_id:
            return []
        callers = self.graph_store.get_callers(self.binary_hash, node_id, "calls")
        results = []
        for node in callers:
            results.append(node.name or f"sub_{node.address:x}")
            if len(results) >= 5:
                break
        return results

    def _get_callees(self, node_id: Optional[int]) -> List[str]:
        if not node_id:
            return []
        edges = self.graph_store.get_edges_for_node(self.binary_hash, node_id)
        results = []
        for edge in edges:
            if edge.source_id == node_id and edge.edge_type == "calls":
                node = self.graph_store.get_node_by_id(edge.target_id)
                if node:
                    results.append(node.name or f"sub_{node.address:x}")
            if len(results) >= 5:
                break
        return results
