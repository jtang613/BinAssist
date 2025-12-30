#!/usr/bin/env python3

import asyncio
import json
from typing import Any, Dict, List, Optional, Set

import binaryninja as bn

from PySide6.QtCore import QThread, Signal

from ..services.analysis_db_service import AnalysisDBService
from ..services.settings_service import SettingsService
from ..services.graphrag.graphrag_service import GraphRAGService
from ..services.graphrag.graph_store import GraphStore
from ..services.graphrag.taint_analyzer import TaintAnalyzer
from ..services.graphrag.query_engine import GraphRAGQueryEngine
from ..services.llm_providers.provider_factory import LLMProviderFactory

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


class SemanticGraphController:
    def __init__(self, view, binary_view: Optional[bn.BinaryView] = None, view_frame=None):
        self.view = view
        self.binary_view = binary_view
        self.view_frame = view_frame
        self.analysis_db = AnalysisDBService()
        self.graph_service = GraphRAGService.get_instance(self.analysis_db)
        self.graph_store = GraphStore(self.analysis_db)
        self.settings_service = SettingsService()
        self.llm_factory = LLMProviderFactory()
        self._current_offset = 0
        self._semantic_worker = None
        self._security_worker = None
        self._reindex_worker = None

        self._connect_signals()

    def _connect_signals(self):
        self.view.go_requested.connect(self.handle_go)
        self.view.reset_requested.connect(self.handle_reset)
        self.view.reindex_requested.connect(self.handle_reindex)
        self.view.semantic_analysis_requested.connect(self.handle_semantic_analysis)
        self.view.security_analysis_requested.connect(self.handle_security_analysis)
        self.view.refresh_names_requested.connect(self.handle_refresh_names)
        self.view.navigate_requested.connect(self.handle_navigate)

        self.view.index_function_requested.connect(self.handle_index_function)
        self.view.save_summary_requested.connect(self.handle_save_summary)
        self.view.add_flag_requested.connect(self.handle_add_flag)
        self.view.remove_flag_requested.connect(self.handle_remove_flag)
        self.view.edge_clicked.connect(self.handle_edge_click)
        self.view.visual_refresh_requested.connect(self.handle_visual_refresh)
        self.view.search_query_requested.connect(self.handle_search_query)

    def set_binary_view(self, binary_view: Optional[bn.BinaryView]):
        self.binary_view = binary_view

    def set_view_frame(self, view_frame):
        self.view_frame = view_frame

    def set_current_offset(self, offset: int):
        self._current_offset = offset
        function = self._get_function_at(offset)
        name = function.name if function else None
        self.view.update_location(offset, name)
        self.refresh_current_view()

    def refresh_current_view(self):
        if not self.binary_view:
            return
        binary_hash = self.analysis_db.get_binary_hash(self.binary_view)
        stats = self.graph_store.get_graph_stats(binary_hash)
        self.view.update_stats(stats["nodes"], stats["edges"], stats["stale"], stats["last_indexed"])

        function = self._get_function_at(self._current_offset)
        if not function:
            self.view.update_status(False, 0, 0, 0)
            self.view.list_view.show_not_indexed()
            self.view.graph_view.show_not_indexed()
            return

        node = self.graph_service.get_node_by_address(binary_hash, "FUNCTION", int(function.start))
        if not node:
            self.view.update_status(False, 0, 0, 0)
            self.view.list_view.show_not_indexed()
            self.view.graph_view.show_not_indexed()
            return

        callers = self.graph_store.get_callers(binary_hash, node.id, "CALLS")
        callees = self._get_callees(binary_hash, node.id)
        self.view.update_status(True, len(callers), len(callees), len(node.security_flags))

        self.view.list_view.show_content()
        self.view.list_view.set_callers(self._nodes_to_entries(callers))
        self.view.list_view.set_callees(self._nodes_to_entries(callees))
        self.view.list_view.set_edges(self._edges_for_list(binary_hash, node.id))
        self.view.list_view.set_security_flags(node.security_flags)
        self.view.list_view.set_summary(node.llm_summary or "")

        self.view.graph_view.show_content()
        self.handle_visual_refresh(self.view.graph_view.n_hops.value(), ["CALLS", "REFERENCES", "DATA_DEPENDS", "CALLS_VULNERABLE"])

    def handle_go(self, text: str):
        addr = self._resolve_address(text)
        if addr is None:
            log.log_warn(f"Go failed: unable to resolve {text}")
            return
        self.handle_navigate(addr)

    def handle_navigate(self, address: int):
        if not self.view_frame:
            return
        try:
            if hasattr(self.view_frame, "getCurrentView"):
                view_name = self.view_frame.getCurrentView()
                if view_name and hasattr(self.view_frame, "navigate"):
                    self.view_frame.navigate(view_name, address)
                    return
            if hasattr(self.view_frame, "navigate"):
                self.view_frame.navigate("Linear", address)
        except Exception as e:
            log.log_warn(f"Navigation failed: {e}")

    def handle_index_function(self, address: int):
        if not self.binary_view:
            return
        binary_hash = self.analysis_db.get_binary_hash(self.binary_view)
        function = self._get_function_at(address or self._current_offset)
        if not function:
            return
        self.graph_service.index_function(self.binary_view, function, binary_hash)
        self.refresh_current_view()

    def handle_reindex(self):
        if not self.binary_view:
            return
        if self._reindex_worker and self._reindex_worker.isRunning():
            self._reindex_worker.cancel()
            self.view.status_label.setText("Status: Stopping reindex...")
            return
        binary_hash = self.analysis_db.get_binary_hash(self.binary_view)
        self.view.status_label.setText("Status: Reindexing...")
        self._set_reindex_button_running(True)
        self._reindex_worker = ReindexWorker(
            self.graph_service,
            self.graph_store,
            self.binary_view,
            binary_hash,
        )
        self._reindex_worker.progress.connect(self._on_reindex_progress)
        self._reindex_worker.completed.connect(self._on_reindex_complete)
        self._reindex_worker.cancelled.connect(self._on_reindex_cancelled)
        self._reindex_worker.failed.connect(self._on_reindex_error)
        self._reindex_worker.start()

    def handle_reset(self):
        if not self.binary_view:
            return
        binary_hash = self.analysis_db.get_binary_hash(self.binary_view)
        self.graph_store.delete_graph(binary_hash)
        self.refresh_current_view()

    def handle_semantic_analysis(self):
        if not self.binary_view:
            return
        provider_config = self.settings_service.get_active_llm_provider()
        if not provider_config:
            log.log_warn("No active LLM provider configured for semantic analysis.")
            self.view.status_label.setText("Status: No LLM provider configured")
            return

        if self._semantic_worker and self._semantic_worker.isRunning():
            self._semantic_worker.cancel()
            self.view.status_label.setText("Status: Stopping semantic analysis...")
            return

        binary_hash = self.analysis_db.get_binary_hash(self.binary_view)
        self.view.status_label.setText("Status: Running semantic analysis...")
        self._set_semantic_button_running(True)
        self._semantic_worker = SemanticAnalysisWorker(
            provider_config,
            self.llm_factory,
            self.graph_service,
            self.binary_view,
            binary_hash,
        )
        self._semantic_worker.progress.connect(self._on_semantic_progress)
        self._semantic_worker.completed.connect(self._on_semantic_complete)
        self._semantic_worker.cancelled.connect(self._on_semantic_cancelled)
        self._semantic_worker.failed.connect(self._on_semantic_error)
        self._semantic_worker.start()

    def handle_security_analysis(self):
        if not self.binary_view:
            return
        if self._security_worker and self._security_worker.isRunning():
            self._security_worker.cancel()
            self.view.status_label.setText("Status: Stopping security analysis...")
            return
        binary_hash = self.analysis_db.get_binary_hash(self.binary_view)
        self.view.status_label.setText("Status: Running security analysis...")
        self._set_security_button_running(True)
        self._security_worker = SecurityAnalysisWorker(self.graph_store, binary_hash)
        self._security_worker.completed.connect(self._on_security_complete)
        self._security_worker.cancelled.connect(self._on_security_cancelled)
        self._security_worker.failed.connect(self._on_security_error)
        self._security_worker.start()

    def handle_refresh_names(self):
        if not self.binary_view:
            return
        binary_hash = self.analysis_db.get_binary_hash(self.binary_view)
        for function in self.binary_view.functions:
            node = self.graph_service.get_node_by_address(binary_hash, "FUNCTION", int(function.start))
            if node and node.name != function.name:
                node.name = function.name
                self.graph_service.upsert_node(node)
        self.refresh_current_view()

    def handle_save_summary(self, summary: str):
        if not self.binary_view:
            return
        function = self._get_function_at(self._current_offset)
        if not function:
            return
        binary_hash = self.analysis_db.get_binary_hash(self.binary_view)
        node = self.graph_service.get_node_by_address(binary_hash, "FUNCTION", int(function.start))
        if not node:
            return
        node.llm_summary = summary
        node.user_edited = True
        node.is_stale = False
        self.graph_service.upsert_node(node)
        self.refresh_current_view()

    def handle_add_flag(self, flag: str):
        if not self.binary_view:
            return
        function = self._get_function_at(self._current_offset)
        if not function:
            return
        binary_hash = self.analysis_db.get_binary_hash(self.binary_view)
        node = self.graph_service.get_node_by_address(binary_hash, "FUNCTION", int(function.start))
        if not node:
            return
        if flag not in node.security_flags:
            node.security_flags.append(flag)
            self.graph_service.upsert_node(node)
        self.refresh_current_view()

    def handle_remove_flag(self, flag: str):
        if not self.binary_view:
            return
        function = self._get_function_at(self._current_offset)
        if not function:
            return
        binary_hash = self.analysis_db.get_binary_hash(self.binary_view)
        node = self.graph_service.get_node_by_address(binary_hash, "FUNCTION", int(function.start))
        if not node:
            return
        if flag in node.security_flags:
            node.security_flags.remove(flag)
            self.graph_service.upsert_node(node)
        self.refresh_current_view()

    def handle_edge_click(self, target_id: str):
        if not target_id:
            return
        node = self.graph_store.get_node_by_id(str(target_id))
        if node and node.address is not None:
            self.handle_navigate(node.address)

    def handle_visual_refresh(self, n_hops: int, edge_types: List[str]):
        if not self.binary_view:
            return
        binary_hash = self.analysis_db.get_binary_hash(self.binary_view)
        function = self._get_function_at(self._current_offset)
        if not function:
            self.view.graph_view.show_not_indexed()
            return
        center = self.graph_service.get_node_by_address(binary_hash, "FUNCTION", int(function.start))
        if not center:
            self.view.graph_view.show_not_indexed()
            return

        nodes, edges = self._collect_graph(binary_hash, center.id, n_hops, set(edge_types))
        center_node = self._node_to_graph_node(center)
        self.view.graph_view.build_graph(center_node, nodes, edges)

    def handle_search_query(self, query_type: str, args: Dict[str, Any]):
        binary_hash = self.analysis_db.get_binary_hash(self.binary_view) if self.binary_view else None
        if not binary_hash:
            self.view.search_view.handle_query_result(json.dumps({"error": "No binary loaded"}))
            return

        try:
            result = self._execute_query(binary_hash, query_type, args)
            self.view.search_view.handle_query_result(json.dumps(result))
        except Exception as e:
            self.view.search_view.handle_query_result(json.dumps({"error": str(e)}))

    def _execute_query(self, binary_hash: str, query_type: str, args: Dict[str, Any]) -> Dict[str, Any]:
        engine = GraphRAGQueryEngine(self.graph_store, binary_hash)
        if query_type == "ga_search_semantic":
            results = engine.search_semantic(args.get("query", ""), args.get("limit", 20))
            return {"results": results}
        if query_type == "ga_get_semantic_analysis":
            address = self._resolve_address_arg(binary_hash, args)
            if address is None:
                return {"error": "Address required"}
            return engine.get_semantic_analysis(address)
        if query_type == "ga_get_similar_functions":
            address = self._resolve_address_arg(binary_hash, args)
            if address is None:
                return {"results": []}
            results = engine.get_similar_functions(address, args.get("limit", 10))
            return {"results": results}
        if query_type == "ga_get_call_context":
            address = self._resolve_address_arg(binary_hash, args)
            if address is None:
                return {"error": "Address required"}
            depth = int(args.get("depth", 1))
            direction = args.get("direction", "both")
            return engine.get_call_context(address, depth, direction)
        if query_type == "ga_get_security_analysis":
            scope = (args.get("scope") or "function").lower()
            if scope == "binary":
                return engine.get_binary_security_analysis()
            address = self._resolve_address_arg(binary_hash, args)
            if address is None:
                return {"error": "Address required"}
            return engine.get_security_analysis(address)
        if query_type == "ga_get_module_summary":
            address = self._resolve_address_arg(binary_hash, args) or 0
            return engine.get_module_summary(address)
        if query_type == "ga_get_activity_analysis":
            address = self._resolve_address_arg(binary_hash, args)
            if address is None:
                return {"error": "Address required"}
            return engine.get_activity_analysis(address)
        return {"error": f"Unknown query type: {query_type}"}

    def _node_from_args(self, binary_hash: str, args: Dict[str, Any]):
        addr = args.get("address")
        if not addr:
            function = self._get_function_at(self._current_offset)
            if not function:
                return None
            return self.graph_service.get_node_by_address(binary_hash, "FUNCTION", int(function.start))
        if isinstance(addr, str):
            address = int(addr, 16) if addr.startswith("0x") else int(addr, 16)
        else:
            address = int(addr)
        return self.graph_service.get_node_by_address(binary_hash, "FUNCTION", address)

    def _resolve_address_arg(self, binary_hash: str, args: Dict[str, Any]) -> Optional[int]:
        addr = args.get("address")
        if not addr:
            function = self._get_function_at(self._current_offset)
            return int(function.start) if function else None
        if isinstance(addr, str):
            return int(addr, 16) if addr.startswith("0x") else int(addr, 16)
        return int(addr)

    def _get_function_at(self, address: int):
        if not self.binary_view:
            return None
        fn = self.binary_view.get_function_at(address)
        if fn:
            return fn
        funcs = self.binary_view.get_functions_containing(address)
        return funcs[0] if funcs else None

    def _resolve_address(self, text: str) -> Optional[int]:
        if text.startswith("0x") or text.startswith("0X"):
            try:
                return int(text, 16)
            except ValueError:
                return None
        if text.isdigit():
            return int(text, 16)
        if not self.binary_view:
            return None
        for function in self.binary_view.functions:
            if function.name == text:
                return int(function.start)
        return None

    def _nodes_to_entries(self, nodes) -> List[Dict[str, Any]]:
        entries = []
        for node in nodes:
            entries.append({
                "name": node.name or f"0x{node.address:x}",
                "address": node.address or 0,
            })
        return entries

    def _edges_for_list(self, binary_hash: str, node_id: int) -> List[Dict[str, Any]]:
        edges = self.graph_store.get_edges_for_node(binary_hash, node_id)
        results = []
        for edge in edges:
            target_id = edge.target_id if edge.source_id == node_id else edge.source_id
            results.append({
                "type": edge.edge_type,
                "target": str(target_id),
                "weight": edge.weight or 1.0,
            })
        return results

    def _collect_graph(self, binary_hash: str, center_id: int, n_hops: int, edge_types: Set[str]):
        max_nodes = 50
        visited = {center_id}
        queue = [(center_id, 0)]
        all_edges = []

        while queue and len(visited) < max_nodes:
            node_id, depth = queue.pop(0)
            if depth >= n_hops:
                continue
            for edge in self.graph_store.get_edges_for_node(binary_hash, node_id):
                if edge_types and edge.edge_type not in edge_types:
                    continue
                all_edges.append(edge)
                neighbor = edge.target_id if edge.source_id == node_id else edge.source_id
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                if len(visited) >= max_nodes:
                    break
                queue.append((neighbor, depth + 1))
            if len(visited) >= max_nodes:
                break

        nodes = []
        for node_id in visited:
            node = self.graph_store.get_node_by_id(node_id)
            if node:
                nodes.append(self._node_to_graph_node(node))

        edge_rows = []
        for edge in all_edges:
            if edge.source_id not in visited or edge.target_id not in visited:
                continue
            edge_rows.append({
                "id": edge.id,
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "type": edge.edge_type,
                "weight": edge.weight or 1.0,
            })

        return nodes, edge_rows

    def _node_to_graph_node(self, node) -> Dict[str, Any]:
        label = node.name or f"0x{node.address:x}"
        return {
            "id": node.id,
            "address": node.address or 0,
            "label": label,
            "summary": node.llm_summary or "",
            "has_vuln": bool(node.security_flags),
        }

    def _node_to_result(self, node) -> Dict[str, Any]:
        if not node:
            return {}
        return {
            "function_name": node.name,
            "address": f"0x{node.address:x}" if node.address is not None else "",
            "summary": node.llm_summary or "",
            "security_flags": node.security_flags,
            "risk_level": node.risk_level,
            "activity_profile": node.activity_profile,
        }

    def _get_callees(self, binary_hash: str, node_id: int):
        edges = self.graph_store.get_edges_for_node(binary_hash, node_id)
        callees = []
        for edge in edges:
            if edge.source_id == node_id and edge.edge_type == "CALLS":
                node = self.graph_store.get_node_by_id(edge.target_id)
                if node:
                    callees.append(node)
        return callees

    def _on_semantic_progress(self, processed: int, total: int, summarized: int, errors: int):
        self.view.status_label.setText(
            f"Status: Summarizing... {processed}/{total} ({errors} errors)"
        )

    def _on_semantic_complete(self, summarized: int, errors: int, elapsed_ms: int):
        self.view.status_label.setText(
            f"Status: Semantic analysis complete ({summarized} summaries, {errors} errors)"
        )
        self._set_semantic_button_running(False)
        self.refresh_current_view()

    def _on_semantic_cancelled(self):
        self.view.status_label.setText("Status: Semantic analysis stopped")
        self._set_semantic_button_running(False)

    def _on_semantic_error(self, message: str):
        log.log_error(f"Semantic analysis failed: {message}")
        self.view.status_label.setText("Status: Semantic analysis failed")
        self._set_semantic_button_running(False)

    def _on_security_complete(self, path_count: int, edges_created: int):
        self.view.status_label.setText(
            f"Status: Security analysis complete ({path_count} paths, {edges_created} edges)"
        )
        self._set_security_button_running(False)
        self.refresh_current_view()

    def _on_security_cancelled(self):
        self.view.status_label.setText("Status: Security analysis stopped")
        self._set_security_button_running(False)

    def _on_security_error(self, message: str):
        log.log_error(f"Security analysis failed: {message}")
        self.view.status_label.setText("Status: Security analysis failed")
        self._set_security_button_running(False)

    def _on_reindex_progress(self, processed: int, total: int):
        self.view.status_label.setText(f"Status: Reindexing... {processed}/{total}")

    def _on_reindex_complete(self, processed: int):
        self.view.status_label.setText(f"Status: Reindex complete ({processed} functions)")
        self._set_reindex_button_running(False)
        self.refresh_current_view()

    def _on_reindex_cancelled(self, processed: int):
        self.view.status_label.setText(f"Status: Reindex stopped ({processed} functions)")
        self._set_reindex_button_running(False)
        self.refresh_current_view()

    def _on_reindex_error(self, message: str):
        log.log_error(f"Reindex failed: {message}")
        self.view.status_label.setText("Status: Reindex failed")
        self._set_reindex_button_running(False)

    def _set_reindex_button_running(self, running: bool):
        if hasattr(self.view, "reindex_button"):
            self.view.reindex_button.setText("Stop" if running else "ReIndex Binary")

    def _set_semantic_button_running(self, running: bool):
        if hasattr(self.view, "semantic_button"):
            self.view.semantic_button.setText("Stop" if running else "Semantic Analysis")

    def _set_security_button_running(self, running: bool):
        if hasattr(self.view, "security_button"):
            self.view.security_button.setText("Stop" if running else "Security Analysis")


class SemanticAnalysisWorker(QThread):
    progress = Signal(int, int, int, int)
    completed = Signal(int, int, int)
    cancelled = Signal()
    failed = Signal(str)

    def __init__(self, provider_config, llm_factory, graph_service,
                 binary_view, binary_hash: str, limit: int = 0):
        super().__init__()
        self.provider_config = provider_config
        self.llm_factory = llm_factory
        self.graph_service = graph_service
        self.binary_view = binary_view
        self.binary_hash = binary_hash
        self.limit = limit
        self._cancelled = False
        self._extractor = None

    def cancel(self):
        self._cancelled = True
        if self._extractor:
            self._extractor.cancel()

    def run(self):
        try:
            asyncio.run(self._async_run())
        except Exception as exc:
            self.failed.emit(str(exc))

    async def _async_run(self):
        provider = self.llm_factory.create_provider(self.provider_config)
        from ..services.graphrag.semantic_extractor import SemanticExtractor
        self._extractor = SemanticExtractor(
            provider,
            self.graph_service.store,
            self.binary_view,
            self.binary_hash,
        )
        result = await self._extractor.summarize_stale_nodes(
            self.limit,
            self._progress_callback,
        )
        if self._cancelled or (self._extractor and self._extractor.cancelled):
            self.cancelled.emit()
            return
        self.completed.emit(result.summarized, result.errors, result.elapsed_ms)

    def _progress_callback(self, processed: int, total: int, summarized: int, errors: int):
        self.progress.emit(processed, total, summarized, errors)


class SecurityAnalysisWorker(QThread):
    completed = Signal(int, int)
    cancelled = Signal()
    failed = Signal(str)

    def __init__(self, graph_store: GraphStore, binary_hash: str, max_paths: int = 100):
        super().__init__()
        self.graph_store = graph_store
        self.binary_hash = binary_hash
        self.max_paths = max_paths
        self._cancelled = False
        self._analyzer = None

    def cancel(self):
        self._cancelled = True
        if self._analyzer:
            self._analyzer.cancel()

    def run(self):
        try:
            self._analyzer = TaintAnalyzer(self.graph_store, self.binary_hash)
            paths = self._analyzer.find_taint_paths(self.max_paths, create_edges=True)
            if self._cancelled or (self._analyzer and self._analyzer.cancelled):
                self.cancelled.emit()
                return
            edges_created = len(paths)
            self.completed.emit(len(paths), edges_created)
        except Exception as exc:
            self.failed.emit(str(exc))


class ReindexWorker(QThread):
    progress = Signal(int, int)
    completed = Signal(int)
    cancelled = Signal(int)
    failed = Signal(str)

    def __init__(self, graph_service: GraphRAGService, graph_store: GraphStore,
                 binary_view, binary_hash: str):
        super().__init__()
        self.graph_service = graph_service
        self.graph_store = graph_store
        self.binary_view = binary_view
        self.binary_hash = binary_hash
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            self.graph_store.delete_graph(self.binary_hash)
            functions = list(self.binary_view.functions)
            total = len(functions)
            processed = 0
            for function in functions:
                if self._cancelled:
                    self.cancelled.emit(processed)
                    return
                self.graph_service.index_function(self.binary_view, function, self.binary_hash)
                processed += 1
                self.progress.emit(processed, total)
            self.completed.emit(processed)
        except Exception as exc:
            self.failed.emit(str(exc))
