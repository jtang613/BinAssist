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
from ..services.graphrag.network_flow_analyzer import NetworkFlowAnalyzer
from ..services.graphrag.query_engine import GraphRAGQueryEngine
from ..services.graphrag.community_detector import CommunityDetector
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
        self._network_worker = None
        self._reindex_worker = None
        self._community_worker = None

        self._connect_signals()

    def _connect_signals(self):
        self.view.go_requested.connect(self.handle_go)
        self.view.reset_requested.connect(self.handle_reset)
        self.view.reindex_requested.connect(self.handle_reindex)
        self.view.semantic_analysis_requested.connect(self.handle_semantic_analysis)
        self.view.security_analysis_requested.connect(self.handle_security_analysis)
        self.view.network_flow_requested.connect(self.handle_network_flow_analysis)
        self.view.community_detection_requested.connect(self.handle_community_detection)
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

        callers = self.graph_store.get_callers(binary_hash, node.id, "calls")
        callees = self._get_callees(binary_hash, node.id)
        self.view.update_status(True, len(callers), len(callees), len(node.security_flags))

        self.view.list_view.show_content()
        self.view.list_view.set_callers(self._nodes_to_entries(callers))
        self.view.list_view.set_callees(self._nodes_to_entries(callees))
        self.view.list_view.set_edges(self._edges_for_list(binary_hash, node.id))
        self.view.list_view.set_security_flags(node.security_flags)
        self.view.list_view.set_summary(node.llm_summary or "")

        self.view.graph_view.show_content()
        self.handle_visual_refresh(self.view.graph_view.n_hops.value(), ["calls", "references", "calls_vulnerable", "network_send", "network_recv"])

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
        if hasattr(self.view, "show_bottom_progress"):
            self.view.show_bottom_progress()
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
        if hasattr(self.view, "show_bottom_progress"):
            self.view.show_bottom_progress()

        # Get RAG/MCP/Force settings from the view if available
        rag_enabled = self.view.is_rag_enabled() if hasattr(self.view, 'is_rag_enabled') else False
        mcp_enabled = self.view.is_mcp_enabled() if hasattr(self.view, 'is_mcp_enabled') else False
        force_enabled = self.view.is_force_enabled() if hasattr(self.view, 'is_force_enabled') else False

        self._semantic_worker = SemanticAnalysisWorker(
            provider_config,
            self.llm_factory,
            self.graph_service,
            self.binary_view,
            binary_hash,
            settings_service=self.settings_service,
            rag_enabled=rag_enabled,
            mcp_enabled=mcp_enabled,
            force=force_enabled,
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
        if hasattr(self.view, "show_bottom_progress"):
            self.view.show_bottom_progress()
        self._security_worker = SecurityAnalysisWorker(self.graph_store, binary_hash)
        self._security_worker.completed.connect(self._on_security_complete)
        self._security_worker.cancelled.connect(self._on_security_cancelled)
        self._security_worker.failed.connect(self._on_security_error)
        self._security_worker.start()

    def handle_network_flow_analysis(self):
        if not self.binary_view:
            return
        if self._network_worker and self._network_worker.isRunning():
            self._network_worker.cancel()
            self.view.status_label.setText("Status: Stopping network flow analysis...")
            return
        binary_hash = self.analysis_db.get_binary_hash(self.binary_view)
        self.view.status_label.setText("Status: Running network flow analysis...")
        self._set_network_button_running(True)
        if hasattr(self.view, "show_bottom_progress"):
            self.view.show_bottom_progress()
        self._network_worker = NetworkFlowAnalysisWorker(self.graph_store, binary_hash)
        self._network_worker.progress.connect(self._on_network_progress)
        self._network_worker.completed.connect(self._on_network_complete)
        self._network_worker.cancelled.connect(self._on_network_cancelled)
        self._network_worker.failed.connect(self._on_network_error)
        self._network_worker.start()

    def handle_community_detection(self):
        """Handle standalone community detection request."""
        if not self.binary_view:
            return
        if self._community_worker and self._community_worker.isRunning():
            self._community_worker.cancel()
            self.view.status_label.setText("Status: Stopping community detection...")
            return
        self._start_community_detection(force=True)

    def _start_community_detection(self, force: bool = True):
        """Start community detection as part of reindex workflow."""
        if not self.binary_view:
            return
        if self._community_worker and self._community_worker.isRunning():
            return  # Already running
        binary_hash = self.analysis_db.get_binary_hash(self.binary_view)
        self._update_panel_status("Detecting communities...")
        self._set_community_button_running(True)
        if hasattr(self.view, "show_bottom_progress"):
            self.view.show_bottom_progress()
        self._community_worker = CommunityDetectionWorker(
            self.graph_store,
            binary_hash,
            min_size=2,
            force=force,
        )
        self._community_worker.progress.connect(self._on_community_progress)
        self._community_worker.completed.connect(self._on_community_complete)
        self._community_worker.cancelled.connect(self._on_community_cancelled)
        self._community_worker.failed.connect(self._on_community_error)
        self._community_worker.start()

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
        node.confidence = 0.95  # User-edited content has high confidence
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
            if edge.source_id == node_id and edge.edge_type == "calls":
                node = self.graph_store.get_node_by_id(edge.target_id)
                if node:
                    callees.append(node)
        return callees

    def _on_semantic_progress(self, processed: int, total: int, summarized: int, errors: int):
        message = f"Summarizing... {processed}/{total} ({errors} errors)"
        self.view.status_label.setText(f"Status: {message}")
        self._update_panel_progress(processed, total, message)

    def _on_semantic_complete(self, summarized: int, errors: int, elapsed_ms: int):
        self.view.status_label.setText(
            f"Status: Semantic analysis complete ({summarized} summaries, {errors} errors)"
        )
        self._set_semantic_button_running(False)
        self._reset_panel_progress()
        self.refresh_current_view()

    def _on_semantic_cancelled(self):
        self.view.status_label.setText("Status: Semantic analysis stopped")
        self._set_semantic_button_running(False)
        self._reset_panel_progress()

    def _on_semantic_error(self, message: str):
        log.log_error(f"Semantic analysis failed: {message}")
        self.view.status_label.setText("Status: Semantic analysis failed")
        self._set_semantic_button_running(False)
        self._reset_panel_progress()

    def _on_security_complete(self, path_count: int, edges_created: int, chain: bool = False):
        self.view.status_label.setText(
            f"Status: Security analysis complete ({path_count} paths, {edges_created} edges)"
        )
        self._set_security_button_running(False)
        if chain:
            # Chain to network flow analysis
            self._start_network_flow_chain()
        else:
            self._reset_panel_progress()
            self.refresh_current_view()

    def _on_security_cancelled(self):
        self.view.status_label.setText("Status: Security analysis stopped")
        self._set_security_button_running(False)
        self._reset_panel_progress()

    def _on_security_error(self, message: str):
        log.log_error(f"Security analysis failed: {message}")
        self.view.status_label.setText("Status: Security analysis failed")
        self._set_security_button_running(False)
        # Continue chain even on error
        if hasattr(self, '_chain_active') and self._chain_active:
            self._start_network_flow_chain()
        else:
            self._reset_panel_progress()

    def _on_network_progress(self, current: int, total: int, message: str):
        self.view.status_label.setText(f"Status: {message}")

    def _on_network_complete(self, send_edges: int, recv_edges: int, chain: bool = False):
        self.view.status_label.setText(
            f"Status: Network flow analysis complete ({send_edges} send, {recv_edges} recv edges)"
        )
        self._set_network_button_running(False)
        if chain:
            # Chain to community detection
            self._start_community_detection(force=True)
        else:
            self._reset_panel_progress()
            self.refresh_current_view()

    def _on_network_cancelled(self):
        self.view.status_label.setText("Status: Network flow analysis stopped")
        self._set_network_button_running(False)
        self._reset_panel_progress()

    def _on_network_error(self, message: str):
        log.log_error(f"Network flow analysis failed: {message}")
        self.view.status_label.setText("Status: Network flow analysis failed")
        self._set_network_button_running(False)
        # Continue chain even on error
        if hasattr(self, '_chain_active') and self._chain_active:
            self._start_community_detection(force=True)
        else:
            self._reset_panel_progress()

    def _on_community_progress(self, iteration: int, max_iterations: int):
        self.view.status_label.setText(
            f"Status: Detecting communities... iteration {iteration}/{max_iterations}"
        )

    def _on_community_complete(self, count: int):
        if hasattr(self, '_chain_active') and self._chain_active:
            self._update_panel_status(f"Full pipeline complete ({count} communities detected)")
            self._update_panel_progress(100, 100, "Complete")
            self._chain_active = False
        else:
            self._update_panel_status(f"Community detection complete ({count} communities detected)")
        self._set_community_button_running(False)
        self._reset_panel_progress()
        self.refresh_current_view()

    def _on_community_cancelled(self):
        self._update_panel_status("Community detection stopped")
        self._set_community_button_running(False)
        self._reset_panel_progress()

    def _on_community_error(self, message: str):
        log.log_error(f"Community detection failed: {message}")
        self._update_panel_status("Community detection failed")
        self._set_community_button_running(False)
        self._reset_panel_progress()

    def _on_reindex_progress(self, processed: int, total: int, message: str):
        self._update_panel_status(f"Reindexing: {message}")
        self._update_panel_progress(processed, total, message)

    def _on_reindex_complete(self, processed: int):
        self._update_panel_status(f"Reindex complete ({processed} functions), running security analysis...")
        self._set_reindex_button_running(False)
        self.refresh_current_view()
        # Auto-trigger chained analysis: Security → Network → Community
        self._start_security_chain()

    def _on_reindex_cancelled(self, processed: int):
        self.view.status_label.setText(f"Status: Reindex stopped ({processed} functions)")
        self._set_reindex_button_running(False)
        self._reset_panel_progress()
        self.refresh_current_view()

    def _on_reindex_error(self, message: str):
        log.log_error(f"Reindex failed: {message}")
        self.view.status_label.setText("Status: Reindex failed")
        self._set_reindex_button_running(False)
        self._reset_panel_progress()

    def _set_reindex_button_running(self, running: bool):
        if hasattr(self.view, "manual_analysis_view"):
            self.view.manual_analysis_view.set_reindex_running(running)
        if hasattr(self.view, "reindex_button"):
            self.view.reindex_button.setText("Stop" if running else "ReIndex Binary")

    def _set_semantic_button_running(self, running: bool):
        if hasattr(self.view, "manual_analysis_view"):
            self.view.manual_analysis_view.set_semantic_running(running)
        if hasattr(self.view, "semantic_button"):
            self.view.semantic_button.setText("Stop" if running else "Semantic Analysis")

    def _set_security_button_running(self, running: bool):
        if hasattr(self.view, "manual_analysis_view"):
            self.view.manual_analysis_view.set_security_running(running)

    def _set_network_button_running(self, running: bool):
        if hasattr(self.view, "manual_analysis_view"):
            self.view.manual_analysis_view.set_network_running(running)

    def _set_community_button_running(self, running: bool):
        if hasattr(self.view, "manual_analysis_view"):
            self.view.manual_analysis_view.set_community_running(running)

    def _update_panel_status(self, message: str):
        """Update both the main status label and the Manual Analysis Panel status."""
        self.view.status_label.setText(f"Status: {message}")
        if hasattr(self.view, "manual_analysis_view"):
            self.view.manual_analysis_view.set_status(message)

    def _update_panel_progress(self, current: int, total: int, message: str = ""):
        """Update both the Manual Analysis Panel and bottom row progress bars."""
        if hasattr(self.view, "manual_analysis_view"):
            self.view.manual_analysis_view.set_progress(current, total, message)
        if hasattr(self.view, "set_bottom_progress"):
            self.view.set_bottom_progress(current, total, message)

    def _reset_panel_progress(self):
        """Reset both progress bars."""
        if hasattr(self.view, "manual_analysis_view"):
            self.view.manual_analysis_view.reset_progress()
        if hasattr(self.view, "hide_bottom_progress"):
            self.view.hide_bottom_progress()

    def _start_security_chain(self):
        """Start security analysis as part of reindex chain."""
        if not self.binary_view:
            self._chain_active = False
            return
        self._chain_active = True
        binary_hash = self.analysis_db.get_binary_hash(self.binary_view)
        self.view.status_label.setText("Status: Running security analysis...")
        self._security_worker = SecurityAnalysisWorker(self.graph_store, binary_hash, chain=True)
        self._security_worker.completed.connect(lambda p, e: self._on_security_complete(p, e, chain=True))
        self._security_worker.cancelled.connect(self._on_security_cancelled)
        self._security_worker.failed.connect(self._on_security_error)
        self._security_worker.start()

    def _start_network_flow_chain(self):
        """Start network flow analysis as part of reindex chain."""
        if not self.binary_view:
            self._chain_active = False
            return
        binary_hash = self.analysis_db.get_binary_hash(self.binary_view)
        self.view.status_label.setText("Status: Running network flow analysis...")
        self._network_worker = NetworkFlowAnalysisWorker(self.graph_store, binary_hash, chain=True)
        self._network_worker.progress.connect(self._on_network_progress)
        self._network_worker.completed.connect(lambda s, r: self._on_network_complete(s, r, chain=True))
        self._network_worker.cancelled.connect(self._on_network_cancelled)
        self._network_worker.failed.connect(self._on_network_error)
        self._network_worker.start()


class SemanticAnalysisWorker(QThread):
    progress = Signal(int, int, int, int)
    completed = Signal(int, int, int)
    cancelled = Signal()
    failed = Signal(str)

    def __init__(self, provider_config, llm_factory, graph_service,
                 binary_view, binary_hash: str, limit: int = 0,
                 settings_service=None, rag_enabled: bool = False,
                 mcp_enabled: bool = False, force: bool = False):
        super().__init__()
        self.provider_config = provider_config
        self.llm_factory = llm_factory
        self.graph_service = graph_service
        self.binary_view = binary_view
        self.binary_hash = binary_hash
        self.limit = limit
        self.settings_service = settings_service
        self.rag_enabled = rag_enabled
        self.mcp_enabled = mcp_enabled
        self.force = force
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

        # Create FunctionSummaryService for unified prompt generation
        from ..services.function_summary_service import FunctionSummaryService
        summary_service = FunctionSummaryService(
            self.binary_view,
            settings_service=self.settings_service,
            rag_enabled=self.rag_enabled,
            mcp_enabled=self.mcp_enabled,
        )

        from ..services.graphrag.semantic_extractor import SemanticExtractor
        self._extractor = SemanticExtractor(
            provider,
            self.graph_service.store,
            self.binary_view,
            self.binary_hash,
            summary_service=summary_service,
            rag_enabled=self.rag_enabled,
            mcp_enabled=self.mcp_enabled,
        )
        result = await self._extractor.summarize_stale_nodes(
            self.limit,
            self._progress_callback,
            force=self.force,
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

    def __init__(self, graph_store: GraphStore, binary_hash: str, max_paths: int = 100, chain: bool = False):
        super().__init__()
        self.graph_store = graph_store
        self.binary_hash = binary_hash
        self.max_paths = max_paths
        self.chain = chain
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
    progress = Signal(int, int, str)  # current, total, message
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
        self._extractor = None

    def cancel(self):
        self._cancelled = True
        if self._extractor:
            self._extractor.cancel()

    def run(self):
        try:
            from ..services.graphrag.structure_extractor import StructureExtractor

            # Non-destructive reindex: do NOT delete_graph() here.
            # The upsert logic will update changed fields while preserving semantic data.
            # Use "Reset Graph" if a full wipe is needed.

            # Use two-phase extraction with progress on both phases
            self._extractor = StructureExtractor(self.binary_view, self.graph_store)

            def progress_callback(current: int, total: int, message: str):
                if not self._cancelled:
                    self.progress.emit(current, total, message)

            result = self._extractor.extract_all_parallel(
                self.binary_hash,
                progress_callback=progress_callback
            )

            if self._cancelled:
                self.cancelled.emit(result.functions_extracted)
                return

            self.completed.emit(result.functions_extracted)
        except Exception as exc:
            self.failed.emit(str(exc))


class CommunityDetectionWorker(QThread):
    """Worker thread for community detection using Label Propagation."""
    progress = Signal(int, int)  # iteration, max_iterations
    completed = Signal(int)       # communities_detected
    cancelled = Signal()
    failed = Signal(str)

    def __init__(self, graph_store: GraphStore, binary_hash: str,
                 min_size: int = 2, force: bool = False):
        super().__init__()
        self.graph_store = graph_store
        self.binary_hash = binary_hash
        self.min_size = min_size
        self.force = force
        self._cancelled = False
        self._detector = None

    def cancel(self):
        self._cancelled = True
        if self._detector:
            self._detector.cancel()

    def run(self):
        try:
            self._detector = CommunityDetector(self.graph_store, self.binary_hash)
            count = self._detector.detect_communities(
                min_size=self.min_size,
                force=self.force,
                progress_callback=self._progress_callback,
            )
            if self._cancelled or (self._detector and self._detector._cancelled):
                self.cancelled.emit()
                return
            self.completed.emit(count)
        except Exception as exc:
            self.failed.emit(str(exc))

    def _progress_callback(self, iteration: int, max_iterations: int):
        self.progress.emit(iteration, max_iterations)


class NetworkFlowAnalysisWorker(QThread):
    """Worker thread for network flow analysis."""
    progress = Signal(int, int, str)  # current, total, message
    completed = Signal(int, int)       # send_edges, recv_edges
    cancelled = Signal()
    failed = Signal(str)

    def __init__(self, graph_store: GraphStore, binary_hash: str, chain: bool = False):
        super().__init__()
        self.graph_store = graph_store
        self.binary_hash = binary_hash
        self.chain = chain
        self._cancelled = False
        self._analyzer = None

    def cancel(self):
        self._cancelled = True
        if self._analyzer:
            self._analyzer.cancel()

    def run(self):
        try:
            self._analyzer = NetworkFlowAnalyzer(self.graph_store, self.binary_hash)
            result = self._analyzer.analyze(progress_callback=self._progress_callback)
            if self._cancelled or (self._analyzer and self._analyzer.cancelled):
                self.cancelled.emit()
                return
            self.completed.emit(result.total_send_edges, result.total_recv_edges)
        except Exception as exc:
            self.failed.emit(str(exc))

    def _progress_callback(self, current: int, total: int, message: str):
        self.progress.emit(current, total, message)
