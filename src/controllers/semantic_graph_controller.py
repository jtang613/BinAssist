#!/usr/bin/env python3

import json
from typing import Any, Dict, List, Optional, Set

import binaryninja as bn

from ..services.analysis_db_service import AnalysisDBService
from ..services.graphrag.graphrag_service import GraphRAGService
from ..services.graphrag.graph_store import GraphStore

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
        self._current_offset = 0

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
        binary_hash = self.analysis_db.get_binary_hash(self.binary_view)
        self.graph_store.delete_graph(binary_hash)
        for function in self.binary_view.functions:
            self.graph_service.index_function(self.binary_view, function, binary_hash)
        self.refresh_current_view()

    def handle_reset(self):
        if not self.binary_view:
            return
        binary_hash = self.analysis_db.get_binary_hash(self.binary_view)
        self.graph_store.delete_graph(binary_hash)
        self.refresh_current_view()

    def handle_semantic_analysis(self):
        log.log_info("Semantic analysis not yet implemented for BinAssist.")

    def handle_security_analysis(self):
        log.log_info("Security analysis not yet implemented for BinAssist.")

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
        try:
            node_id = int(target_id)
        except Exception:
            return
        node = self.graph_store.get_node_by_id(node_id)
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
        if query_type == "ga_search_semantic":
            results = self.graph_store.search_nodes(binary_hash, args.get("query", ""), args.get("limit", 20))
            return {"results": [self._node_to_result(n) for n in results]}
        if query_type == "ga_get_semantic_analysis":
            node = self._node_from_args(binary_hash, args)
            return self._node_to_result(node) if node else {"error": "Not indexed"}
        if query_type == "ga_get_similar_functions":
            node = self._node_from_args(binary_hash, args)
            if not node:
                return {"results": []}
            results = self.graph_store.search_nodes(binary_hash, node.name or "", args.get("limit", 10))
            filtered = [self._node_to_result(n) for n in results if n.id != node.id]
            return {"results": filtered}
        if query_type == "ga_get_call_context":
            node = self._node_from_args(binary_hash, args)
            if not node:
                return {"error": "Not indexed"}
            callers = self.graph_store.get_callers(binary_hash, node.id, "CALLS")
            callees = self._get_callees(binary_hash, node.id)
            return {
                "function_name": node.name,
                "address": f"0x{node.address:x}",
                "summary": node.llm_summary,
                "callers": [self._node_to_result(n) for n in callers],
                "callees": [self._node_to_result(n) for n in callees],
            }
        if query_type == "ga_get_security_analysis":
            node = self._node_from_args(binary_hash, args)
            if not node:
                return {"error": "Not indexed"}
            return {
                "function_name": node.name,
                "address": f"0x{node.address:x}",
                "security_flags": node.security_flags,
                "risk_level": node.risk_level,
                "activity_profile": node.activity_profile,
            }
        if query_type == "ga_get_module_summary":
            return {"module_summary": "Community detection not yet available in BinAssist."}
        if query_type == "ga_get_activity_analysis":
            node = self._node_from_args(binary_hash, args)
            if not node:
                return {"error": "Not indexed"}
            return {
                "function_name": node.name,
                "address": f"0x{node.address:x}",
                "network_apis": node.network_apis,
                "file_io_apis": node.file_io_apis,
                "risk_level": node.risk_level,
                "activity_profile": node.activity_profile,
            }
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
        visited = {center_id}
        frontier = {center_id}
        all_edges = []
        for _ in range(n_hops):
            next_frontier = set()
            for node_id in frontier:
                for edge in self.graph_store.get_edges_for_node(binary_hash, node_id):
                    if edge_types and edge.edge_type not in edge_types:
                        continue
                    all_edges.append(edge)
                    neighbor = edge.target_id if edge.source_id == node_id else edge.source_id
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.add(neighbor)
            frontier = next_frontier

        nodes = []
        for node_id in visited:
            node = self.graph_store.get_node_by_id(node_id)
            if node:
                nodes.append(self._node_to_graph_node(node))

        edge_rows = []
        for edge in all_edges:
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
