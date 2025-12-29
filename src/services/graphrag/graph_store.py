#!/usr/bin/env python3

import json
from typing import Any, Dict, List, Optional

from .models import GraphNode, GraphEdge
from ..analysis_db_service import AnalysisDBService

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


class GraphStore:
    """SQLite-backed storage for GraphRAG nodes and edges."""

    def __init__(self, analysis_db: Optional[AnalysisDBService] = None):
        self.analysis_db = analysis_db or AnalysisDBService()
        self._db_lock = self.analysis_db.get_db_lock()

    @staticmethod
    def _serialize_list(values: Optional[List[str]]) -> str:
        if not values:
            return "[]"
        return json.dumps(list(values))

    @staticmethod
    def _deserialize_list(value: Optional[str]) -> List[str]:
        if not value:
            return []
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []

    def get_node_by_address(self, binary_hash: str, node_type: str, address: int) -> Optional[GraphNode]:
        if not binary_hash:
            return None
        with self._db_lock:
            conn = self.analysis_db.get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, binary_hash, node_type, address, name, raw_code, llm_summary,
                           security_flags, network_apis, file_io_apis, ip_addresses, urls,
                           file_paths, domains, registry_keys, activity_profile,
                           risk_level, is_stale, user_edited, created_at, updated_at
                    FROM GraphNodes
                    WHERE binary_hash = ? AND node_type = ? AND address = ?
                ''', (binary_hash, node_type, address))
                row = cursor.fetchone()
                if not row:
                    return None
                return GraphNode(
                    id=row[0],
                    binary_hash=row[1],
                    node_type=row[2],
                    address=row[3],
                    name=row[4],
                    raw_code=row[5],
                    llm_summary=row[6],
                    security_flags=self._deserialize_list(row[7]),
                    network_apis=self._deserialize_list(row[8]),
                    file_io_apis=self._deserialize_list(row[9]),
                    ip_addresses=self._deserialize_list(row[10]),
                    urls=self._deserialize_list(row[11]),
                    file_paths=self._deserialize_list(row[12]),
                    domains=self._deserialize_list(row[13]),
                    registry_keys=self._deserialize_list(row[14]),
                    activity_profile=row[15],
                    risk_level=row[16],
                    is_stale=bool(row[17]),
                    user_edited=bool(row[18]),
                    created_at=row[19],
                    updated_at=row[20],
                )
            finally:
                conn.close()

    def get_node_by_id(self, node_id: int) -> Optional[GraphNode]:
        if not node_id:
            return None
        with self._db_lock:
            conn = self.analysis_db.get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, binary_hash, node_type, address, name, raw_code, llm_summary,
                           security_flags, network_apis, file_io_apis, ip_addresses, urls,
                           file_paths, domains, registry_keys, activity_profile,
                           risk_level, is_stale, user_edited, created_at, updated_at
                    FROM GraphNodes
                    WHERE id = ?
                ''', (node_id,))
                row = cursor.fetchone()
                if not row:
                    return None
                return GraphNode(
                    id=row[0],
                    binary_hash=row[1],
                    node_type=row[2],
                    address=row[3],
                    name=row[4],
                    raw_code=row[5],
                    llm_summary=row[6],
                    security_flags=self._deserialize_list(row[7]),
                    network_apis=self._deserialize_list(row[8]),
                    file_io_apis=self._deserialize_list(row[9]),
                    ip_addresses=self._deserialize_list(row[10]),
                    urls=self._deserialize_list(row[11]),
                    file_paths=self._deserialize_list(row[12]),
                    domains=self._deserialize_list(row[13]),
                    registry_keys=self._deserialize_list(row[14]),
                    activity_profile=row[15],
                    risk_level=row[16],
                    is_stale=bool(row[17]),
                    user_edited=bool(row[18]),
                    created_at=row[19],
                    updated_at=row[20],
                )
            finally:
                conn.close()

    def upsert_node(self, node: GraphNode) -> GraphNode:
        if not node.binary_hash:
            raise ValueError("binary_hash is required for GraphNode upsert")
        if node.address is None:
            raise ValueError("address is required for GraphNode upsert")

        security_flags = self._serialize_list(node.security_flags)
        network_apis = self._serialize_list(node.network_apis)
        file_io_apis = self._serialize_list(node.file_io_apis)
        ip_addresses = self._serialize_list(node.ip_addresses)
        urls = self._serialize_list(node.urls)
        file_paths = self._serialize_list(node.file_paths)
        domains = self._serialize_list(node.domains)
        registry_keys = self._serialize_list(node.registry_keys)

        with self._db_lock:
            conn = self.analysis_db.get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO GraphNodes (
                        binary_hash, node_type, address, name, raw_code, llm_summary,
                        security_flags, network_apis, file_io_apis, ip_addresses, urls,
                        file_paths, domains, registry_keys, activity_profile,
                        risk_level, is_stale, user_edited, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(binary_hash, node_type, address) DO UPDATE SET
                        name = excluded.name,
                        raw_code = excluded.raw_code,
                        llm_summary = excluded.llm_summary,
                        security_flags = excluded.security_flags,
                        network_apis = excluded.network_apis,
                        file_io_apis = excluded.file_io_apis,
                        ip_addresses = excluded.ip_addresses,
                        urls = excluded.urls,
                        file_paths = excluded.file_paths,
                        domains = excluded.domains,
                        registry_keys = excluded.registry_keys,
                        activity_profile = excluded.activity_profile,
                        risk_level = excluded.risk_level,
                        is_stale = excluded.is_stale,
                        user_edited = excluded.user_edited,
                        updated_at = CURRENT_TIMESTAMP
                ''', (
                    node.binary_hash,
                    node.node_type,
                    node.address,
                    node.name,
                    node.raw_code,
                    node.llm_summary,
                    security_flags,
                    network_apis,
                    file_io_apis,
                    ip_addresses,
                    urls,
                    file_paths,
                    domains,
                    registry_keys,
                    node.activity_profile,
                    node.risk_level,
                    1 if node.is_stale else 0,
                    1 if node.user_edited else 0
                ))
                conn.commit()

                cursor.execute('''
                    SELECT id FROM GraphNodes
                    WHERE binary_hash = ? AND node_type = ? AND address = ?
                ''', (node.binary_hash, node.node_type, node.address))
                row = cursor.fetchone()
                if row:
                    node.id = row[0]
                return node
            finally:
                conn.close()

    def add_edge(self, edge: GraphEdge) -> GraphEdge:
        if not edge.binary_hash:
            raise ValueError("binary_hash is required for GraphEdge insert")
        if not edge.source_id or not edge.target_id or not edge.edge_type:
            raise ValueError("source_id, target_id, and edge_type are required for GraphEdge insert")

        with self._db_lock:
            conn = self.analysis_db.get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO GraphEdges (
                        binary_hash, source_id, target_id, edge_type, weight, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(binary_hash, source_id, target_id, edge_type) DO NOTHING
                ''', (
                    edge.binary_hash,
                    edge.source_id,
                    edge.target_id,
                    edge.edge_type,
                    edge.weight,
                    edge.metadata
                ))
                conn.commit()
                if cursor.lastrowid:
                    edge.id = cursor.lastrowid
                return edge
            finally:
                conn.close()

    def get_callers(self, binary_hash: str, node_id: int, edge_type: str = "CALLS") -> List[GraphNode]:
        if not binary_hash:
            return []
        with self._db_lock:
            conn = self.analysis_db.get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT n.id, n.binary_hash, n.node_type, n.address, n.name, n.raw_code,
                           n.llm_summary, n.security_flags, n.network_apis, n.file_io_apis,
                           n.ip_addresses, n.urls, n.file_paths, n.domains, n.registry_keys,
                           n.activity_profile, n.risk_level, n.is_stale, n.user_edited,
                           n.created_at, n.updated_at
                    FROM GraphEdges e
                    JOIN GraphNodes n ON n.id = e.source_id
                    WHERE e.binary_hash = ? AND e.target_id = ? AND e.edge_type = ?
                ''', (binary_hash, node_id, edge_type))
                rows = cursor.fetchall()
                callers = []
                for row in rows:
                    callers.append(GraphNode(
                        id=row[0],
                        binary_hash=row[1],
                        node_type=row[2],
                        address=row[3],
                        name=row[4],
                        raw_code=row[5],
                        llm_summary=row[6],
                        security_flags=self._deserialize_list(row[7]),
                        network_apis=self._deserialize_list(row[8]),
                        file_io_apis=self._deserialize_list(row[9]),
                        ip_addresses=self._deserialize_list(row[10]),
                        urls=self._deserialize_list(row[11]),
                        file_paths=self._deserialize_list(row[12]),
                        domains=self._deserialize_list(row[13]),
                        registry_keys=self._deserialize_list(row[14]),
                        activity_profile=row[15],
                        risk_level=row[16],
                        is_stale=bool(row[17]),
                        user_edited=bool(row[18]),
                        created_at=row[19],
                        updated_at=row[20],
                    ))
                return callers
            finally:
                conn.close()

    def get_edges_for_node(self, binary_hash: str, node_id: int) -> List[GraphEdge]:
        if not binary_hash or not node_id:
            return []
        with self._db_lock:
            conn = self.analysis_db.get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, binary_hash, source_id, target_id, edge_type, weight, metadata, created_at
                    FROM GraphEdges
                    WHERE binary_hash = ? AND (source_id = ? OR target_id = ?)
                ''', (binary_hash, node_id, node_id))
                rows = cursor.fetchall()
                edges = []
                for row in rows:
                    edges.append(GraphEdge(
                        id=row[0],
                        binary_hash=row[1],
                        source_id=row[2],
                        target_id=row[3],
                        edge_type=row[4],
                        weight=row[5] if row[5] is not None else 1.0,
                        metadata=row[6],
                        created_at=row[7],
                    ))
                return edges
            finally:
                conn.close()

    def get_graph_stats(self, binary_hash: str) -> Dict[str, Any]:
        if not binary_hash:
            return {"nodes": 0, "edges": 0, "stale": 0, "last_indexed": None}
        with self._db_lock:
            conn = self.analysis_db.get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM GraphNodes WHERE binary_hash = ?', (binary_hash,))
                nodes = cursor.fetchone()[0]
                cursor.execute('SELECT COUNT(*) FROM GraphEdges WHERE binary_hash = ?', (binary_hash,))
                edges = cursor.fetchone()[0]
                cursor.execute('SELECT COUNT(*) FROM GraphNodes WHERE binary_hash = ? AND is_stale = 1', (binary_hash,))
                stale = cursor.fetchone()[0]
                cursor.execute('SELECT MAX(updated_at) FROM GraphNodes WHERE binary_hash = ?', (binary_hash,))
                last = cursor.fetchone()[0]
                return {"nodes": nodes, "edges": edges, "stale": stale, "last_indexed": last}
            finally:
                conn.close()

    def delete_graph(self, binary_hash: str) -> None:
        if not binary_hash:
            return
        with self._db_lock:
            conn = self.analysis_db.get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM GraphEdges WHERE binary_hash = ?', (binary_hash,))
                cursor.execute('DELETE FROM GraphNodes WHERE binary_hash = ?', (binary_hash,))
                conn.commit()
            finally:
                conn.close()

    def search_nodes(self, binary_hash: str, query: str, limit: int = 10) -> List[GraphNode]:
        if not binary_hash or not query:
            return []

        query = query.strip()
        with self._db_lock:
            conn = self.analysis_db.get_connection()
            try:
                cursor = conn.cursor()
                try:
                    cursor.execute('''
                        SELECT n.id, n.binary_hash, n.node_type, n.address, n.name, n.raw_code,
                               n.llm_summary, n.security_flags, n.network_apis, n.file_io_apis,
                               n.ip_addresses, n.urls, n.file_paths, n.domains, n.registry_keys,
                               n.activity_profile, n.risk_level, n.is_stale, n.user_edited,
                               n.created_at, n.updated_at
                        FROM GraphNodeFTS f
                        JOIN GraphNodes n ON n.id = f.rowid
                        WHERE n.binary_hash = ? AND GraphNodeFTS MATCH ?
                        LIMIT ?
                    ''', (binary_hash, query, limit))
                except Exception:
                    cursor.execute('''
                        SELECT id, binary_hash, node_type, address, name, raw_code,
                               llm_summary, security_flags, network_apis, file_io_apis,
                               ip_addresses, urls, file_paths, domains, registry_keys,
                               activity_profile, risk_level, is_stale, user_edited,
                               created_at, updated_at
                        FROM GraphNodes
                        WHERE binary_hash = ? AND (name LIKE ? OR llm_summary LIKE ?)
                        LIMIT ?
                    ''', (binary_hash, f"%{query}%", f"%{query}%", limit))

                rows = cursor.fetchall()
                results = []
                for row in rows:
                    results.append(GraphNode(
                        id=row[0],
                        binary_hash=row[1],
                        node_type=row[2],
                        address=row[3],
                        name=row[4],
                        raw_code=row[5],
                        llm_summary=row[6],
                        security_flags=self._deserialize_list(row[7]),
                        network_apis=self._deserialize_list(row[8]),
                        file_io_apis=self._deserialize_list(row[9]),
                        ip_addresses=self._deserialize_list(row[10]),
                        urls=self._deserialize_list(row[11]),
                        file_paths=self._deserialize_list(row[12]),
                        domains=self._deserialize_list(row[13]),
                        registry_keys=self._deserialize_list(row[14]),
                        activity_profile=row[15],
                        risk_level=row[16],
                        is_stale=bool(row[17]),
                        user_edited=bool(row[18]),
                        created_at=row[19],
                        updated_at=row[20],
                    ))
                return results
            finally:
                conn.close()
