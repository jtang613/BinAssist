#!/usr/bin/env python3

import json
import time
import uuid
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

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    def get_node_by_address(self, binary_hash: str, node_type: str, address: int) -> Optional[GraphNode]:
        if not binary_hash:
            return None
        with self._db_lock:
            conn = self.analysis_db.get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, binary_id, type, address, name, raw_content, llm_summary,
                           confidence, embedding, security_flags, network_apis, file_io_apis,
                           ip_addresses, urls, file_paths, domains, registry_keys, risk_level,
                           activity_profile, analysis_depth, created_at, updated_at,
                           is_stale, user_edited
                    FROM graph_nodes
                    WHERE binary_id = ? AND type = ? AND address = ?
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
                    confidence=row[7] if row[7] is not None else 0.0,
                    embedding=row[8],
                    security_flags=self._deserialize_list(row[9]),
                    network_apis=self._deserialize_list(row[10]),
                    file_io_apis=self._deserialize_list(row[11]),
                    ip_addresses=self._deserialize_list(row[12]),
                    urls=self._deserialize_list(row[13]),
                    file_paths=self._deserialize_list(row[14]),
                    domains=self._deserialize_list(row[15]),
                    registry_keys=self._deserialize_list(row[16]),
                    risk_level=row[17],
                    activity_profile=row[18],
                    analysis_depth=row[19] if row[19] is not None else 0,
                    created_at=row[20],
                    updated_at=row[21],
                    is_stale=bool(row[22]),
                    user_edited=bool(row[23]),
                )
            finally:
                conn.close()

    def get_node_by_id(self, node_id: str) -> Optional[GraphNode]:
        if not node_id:
            return None
        with self._db_lock:
            conn = self.analysis_db.get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, binary_id, type, address, name, raw_content, llm_summary,
                           confidence, embedding, security_flags, network_apis, file_io_apis,
                           ip_addresses, urls, file_paths, domains, registry_keys, risk_level,
                           activity_profile, analysis_depth, created_at, updated_at,
                           is_stale, user_edited
                    FROM graph_nodes
                    WHERE id = ?
                ''', (str(node_id),))
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
                    confidence=row[7] if row[7] is not None else 0.0,
                    embedding=row[8],
                    security_flags=self._deserialize_list(row[9]),
                    network_apis=self._deserialize_list(row[10]),
                    file_io_apis=self._deserialize_list(row[11]),
                    ip_addresses=self._deserialize_list(row[12]),
                    urls=self._deserialize_list(row[13]),
                    file_paths=self._deserialize_list(row[14]),
                    domains=self._deserialize_list(row[15]),
                    registry_keys=self._deserialize_list(row[16]),
                    risk_level=row[17],
                    activity_profile=row[18],
                    analysis_depth=row[19] if row[19] is not None else 0,
                    created_at=row[20],
                    updated_at=row[21],
                    is_stale=bool(row[22]),
                    user_edited=bool(row[23]),
                )
            finally:
                conn.close()

    def upsert_node(self, node: GraphNode) -> GraphNode:
        if not node.binary_hash:
            raise ValueError("binary_hash is required for GraphNode upsert")
        if node.address is None:
            raise ValueError("address is required for GraphNode upsert")

        if not node.id:
            existing = self.get_node_by_address(node.binary_hash, node.node_type, node.address)
            node.id = existing.id if existing else str(uuid.uuid4())

        now_ms = self._now_ms()
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
                    INSERT INTO graph_nodes (
                        id, type, address, binary_id, name, raw_content, llm_summary,
                        confidence, embedding, security_flags, network_apis, file_io_apis,
                        ip_addresses, urls, file_paths, domains, registry_keys, risk_level,
                        activity_profile, analysis_depth, created_at, updated_at, is_stale,
                        user_edited
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        type = excluded.type,
                        address = excluded.address,
                        binary_id = excluded.binary_id,
                        name = excluded.name,
                        raw_content = excluded.raw_content,
                        llm_summary = excluded.llm_summary,
                        confidence = excluded.confidence,
                        embedding = excluded.embedding,
                        security_flags = excluded.security_flags,
                        network_apis = excluded.network_apis,
                        file_io_apis = excluded.file_io_apis,
                        ip_addresses = excluded.ip_addresses,
                        urls = excluded.urls,
                        file_paths = excluded.file_paths,
                        domains = excluded.domains,
                        registry_keys = excluded.registry_keys,
                        risk_level = excluded.risk_level,
                        activity_profile = excluded.activity_profile,
                        analysis_depth = excluded.analysis_depth,
                        updated_at = excluded.updated_at,
                        is_stale = excluded.is_stale,
                        user_edited = excluded.user_edited
                ''', (
                    node.id,
                    node.node_type,
                    node.address,
                    node.binary_hash,
                    node.name,
                    node.raw_code,
                    node.llm_summary,
                    node.confidence,
                    node.embedding,
                    security_flags,
                    network_apis,
                    file_io_apis,
                    ip_addresses,
                    urls,
                    file_paths,
                    domains,
                    registry_keys,
                    node.risk_level,
                    node.activity_profile,
                    node.analysis_depth,
                    now_ms,
                    now_ms,
                    1 if node.is_stale else 0,
                    1 if node.user_edited else 0
                ))
                conn.commit()
                return node
            finally:
                conn.close()

    def add_edge(self, edge: GraphEdge) -> GraphEdge:
        if not edge.binary_hash:
            raise ValueError("binary_hash is required for GraphEdge insert")
        if not edge.source_id or not edge.target_id or not edge.edge_type:
            raise ValueError("source_id, target_id, and edge_type are required for GraphEdge insert")
        if not edge.id:
            edge.id = str(uuid.uuid4())

        now_ms = self._now_ms()
        with self._db_lock:
            conn = self.analysis_db.get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT 1 FROM graph_edges
                    WHERE source_id = ? AND target_id = ? AND type = ?
                    LIMIT 1
                ''', (edge.source_id, edge.target_id, edge.edge_type))
                if cursor.fetchone():
                    return edge
                cursor.execute('''
                    INSERT INTO graph_edges (
                        id, source_id, target_id, type, weight, metadata, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    edge.id,
                    edge.source_id,
                    edge.target_id,
                    edge.edge_type,
                    edge.weight,
                    edge.metadata,
                    now_ms
                ))
                conn.commit()
                return edge
            finally:
                conn.close()

    def get_callers(self, binary_hash: str, node_id: str, edge_type: str = "CALLS") -> List[GraphNode]:
        if not binary_hash or not node_id:
            return []
        with self._db_lock:
            conn = self.analysis_db.get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT n.id, n.binary_id, n.type, n.address, n.name, n.raw_content,
                           n.llm_summary, n.confidence, n.embedding, n.security_flags,
                           n.network_apis, n.file_io_apis, n.ip_addresses, n.urls,
                           n.file_paths, n.domains, n.registry_keys, n.risk_level,
                           n.activity_profile, n.analysis_depth, n.created_at, n.updated_at,
                           n.is_stale, n.user_edited
                    FROM graph_edges e
                    JOIN graph_nodes n ON n.id = e.source_id
                    JOIN graph_nodes t ON t.id = e.target_id
                    WHERE t.id = ? AND e.type = ? AND n.binary_id = ? AND t.binary_id = ?
                ''', (str(node_id), edge_type, binary_hash, binary_hash))
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
                        confidence=row[7] if row[7] is not None else 0.0,
                        embedding=row[8],
                        security_flags=self._deserialize_list(row[9]),
                        network_apis=self._deserialize_list(row[10]),
                        file_io_apis=self._deserialize_list(row[11]),
                        ip_addresses=self._deserialize_list(row[12]),
                        urls=self._deserialize_list(row[13]),
                        file_paths=self._deserialize_list(row[14]),
                        domains=self._deserialize_list(row[15]),
                        registry_keys=self._deserialize_list(row[16]),
                        risk_level=row[17],
                        activity_profile=row[18],
                        analysis_depth=row[19] if row[19] is not None else 0,
                        created_at=row[20],
                        updated_at=row[21],
                        is_stale=bool(row[22]),
                        user_edited=bool(row[23]),
                    ))
                return callers
            finally:
                conn.close()

    def get_edges_for_node(self, binary_hash: str, node_id: str) -> List[GraphEdge]:
        if not binary_hash or not node_id:
            return []
        with self._db_lock:
            conn = self.analysis_db.get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT e.id, e.source_id, e.target_id, e.type, e.weight, e.metadata, e.created_at
                    FROM graph_edges e
                    JOIN graph_nodes s ON s.id = e.source_id
                    JOIN graph_nodes t ON t.id = e.target_id
                    WHERE (e.source_id = ? OR e.target_id = ?)
                      AND s.binary_id = ? AND t.binary_id = ?
                ''', (str(node_id), str(node_id), binary_hash, binary_hash))
                rows = cursor.fetchall()
                edges = []
                for row in rows:
                    edges.append(GraphEdge(
                        id=row[0],
                        binary_hash=binary_hash,
                        source_id=row[1],
                        target_id=row[2],
                        edge_type=row[3],
                        weight=row[4] if row[4] is not None else 1.0,
                        metadata=row[5],
                        created_at=row[6],
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
                cursor.execute('SELECT COUNT(*) FROM graph_nodes WHERE binary_id = ?', (binary_hash,))
                nodes = cursor.fetchone()[0]
                cursor.execute('''
                    SELECT COUNT(*)
                    FROM graph_edges e
                    JOIN graph_nodes s ON s.id = e.source_id
                    JOIN graph_nodes t ON t.id = e.target_id
                    WHERE s.binary_id = ? AND t.binary_id = ?
                ''', (binary_hash, binary_hash))
                edges = cursor.fetchone()[0]
                cursor.execute('SELECT COUNT(*) FROM graph_nodes WHERE binary_id = ? AND is_stale = 1', (binary_hash,))
                stale = cursor.fetchone()[0]
                cursor.execute('SELECT MAX(updated_at) FROM graph_nodes WHERE binary_id = ?', (binary_hash,))
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
                cursor.execute('DELETE FROM graph_nodes WHERE binary_id = ?', (binary_hash,))
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
                        SELECT n.id, n.binary_id, n.type, n.address, n.name, n.raw_content,
                               n.llm_summary, n.confidence, n.embedding, n.security_flags,
                               n.network_apis, n.file_io_apis, n.ip_addresses, n.urls,
                               n.file_paths, n.domains, n.registry_keys, n.risk_level,
                               n.activity_profile, n.analysis_depth, n.created_at, n.updated_at,
                               n.is_stale, n.user_edited
                        FROM node_fts f
                        JOIN graph_nodes n ON n.id = f.id
                        WHERE n.binary_id = ? AND node_fts MATCH ?
                        LIMIT ?
                    ''', (binary_hash, query, limit))
                except Exception:
                    cursor.execute('''
                        SELECT id, binary_id, type, address, name, raw_content,
                               llm_summary, confidence, embedding, security_flags,
                               network_apis, file_io_apis, ip_addresses, urls,
                               file_paths, domains, registry_keys, risk_level,
                               activity_profile, analysis_depth, created_at, updated_at,
                               is_stale, user_edited
                        FROM graph_nodes
                        WHERE binary_id = ? AND (name LIKE ? OR llm_summary LIKE ?)
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
                        confidence=row[7] if row[7] is not None else 0.0,
                        embedding=row[8],
                        security_flags=self._deserialize_list(row[9]),
                        network_apis=self._deserialize_list(row[10]),
                        file_io_apis=self._deserialize_list(row[11]),
                        ip_addresses=self._deserialize_list(row[12]),
                        urls=self._deserialize_list(row[13]),
                        file_paths=self._deserialize_list(row[14]),
                        domains=self._deserialize_list(row[15]),
                        registry_keys=self._deserialize_list(row[16]),
                        risk_level=row[17],
                        activity_profile=row[18],
                        analysis_depth=row[19] if row[19] is not None else 0,
                        created_at=row[20],
                        updated_at=row[21],
                        is_stale=bool(row[22]),
                        user_edited=bool(row[23]),
                    ))
                return results
            finally:
                conn.close()

    def get_stale_nodes(self, binary_hash: str, limit: int = 0) -> List[GraphNode]:
        if not binary_hash:
            return []
        limit_value = limit if limit > 0 else 1000000
        with self._db_lock:
            conn = self.analysis_db.get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, binary_id, type, address, name, raw_content,
                           llm_summary, confidence, embedding, security_flags,
                           network_apis, file_io_apis, ip_addresses, urls,
                           file_paths, domains, registry_keys, risk_level,
                           activity_profile, analysis_depth, created_at, updated_at,
                           is_stale, user_edited
                    FROM graph_nodes
                    WHERE binary_id = ?
                      AND (is_stale = 1 OR llm_summary IS NULL OR llm_summary = '')
                    LIMIT ?
                ''', (binary_hash, limit_value))
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
                        confidence=row[7] if row[7] is not None else 0.0,
                        embedding=row[8],
                        security_flags=self._deserialize_list(row[9]),
                        network_apis=self._deserialize_list(row[10]),
                        file_io_apis=self._deserialize_list(row[11]),
                        ip_addresses=self._deserialize_list(row[12]),
                        urls=self._deserialize_list(row[13]),
                        file_paths=self._deserialize_list(row[14]),
                        domains=self._deserialize_list(row[15]),
                        registry_keys=self._deserialize_list(row[16]),
                        risk_level=row[17],
                        activity_profile=row[18],
                        analysis_depth=row[19] if row[19] is not None else 0,
                        created_at=row[20],
                        updated_at=row[21],
                        is_stale=bool(row[22]),
                        user_edited=bool(row[23]),
                    ))
                return results
            finally:
                conn.close()

    def get_nodes_by_type(self, binary_hash: str, node_type: str = "FUNCTION") -> List[GraphNode]:
        if not binary_hash:
            return []
        with self._db_lock:
            conn = self.analysis_db.get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, binary_id, type, address, name, raw_content,
                           llm_summary, confidence, embedding, security_flags,
                           network_apis, file_io_apis, ip_addresses, urls,
                           file_paths, domains, registry_keys, risk_level,
                           activity_profile, analysis_depth, created_at, updated_at,
                           is_stale, user_edited
                    FROM graph_nodes
                    WHERE binary_id = ? AND type = ?
                ''', (binary_hash, node_type))
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
                        confidence=row[7] if row[7] is not None else 0.0,
                        embedding=row[8],
                        security_flags=self._deserialize_list(row[9]),
                        network_apis=self._deserialize_list(row[10]),
                        file_io_apis=self._deserialize_list(row[11]),
                        ip_addresses=self._deserialize_list(row[12]),
                        urls=self._deserialize_list(row[13]),
                        file_paths=self._deserialize_list(row[14]),
                        domains=self._deserialize_list(row[15]),
                        registry_keys=self._deserialize_list(row[16]),
                        risk_level=row[17],
                        activity_profile=row[18],
                        analysis_depth=row[19] if row[19] is not None else 0,
                        created_at=row[20],
                        updated_at=row[21],
                        is_stale=bool(row[22]),
                        user_edited=bool(row[23]),
                    ))
                return results
            finally:
                conn.close()

    # ==================== Community Detection Methods ====================

    def get_edges_by_types(self, binary_hash: str, edge_types: List[str]) -> List[GraphEdge]:
        """Get all edges of specified types for a binary (for community detection)."""
        if not binary_hash or not edge_types:
            return []
        with self._db_lock:
            conn = self.analysis_db.get_connection()
            try:
                cursor = conn.cursor()
                placeholders = ','.join('?' * len(edge_types))
                cursor.execute(f'''
                    SELECT e.id, e.source_id, e.target_id, e.type, e.weight, e.metadata, e.created_at
                    FROM graph_edges e
                    JOIN graph_nodes s ON s.id = e.source_id
                    JOIN graph_nodes t ON t.id = e.target_id
                    WHERE s.binary_id = ? AND t.binary_id = ? AND e.type IN ({placeholders})
                ''', (binary_hash, binary_hash, *edge_types))
                rows = cursor.fetchall()
                edges = []
                for row in rows:
                    edges.append(GraphEdge(
                        id=row[0],
                        binary_hash=binary_hash,
                        source_id=row[1],
                        target_id=row[2],
                        edge_type=row[3],
                        weight=row[4] if row[4] is not None else 1.0,
                        metadata=row[5],
                        created_at=row[6],
                    ))
                return edges
            finally:
                conn.close()

    def communities_exist(self, binary_hash: str) -> bool:
        """Check if communities have been detected for a binary."""
        if not binary_hash:
            return False
        with self._db_lock:
            conn = self.analysis_db.get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    'SELECT COUNT(*) FROM graph_communities WHERE binary_id = ?',
                    (binary_hash,)
                )
                count = cursor.fetchone()[0]
                return count > 0
            finally:
                conn.close()

    def delete_communities(self, binary_hash: str) -> int:
        """Delete all communities for a binary (for re-detection). Returns count deleted."""
        if not binary_hash:
            return 0
        with self._db_lock:
            conn = self.analysis_db.get_connection()
            try:
                cursor = conn.cursor()
                # Get count before delete
                cursor.execute(
                    'SELECT COUNT(*) FROM graph_communities WHERE binary_id = ?',
                    (binary_hash,)
                )
                count = cursor.fetchone()[0]
                # Delete communities (CASCADE will delete community_members)
                cursor.execute(
                    'DELETE FROM graph_communities WHERE binary_id = ?',
                    (binary_hash,)
                )
                conn.commit()
                return count
            finally:
                conn.close()

    def save_community(self, binary_hash: str, community: Dict[str, Any]) -> str:
        """Save a community to the database. Returns community_id."""
        if not binary_hash:
            raise ValueError("binary_hash is required for community save")

        community_id = community.get('id') or str(uuid.uuid4())
        now_ms = self._now_ms()

        with self._db_lock:
            conn = self.analysis_db.get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO graph_communities (
                        id, level, binary_id, parent_community_id, name, summary,
                        member_count, is_stale, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        name = excluded.name,
                        summary = excluded.summary,
                        member_count = excluded.member_count,
                        is_stale = excluded.is_stale,
                        updated_at = excluded.updated_at
                ''', (
                    community_id,
                    community.get('level', 0),
                    binary_hash,
                    community.get('parent_community_id'),
                    community.get('name', 'Unknown Module'),
                    community.get('summary', ''),
                    community.get('member_count', 0),
                    1 if community.get('is_stale', False) else 0,
                    now_ms,
                    now_ms,
                ))
                conn.commit()
                return community_id
            finally:
                conn.close()

    def add_community_member(self, community_id: str, node_id: str, score: float = 1.0) -> None:
        """Add a node as a member of a community."""
        if not community_id or not node_id:
            return
        with self._db_lock:
            conn = self.analysis_db.get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO community_members (community_id, node_id, membership_score)
                    VALUES (?, ?, ?)
                    ON CONFLICT(community_id, node_id) DO UPDATE SET
                        membership_score = excluded.membership_score
                ''', (community_id, node_id, score))
                conn.commit()
            finally:
                conn.close()

    def get_communities(self, binary_hash: str) -> List[Dict[str, Any]]:
        """Get all communities for a binary."""
        if not binary_hash:
            return []
        with self._db_lock:
            conn = self.analysis_db.get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, level, binary_id, parent_community_id, name, summary,
                           member_count, is_stale, created_at, updated_at
                    FROM graph_communities
                    WHERE binary_id = ?
                    ORDER BY member_count DESC
                ''', (binary_hash,))
                rows = cursor.fetchall()
                communities = []
                for row in rows:
                    communities.append({
                        'id': row[0],
                        'level': row[1],
                        'binary_id': row[2],
                        'parent_community_id': row[3],
                        'name': row[4],
                        'summary': row[5],
                        'member_count': row[6],
                        'is_stale': bool(row[7]),
                        'created_at': row[8],
                        'updated_at': row[9],
                    })
                return communities
            finally:
                conn.close()

    def get_community_members(self, community_id: str) -> List[GraphNode]:
        """Get all member nodes of a community."""
        if not community_id:
            return []
        with self._db_lock:
            conn = self.analysis_db.get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT n.id, n.binary_id, n.type, n.address, n.name, n.raw_content,
                           n.llm_summary, n.confidence, n.embedding, n.security_flags,
                           n.network_apis, n.file_io_apis, n.ip_addresses, n.urls,
                           n.file_paths, n.domains, n.registry_keys, n.risk_level,
                           n.activity_profile, n.analysis_depth, n.created_at, n.updated_at,
                           n.is_stale, n.user_edited
                    FROM community_members cm
                    JOIN graph_nodes n ON n.id = cm.node_id
                    WHERE cm.community_id = ?
                    ORDER BY n.name
                ''', (community_id,))
                rows = cursor.fetchall()
                members = []
                for row in rows:
                    members.append(GraphNode(
                        id=row[0],
                        binary_hash=row[1],
                        node_type=row[2],
                        address=row[3],
                        name=row[4],
                        raw_code=row[5],
                        llm_summary=row[6],
                        confidence=row[7] if row[7] is not None else 0.0,
                        embedding=row[8],
                        security_flags=self._deserialize_list(row[9]),
                        network_apis=self._deserialize_list(row[10]),
                        file_io_apis=self._deserialize_list(row[11]),
                        ip_addresses=self._deserialize_list(row[12]),
                        urls=self._deserialize_list(row[13]),
                        file_paths=self._deserialize_list(row[14]),
                        domains=self._deserialize_list(row[15]),
                        registry_keys=self._deserialize_list(row[16]),
                        risk_level=row[17],
                        activity_profile=row[18],
                        analysis_depth=row[19] if row[19] is not None else 0,
                        created_at=row[20],
                        updated_at=row[21],
                        is_stale=bool(row[22]),
                        user_edited=bool(row[23]),
                    ))
                return members
            finally:
                conn.close()

    def get_community_for_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get the community that contains a specific node."""
        if not node_id:
            return None
        with self._db_lock:
            conn = self.analysis_db.get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT c.id, c.level, c.binary_id, c.parent_community_id, c.name,
                           c.summary, c.member_count, c.is_stale, c.created_at, c.updated_at
                    FROM community_members cm
                    JOIN graph_communities c ON c.id = cm.community_id
                    WHERE cm.node_id = ?
                    LIMIT 1
                ''', (node_id,))
                row = cursor.fetchone()
                if not row:
                    return None
                return {
                    'id': row[0],
                    'level': row[1],
                    'binary_id': row[2],
                    'parent_community_id': row[3],
                    'name': row[4],
                    'summary': row[5],
                    'member_count': row[6],
                    'is_stale': bool(row[7]),
                    'created_at': row[8],
                    'updated_at': row[9],
                }
            finally:
                conn.close()
