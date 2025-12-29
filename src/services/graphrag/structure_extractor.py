#!/usr/bin/env python3

from typing import Optional

from .graph_store import GraphStore
from .models import GraphNode, GraphEdge
from .security_feature_extractor import SecurityFeatureExtractor

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


class StructureExtractor:
    """Extracts function nodes and relationships into the GraphRAG store."""

    def __init__(self, binary_view, graph_store: GraphStore):
        self.binary_view = binary_view
        self.graph_store = graph_store
        self.security_extractor = SecurityFeatureExtractor(binary_view)

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

        features = self.security_extractor.extract_features(function)
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
        node.is_stale = True

        node = self.graph_store.upsert_node(node)

        self._extract_call_edges(function, binary_hash, node)
        return node

    def _extract_call_edges(self, function, binary_hash: str, node: GraphNode) -> None:
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

            self.graph_store.add_edge(GraphEdge(
                binary_hash=binary_hash,
                source_id=node.id,
                target_id=callee_node.id,
                edge_type="CALLS",
                weight=1.0,
            ))

            if callee_node.security_flags and any(flag.endswith("_RISK") for flag in callee_node.security_flags):
                self.graph_store.add_edge(GraphEdge(
                    binary_hash=binary_hash,
                    source_id=node.id,
                    target_id=callee_node.id,
                    edge_type="CALLS_VULNERABLE",
                    weight=1.0,
                ))
                if "CALLS_VULNERABLE_FUNCTION" not in node.security_flags:
                    node.security_flags.append("CALLS_VULNERABLE_FUNCTION")
                    self.graph_store.upsert_node(node)
