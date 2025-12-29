#!/usr/bin/env python3

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GraphNode:
    id: Optional[int] = None
    binary_hash: str = ""
    node_type: str = "FUNCTION"
    address: Optional[int] = None
    name: Optional[str] = None
    raw_code: Optional[str] = None
    llm_summary: Optional[str] = None
    security_flags: List[str] = field(default_factory=list)
    network_apis: List[str] = field(default_factory=list)
    file_io_apis: List[str] = field(default_factory=list)
    ip_addresses: List[str] = field(default_factory=list)
    urls: List[str] = field(default_factory=list)
    file_paths: List[str] = field(default_factory=list)
    domains: List[str] = field(default_factory=list)
    registry_keys: List[str] = field(default_factory=list)
    activity_profile: Optional[str] = None
    risk_level: Optional[str] = None
    is_stale: bool = True
    user_edited: bool = False
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass
class GraphEdge:
    id: Optional[int] = None
    binary_hash: str = ""
    source_id: int = 0
    target_id: int = 0
    edge_type: str = ""
    weight: float = 1.0
    metadata: Optional[str] = None
    created_at: Optional[str] = None
