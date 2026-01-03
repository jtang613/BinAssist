"""
SymGraph.ai data models for BinAssist.

This module provides data structures for SymGraph.ai API integration,
following the existing BinAssist model architecture patterns.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class ConflictAction(Enum):
    """Action type for conflict resolution during pull."""
    NEW = "new"           # Remote only, doesn't exist locally
    CONFLICT = "conflict" # Different values locally and remotely
    SAME = "same"         # Identical values locally and remotely
    LOCAL_ONLY = "local"  # Exists locally but not remotely


class PushScope(Enum):
    """Scope for push operations."""
    FULL_BINARY = "full"
    CURRENT_FUNCTION = "function"


class SymbolType(Enum):
    """Type of symbol."""
    FUNCTION = "function"
    DATA = "data"
    LABEL = "label"
    TYPE = "type"
    COMMENT = "comment"


@dataclass
class BinaryStats:
    """Statistics for a binary in SymGraph.ai."""
    symbol_count: int = 0
    function_count: int = 0
    graph_node_count: int = 0
    query_count: int = 0
    last_queried_at: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BinaryStats':
        """Create stats from API response dictionary.

        Handles both nested stats (in 'stats' key) and flat response formats.
        """
        # Check if stats are nested inside a 'stats' key
        stats_data = data.get('stats', data)

        # last_queried_at might be at top level even when stats are nested
        last_queried = data.get('last_queried_at') or stats_data.get('last_queried_at')

        return cls(
            symbol_count=stats_data.get('symbol_count', 0),
            function_count=stats_data.get('function_count', 0),
            graph_node_count=stats_data.get('graph_node_count', 0),
            query_count=stats_data.get('query_count', 0),
            last_queried_at=last_queried
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            'symbol_count': self.symbol_count,
            'function_count': self.function_count,
            'graph_node_count': self.graph_node_count,
            'query_count': self.query_count,
            'last_queried_at': self.last_queried_at
        }


@dataclass
class Symbol:
    """A symbol from SymGraph.ai."""
    address: int
    symbol_type: str
    name: Optional[str] = None
    data_type: Optional[str] = None
    confidence: float = 0.0
    provenance: str = "unknown"
    source: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Symbol':
        """Create symbol from API response dictionary."""
        return cls(
            address=data.get('address', 0),
            symbol_type=data.get('symbol_type', 'function'),
            name=data.get('name'),
            data_type=data.get('data_type'),
            confidence=data.get('confidence', 0.0),
            provenance=data.get('provenance', 'unknown'),
            source=data.get('source')
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert symbol to dictionary."""
        result = {
            'address': self.address,
            'symbol_type': self.symbol_type,
            'confidence': self.confidence,
            'provenance': self.provenance
        }
        if self.name:
            result['name'] = self.name
        if self.data_type:
            result['data_type'] = self.data_type
        if self.source:
            result['source'] = self.source
        return result


@dataclass
class GraphNode:
    """A graph node from SymGraph.ai."""
    id: Optional[str] = None
    address: int = 0
    node_type: str = "function"
    name: Optional[str] = None
    summary: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphNode':
        """Create graph node from API response dictionary."""
        return cls(
            id=data.get('id'),
            address=data.get('address', 0),
            node_type=data.get('node_type', 'function'),
            name=data.get('name'),
            summary=data.get('summary'),
            properties=data.get('properties', {})
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert graph node to dictionary."""
        result = {
            'address': self.address,
            'node_type': self.node_type
        }
        if self.id:
            result['id'] = self.id
        if self.name:
            result['name'] = self.name
        if self.summary:
            result['summary'] = self.summary
        if self.properties:
            result['properties'] = self.properties
        return result


@dataclass
class GraphEdge:
    """A graph edge from SymGraph.ai."""
    source_address: int
    target_address: int
    edge_type: str = "calls"
    properties: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphEdge':
        """Create graph edge from API response dictionary."""
        return cls(
            source_address=data.get('source_address', 0),
            target_address=data.get('target_address', 0),
            edge_type=data.get('edge_type', 'calls'),
            properties=data.get('properties', {})
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert graph edge to dictionary."""
        result = {
            'source_address': self.source_address,
            'target_address': self.target_address,
            'edge_type': self.edge_type
        }
        if self.properties:
            result['properties'] = self.properties
        return result


@dataclass
class ConflictEntry:
    """An entry in the conflict resolution table."""
    address: int
    local_name: Optional[str]
    remote_name: Optional[str]
    action: ConflictAction
    selected: bool
    remote_symbol: Optional[Symbol] = None
    local_symbol: Optional[Symbol] = None

    @classmethod
    def create_new(cls, address: int, remote_symbol: Symbol) -> 'ConflictEntry':
        """Create a NEW conflict entry (remote only)."""
        return cls(
            address=address,
            local_name=None,
            remote_name=remote_symbol.name,
            action=ConflictAction.NEW,
            selected=True,  # New items checked by default
            remote_symbol=remote_symbol
        )

    @classmethod
    def create_conflict(cls, address: int, local_name: str, remote_symbol: Symbol) -> 'ConflictEntry':
        """Create a CONFLICT entry (different values)."""
        return cls(
            address=address,
            local_name=local_name,
            remote_name=remote_symbol.name,
            action=ConflictAction.CONFLICT,
            selected=False,  # Conflicts unchecked by default
            remote_symbol=remote_symbol
        )

    @classmethod
    def create_same(cls, address: int, name: str, remote_symbol: Symbol) -> 'ConflictEntry':
        """Create a SAME entry (identical values)."""
        return cls(
            address=address,
            local_name=name,
            remote_name=name,
            action=ConflictAction.SAME,
            selected=True,  # Same items checked by default
            remote_symbol=remote_symbol
        )

    def to_table_row(self) -> tuple:
        """Convert to table row data for UI display."""
        return (
            self.selected,
            f"0x{self.address:x}",
            self.local_name or "<none>",
            self.remote_name or "<none>",
            self.action.value.upper()
        )


@dataclass
class SymbolExport:
    """Exported symbol data from SymGraph.ai."""
    binary_sha256: str
    symbols: List[Symbol]
    export_version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SymbolExport':
        """Create export from API response dictionary."""
        symbols = [Symbol.from_dict(s) for s in data.get('symbols', [])]
        return cls(
            binary_sha256=data.get('binary_sha256', ''),
            symbols=symbols,
            export_version=data.get('export_version', '1.0'),
            metadata=data.get('metadata', {})
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert export to dictionary."""
        return {
            'binary_sha256': self.binary_sha256,
            'symbols': [s.to_dict() for s in self.symbols],
            'export_version': self.export_version,
            'metadata': self.metadata
        }


@dataclass
class GraphExport:
    """Exported graph data from SymGraph.ai."""
    binary_sha256: str
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    export_version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphExport':
        """Create export from API response dictionary."""
        nodes = [GraphNode.from_dict(n) for n in data.get('nodes', [])]
        edges = [GraphEdge.from_dict(e) for e in data.get('edges', [])]
        return cls(
            binary_sha256=data.get('binary_sha256', ''),
            nodes=nodes,
            edges=edges,
            export_version=data.get('export_version', '1.0'),
            metadata=data.get('metadata', {})
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert export to dictionary."""
        return {
            'binary_sha256': self.binary_sha256,
            'nodes': [n.to_dict() for n in self.nodes],
            'edges': [e.to_dict() for e in self.edges],
            'export_version': self.export_version,
            'metadata': self.metadata
        }


@dataclass
class QueryResult:
    """Result of a SymGraph.ai query operation."""
    exists: bool
    stats: Optional[BinaryStats] = None
    error: Optional[str] = None

    @classmethod
    def found(cls, stats: BinaryStats) -> 'QueryResult':
        """Create a successful query result."""
        return cls(exists=True, stats=stats)

    @classmethod
    def not_found(cls) -> 'QueryResult':
        """Create a not-found query result."""
        return cls(exists=False)

    @classmethod
    def error_result(cls, error: str) -> 'QueryResult':
        """Create an error query result."""
        return cls(exists=False, error=error)


@dataclass
class PushResult:
    """Result of a SymGraph.ai push operation."""
    success: bool
    symbols_pushed: int = 0
    nodes_pushed: int = 0
    edges_pushed: int = 0
    error: Optional[str] = None

    @classmethod
    def success_result(cls, symbols: int = 0, nodes: int = 0, edges: int = 0) -> 'PushResult':
        """Create a successful push result."""
        return cls(
            success=True,
            symbols_pushed=symbols,
            nodes_pushed=nodes,
            edges_pushed=edges
        )

    @classmethod
    def failure_result(cls, error: str) -> 'PushResult':
        """Create a failed push result."""
        return cls(success=False, error=error)


@dataclass
class PullPreviewResult:
    """Result of a pull preview operation."""
    success: bool
    conflicts: List[ConflictEntry] = field(default_factory=list)
    error: Optional[str] = None

    @classmethod
    def success_result(cls, conflicts: List[ConflictEntry]) -> 'PullPreviewResult':
        """Create a successful pull preview result."""
        return cls(success=True, conflicts=conflicts)

    @classmethod
    def failure_result(cls, error: str) -> 'PullPreviewResult':
        """Create a failed pull preview result."""
        return cls(success=False, error=error)

    def get_sorted_conflicts(self) -> List[ConflictEntry]:
        """Get conflicts sorted with CONFLICT items first."""
        return sorted(
            self.conflicts,
            key=lambda x: (x.action != ConflictAction.CONFLICT, x.address)
        )
