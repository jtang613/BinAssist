#!/usr/bin/env python3
"""
SymGraph.ai API Client Service for BinAssist.

This service provides a clean interface for SymGraph.ai API integration,
following the BinAssist SOA architecture patterns.
"""

import asyncio
import threading
from typing import Dict, List, Optional, Any

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from .settings_service import settings_service
from .models.symgraph_models import (
    BinaryStats, Symbol, GraphNode, GraphEdge,
    SymbolExport, GraphExport, QueryResult, PushResult,
    PullPreviewResult, ConflictEntry, ConflictAction
)

try:
    import binaryninja
    log = binaryninja.log.Logger(0, "BinAssist")
except ImportError:
    # Fallback for testing outside Binary Ninja
    class MockLog:
        @staticmethod
        def log_info(msg): print(f"[BinAssist] INFO: {msg}")
        @staticmethod
        def log_error(msg): print(f"[BinAssist] ERROR: {msg}")
        @staticmethod
        def log_warn(msg): print(f"[BinAssist] WARN: {msg}")
        @staticmethod
        def log_debug(msg): print(f"[BinAssist] DEBUG: {msg}")
    log = MockLog()


class SymGraphServiceError(Exception):
    """Base exception for SymGraph service errors."""
    pass


class SymGraphAuthError(SymGraphServiceError):
    """Authentication error - API key missing or invalid."""
    pass


class SymGraphNetworkError(SymGraphServiceError):
    """Network error - unable to reach SymGraph.ai API."""
    pass


class SymGraphAPIError(SymGraphServiceError):
    """API error - server returned an error response."""
    def __init__(self, message: str, status_code: int = None):
        super().__init__(message)
        self.status_code = status_code


class SymGraphService:
    """
    SymGraph.ai API client service providing:
    - Binary existence checking (unauthenticated)
    - Binary stats retrieval (unauthenticated)
    - Symbol and graph data operations (authenticated)
    - Thread-safe operations
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the SymGraph service."""
        if hasattr(self, '_initialized'):
            return

        self._initialized = True
        self._client_lock = threading.Lock()

        if not HTTPX_AVAILABLE:
            log.log_warn("httpx package not available - SymGraph.ai integration disabled")

    @property
    def base_url(self) -> str:
        """Get the configured API base URL."""
        return settings_service.get_symgraph_api_url().rstrip('/')

    @property
    def api_key(self) -> Optional[str]:
        """Get the configured API key."""
        key = settings_service.get_symgraph_api_key()
        return key if key and key.strip() else None

    @property
    def has_api_key(self) -> bool:
        """Check if an API key is configured."""
        return settings_service.has_symgraph_api_key()

    def _get_headers(self, authenticated: bool = False) -> Dict[str, str]:
        """Get HTTP headers for requests."""
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'BinAssist-SymGraph/1.0'
        }
        if authenticated and self.api_key:
            headers['X-API-Key'] = self.api_key
        return headers

    def _check_httpx(self):
        """Check if httpx is available."""
        if not HTTPX_AVAILABLE:
            raise SymGraphServiceError("httpx package not installed. Install with: pip install httpx")

    def _check_auth(self):
        """Check if authentication is configured."""
        if not self.has_api_key:
            raise SymGraphAuthError("SymGraph.ai API key not configured. Add your API key in Settings > SymGraph.ai")

    # === Unauthenticated Operations ===

    async def check_binary_exists(self, sha256: str) -> bool:
        """
        Check if a binary exists in SymGraph.ai (unauthenticated).

        Args:
            sha256: SHA256 hash of the binary

        Returns:
            True if binary exists, False otherwise
        """
        self._check_httpx()

        url = f"{self.base_url}/api/v1/binaries/{sha256}"
        log.log_debug(f"Checking binary existence: {url}")

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.head(url, headers=self._get_headers())

                if response.status_code == 200:
                    return True
                elif response.status_code == 404:
                    return False
                else:
                    log.log_warn(f"Unexpected status checking binary: {response.status_code}")
                    return False

        except httpx.TimeoutException:
            raise SymGraphNetworkError(f"Timeout connecting to {self.base_url}")
        except httpx.RequestError as e:
            raise SymGraphNetworkError(f"Network error: {str(e)}")

    async def get_binary_stats(self, sha256: str) -> Optional[BinaryStats]:
        """
        Get binary statistics from SymGraph.ai (unauthenticated).

        Args:
            sha256: SHA256 hash of the binary

        Returns:
            BinaryStats if binary exists, None otherwise
        """
        self._check_httpx()

        url = f"{self.base_url}/api/v1/binaries/{sha256}/stats"
        log.log_debug(f"Getting binary stats: {url}")

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=self._get_headers())

                if response.status_code == 200:
                    data = response.json()
                    return BinaryStats.from_dict(data)
                elif response.status_code == 404:
                    return None
                else:
                    log.log_warn(f"Error getting stats: {response.status_code}")
                    return None

        except httpx.TimeoutException:
            raise SymGraphNetworkError(f"Timeout connecting to {self.base_url}")
        except httpx.RequestError as e:
            raise SymGraphNetworkError(f"Network error: {str(e)}")

    async def query_binary(self, sha256: str) -> QueryResult:
        """
        Query SymGraph.ai for binary info (unauthenticated).

        Args:
            sha256: SHA256 hash of the binary

        Returns:
            QueryResult with exists status and stats
        """
        try:
            exists = await self.check_binary_exists(sha256)
            if not exists:
                return QueryResult.not_found()

            stats = await self.get_binary_stats(sha256)
            if stats:
                return QueryResult.found(stats)
            else:
                return QueryResult(exists=True)  # Exists but no stats available

        except SymGraphServiceError as e:
            return QueryResult.error_result(str(e))

    # === Authenticated Operations ===

    async def get_symbols(self, sha256: str, symbol_type: Optional[str] = None) -> List[Symbol]:
        """
        Get symbols for a binary (authenticated).

        Args:
            sha256: SHA256 hash of the binary
            symbol_type: Optional filter by symbol type (e.g., "function", "data", "type")

        Returns:
            List of Symbol objects
        """
        self._check_httpx()
        self._check_auth()

        url = f"{self.base_url}/api/v1/binaries/{sha256}/symbols"
        if symbol_type:
            url += f"?type={symbol_type}"
        log.log_debug(f"Getting symbols: {url}")

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(
                    url,
                    headers=self._get_headers(authenticated=True)
                )

                if response.status_code == 200:
                    data = response.json()
                    symbols_data = data.get('symbols', data) if isinstance(data, dict) else data
                    return [Symbol.from_dict(s) for s in symbols_data]
                elif response.status_code == 401:
                    raise SymGraphAuthError("Invalid API key")
                elif response.status_code == 404:
                    return []
                else:
                    raise SymGraphAPIError(f"Error getting symbols: {response.status_code}", response.status_code)

        except httpx.TimeoutException:
            raise SymGraphNetworkError(f"Timeout connecting to {self.base_url}")
        except httpx.RequestError as e:
            raise SymGraphNetworkError(f"Network error: {str(e)}")

    async def push_symbols_bulk(self, sha256: str, symbols: List[Dict[str, Any]]) -> PushResult:
        """
        Push symbols to SymGraph.ai in bulk (authenticated).

        Args:
            sha256: SHA256 hash of the binary
            symbols: List of symbol dictionaries

        Returns:
            PushResult indicating success/failure
        """
        self._check_httpx()
        self._check_auth()

        url = f"{self.base_url}/api/v1/binaries/{sha256}/symbols/bulk"
        log.log_debug(f"Pushing {len(symbols)} symbols to: {url}")

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    url,
                    headers=self._get_headers(authenticated=True),
                    json={'symbols': symbols}
                )

                if response.status_code in (200, 201):
                    data = response.json()
                    return PushResult.success_result(
                        symbols=data.get('symbols_created', len(symbols))
                    )
                elif response.status_code == 401:
                    raise SymGraphAuthError("Invalid API key")
                else:
                    error_msg = response.text or f"HTTP {response.status_code}"
                    raise SymGraphAPIError(f"Error pushing symbols: {error_msg}", response.status_code)

        except httpx.TimeoutException:
            raise SymGraphNetworkError(f"Timeout connecting to {self.base_url}")
        except httpx.RequestError as e:
            raise SymGraphNetworkError(f"Network error: {str(e)}")

    async def export_symbols(self, sha256: str) -> Optional[SymbolExport]:
        """
        Export all symbols for a binary (authenticated).

        Args:
            sha256: SHA256 hash of the binary

        Returns:
            SymbolExport object or None if not found
        """
        self._check_httpx()
        self._check_auth()

        url = f"{self.base_url}/api/v1/binaries/{sha256}/export"
        log.log_debug(f"Exporting symbols: {url}")

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.get(
                    url,
                    headers=self._get_headers(authenticated=True)
                )

                if response.status_code == 200:
                    data = response.json()
                    return SymbolExport.from_dict(data)
                elif response.status_code == 401:
                    raise SymGraphAuthError("Invalid API key")
                elif response.status_code == 404:
                    return None
                else:
                    raise SymGraphAPIError(f"Error exporting symbols: {response.status_code}", response.status_code)

        except httpx.TimeoutException:
            raise SymGraphNetworkError(f"Timeout connecting to {self.base_url}")
        except httpx.RequestError as e:
            raise SymGraphNetworkError(f"Network error: {str(e)}")

    async def import_symbols(self, sha256: str, export_data: Dict[str, Any]) -> PushResult:
        """
        Import symbols to SymGraph.ai (authenticated).

        Args:
            sha256: SHA256 hash of the binary
            export_data: Symbol export data dictionary

        Returns:
            PushResult indicating success/failure
        """
        self._check_httpx()
        self._check_auth()

        url = f"{self.base_url}/api/v1/binaries/{sha256}/import"
        log.log_debug(f"Importing symbols to: {url}")

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    url,
                    headers=self._get_headers(authenticated=True),
                    json=export_data
                )

                if response.status_code in (200, 201):
                    data = response.json()
                    return PushResult.success_result(
                        symbols=data.get('symbols_imported', 0)
                    )
                elif response.status_code == 401:
                    raise SymGraphAuthError("Invalid API key")
                else:
                    error_msg = response.text or f"HTTP {response.status_code}"
                    raise SymGraphAPIError(f"Error importing symbols: {error_msg}", response.status_code)

        except httpx.TimeoutException:
            raise SymGraphNetworkError(f"Timeout connecting to {self.base_url}")
        except httpx.RequestError as e:
            raise SymGraphNetworkError(f"Network error: {str(e)}")

    async def export_graph(self, sha256: str) -> Optional[GraphExport]:
        """
        Export graph data for a binary (authenticated).

        Args:
            sha256: SHA256 hash of the binary

        Returns:
            GraphExport object or None if not found
        """
        self._check_httpx()
        self._check_auth()

        url = f"{self.base_url}/api/v1/binaries/{sha256}/graph/export"
        log.log_debug(f"Exporting graph: {url}")

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.get(
                    url,
                    headers=self._get_headers(authenticated=True)
                )

                if response.status_code == 200:
                    data = response.json()
                    return GraphExport.from_dict(data)
                elif response.status_code == 401:
                    raise SymGraphAuthError("Invalid API key")
                elif response.status_code == 404:
                    return None
                else:
                    raise SymGraphAPIError(f"Error exporting graph: {response.status_code}", response.status_code)

        except httpx.TimeoutException:
            raise SymGraphNetworkError(f"Timeout connecting to {self.base_url}")
        except httpx.RequestError as e:
            raise SymGraphNetworkError(f"Network error: {str(e)}")

    async def import_graph(self, sha256: str, export_data: Dict[str, Any]) -> PushResult:
        """
        Import graph data to SymGraph.ai (authenticated).

        Args:
            sha256: SHA256 hash of the binary
            export_data: Graph export data dictionary

        Returns:
            PushResult indicating success/failure
        """
        self._check_httpx()
        self._check_auth()

        url = f"{self.base_url}/api/v1/binaries/{sha256}/graph/import"
        log.log_debug(f"Importing graph to: {url}")

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    url,
                    headers=self._get_headers(authenticated=True),
                    json=export_data
                )

                if response.status_code in (200, 201):
                    data = response.json()
                    return PushResult.success_result(
                        nodes=data.get('nodes_imported', 0),
                        edges=data.get('edges_imported', 0)
                    )
                elif response.status_code == 401:
                    raise SymGraphAuthError("Invalid API key")
                else:
                    error_msg = response.text or f"HTTP {response.status_code}"
                    raise SymGraphAPIError(f"Error importing graph: {error_msg}", response.status_code)

        except httpx.TimeoutException:
            raise SymGraphNetworkError(f"Timeout connecting to {self.base_url}")
        except httpx.RequestError as e:
            raise SymGraphNetworkError(f"Network error: {str(e)}")

    # === Fingerprint Operations ===

    async def add_fingerprint(self, sha256: str, fp_type: str, fp_value: str) -> bool:
        """
        Add a fingerprint to a binary (authenticated).

        Args:
            sha256: SHA256 hash of the binary
            fp_type: Fingerprint type (e.g., "pdb_guid", "build_id")
            fp_value: Fingerprint value

        Returns:
            True if successful, False otherwise
        """
        self._check_httpx()
        self._check_auth()

        url = f"{self.base_url}/api/v1/binaries/{sha256}/fingerprints"
        log.log_debug(f"Adding fingerprint {fp_type}={fp_value} to {sha256[:16]}...")

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    url,
                    headers=self._get_headers(authenticated=True),
                    json={'type': fp_type, 'value': fp_value}
                )

                if response.status_code in (200, 201):
                    log.log_info(f"Added fingerprint: {fp_type}={fp_value}")
                    return True
                elif response.status_code == 401:
                    raise SymGraphAuthError("Invalid API key")
                elif response.status_code == 409:
                    # Fingerprint already exists - not an error
                    log.log_debug(f"Fingerprint already exists: {fp_type}={fp_value}")
                    return True
                else:
                    log.log_warn(f"Failed to add fingerprint: {response.status_code}")
                    return False

        except httpx.TimeoutException:
            raise SymGraphNetworkError(f"Timeout connecting to {self.base_url}")
        except httpx.RequestError as e:
            raise SymGraphNetworkError(f"Network error: {str(e)}")

    # === Helper Methods for Pull Preview ===

    def build_conflict_entries(
        self,
        local_symbols: Dict[int, str],
        remote_symbols: List[Symbol]
    ) -> List[ConflictEntry]:
        """
        Build conflict entries by comparing local and remote symbols.

        Args:
            local_symbols: Dict mapping address to local name
            remote_symbols: List of remote Symbol objects

        Returns:
            List of ConflictEntry objects for the conflict resolution table
        """
        conflicts = []
        remote_addresses = set()

        for remote_sym in remote_symbols:
            addr = remote_sym.address
            remote_addresses.add(addr)
            local_name = local_symbols.get(addr)

            if local_name is None:
                # Remote only - NEW
                conflicts.append(ConflictEntry.create_new(addr, remote_sym))
            elif local_name == remote_sym.name:
                # Same value - SAME
                conflicts.append(ConflictEntry.create_same(addr, local_name, remote_sym))
            else:
                # Different values - CONFLICT
                conflicts.append(ConflictEntry.create_conflict(addr, local_name, remote_sym))

        return conflicts


# Singleton instance
symgraph_service = SymGraphService()
