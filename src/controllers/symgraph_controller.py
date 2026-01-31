#!/usr/bin/env python3
"""
SymGraph Controller for BinAssist.

This controller manages the SymGraph tab functionality including
querying, pushing, and pulling symbols and graph data.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Callable
from PySide6.QtWidgets import QMessageBox
from PySide6.QtCore import QThread, Signal, QObject

from ..services.analysis_db_service import analysis_db_service
from ..services.graphrag.graph_store import GraphStore
from ..services.graphrag.models import GraphNode as LocalGraphNode, GraphEdge as LocalGraphEdge, NodeType, EdgeType

from ..services.symgraph_service import (
    symgraph_service, SymGraphServiceError, SymGraphAuthError,
    SymGraphNetworkError, SymGraphAPIError, is_default_name
)
from ..services.models.symgraph_models import (
    BinaryStats, Symbol, ConflictEntry, ConflictAction,
    QueryResult, PushResult, PullPreviewResult, PushScope
)
from ..views.symgraph_tab_view import SymGraphTabView

# Setup BinAssist logger
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


class AsyncWorker(QThread):
    """Generic async worker thread for running coroutines."""

    finished = Signal(object)  # result
    error = Signal(str)  # error message

    def __init__(self, coro_func: Callable, *args, **kwargs):
        super().__init__()
        self.coro_func = coro_func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        """Execute the coroutine in a new event loop."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self.coro_func(*self.args, **self.kwargs))
                self.finished.emit(result)
            finally:
                loop.close()
        except Exception as e:
            log.log_error(f"AsyncWorker error: {e}")
            self.error.emit(str(e))


class QueryWorker(QThread):
    """Worker thread for querying SymGraph."""

    query_complete = Signal(object)  # QueryResult
    query_error = Signal(str)

    def __init__(self, sha256: str):
        super().__init__()
        self.sha256 = sha256

    def run(self):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(symgraph_service.query_binary(self.sha256))
                self.query_complete.emit(result)
            finally:
                loop.close()
        except Exception as e:
            log.log_error(f"Query error: {e}")
            self.query_error.emit(str(e))


class PushWorker(QThread):
    """Worker thread for pushing to SymGraph."""

    push_complete = Signal(object)  # PushResult
    push_error = Signal(str)

    def __init__(self, sha256: str, symbols: List[Dict], graph_data: Optional[Dict] = None,
                 fingerprints: Optional[List[Dict[str, str]]] = None):
        super().__init__()
        self.sha256 = sha256
        self.symbols = symbols
        self.graph_data = graph_data
        self.fingerprints = fingerprints or []  # List of {'type': str, 'value': str}

    def run(self):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                total_result = PushResult(success=True)

                # Push symbols if provided
                if self.symbols:
                    result = loop.run_until_complete(
                        symgraph_service.push_symbols_bulk(self.sha256, self.symbols)
                    )
                    total_result.symbols_pushed = result.symbols_pushed
                    if not result.success:
                        total_result.success = False
                        total_result.error = result.error

                # Push graph if provided
                if self.graph_data and total_result.success:
                    result = loop.run_until_complete(
                        symgraph_service.import_graph(self.sha256, self.graph_data)
                    )
                    total_result.nodes_pushed = result.nodes_pushed
                    total_result.edges_pushed = result.edges_pushed
                    if not result.success:
                        total_result.success = False
                        total_result.error = result.error

                # Add fingerprints (for BuildID/PDB GUID matching)
                if self.fingerprints and total_result.success:
                    for fp in self.fingerprints:
                        try:
                            loop.run_until_complete(
                                symgraph_service.add_fingerprint(
                                    self.sha256, fp['type'], fp['value']
                                )
                            )
                        except Exception as e:
                            log.log_warn(f"Failed to add fingerprint {fp['type']}: {e}")
                            # Non-fatal, continue

                self.push_complete.emit(total_result)
            finally:
                loop.close()
        except SymGraphAuthError as e:
            self.push_error.emit(f"Authentication required: {e}")
        except SymGraphNetworkError as e:
            self.push_error.emit(f"Network error: {e}")
        except Exception as e:
            log.log_error(f"Push error: {e}")
            self.push_error.emit(str(e))


class PullPreviewWorker(QThread):
    """Worker thread for pulling symbols from SymGraph and building conflicts."""

    progress = Signal(str)  # status message
    preview_complete = Signal(list, object, object)  # conflicts, graph_export, graph_stats
    preview_error = Signal(str)

    def __init__(self, sha256: str, bv, pull_config: dict = None):
        super().__init__()
        self.sha256 = sha256
        self.bv = bv  # Binary Ninja binary view for getting local symbols
        self.pull_config = pull_config or {
            'symbol_types': ['function'],
            'min_confidence': 0.0,
            'include_graph': False
        }
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            symbol_types = self.pull_config.get('symbol_types', ['function'])
            min_confidence = self.pull_config.get('min_confidence', 0.0)

            # Step 1: Fetch remote symbols from API for each selected type
            all_remote_symbols = []
            include_graph = bool(self.pull_config.get('include_graph', False))
            graph_export = None
            graph_stats = None
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                for sym_type in symbol_types:
                    if self._cancelled:
                        return

                    self.progress.emit(f"Fetching {sym_type} symbols...")
                    remote_symbols = loop.run_until_complete(
                        symgraph_service.get_symbols(self.sha256, symbol_type=sym_type)
                    )
                    # Handle None or empty results safely
                    if remote_symbols:
                        all_remote_symbols.extend(remote_symbols)
                        log.log_info(f"Fetched {len(remote_symbols)} {sym_type} symbols from API")
                    else:
                        log.log_info(f"No {sym_type} symbols returned from API")

                if include_graph:
                    try:
                        self.progress.emit("Fetching graph data...")
                        graph_export = loop.run_until_complete(
                            symgraph_service.export_graph(self.sha256)
                        )
                        if graph_export:
                            graph_stats = self._get_graph_stats(graph_export)
                            log.log_info(
                                f"Fetched graph data: {graph_stats.get('nodes', 0)} nodes, {graph_stats.get('edges', 0)} edges"
                            )
                    except Exception as e:
                        log.log_warn(f"Graph export failed: {e}")
            finally:
                loop.close()

            if self._cancelled:
                return

            log.log_info(f"Total fetched: {len(all_remote_symbols)} symbols from API")

            if not all_remote_symbols:
                self.preview_complete.emit([], graph_export, graph_stats)
                return

            # Step 2: Get local symbols from Binary Ninja
            self.progress.emit("Collecting local symbols...")
            local_symbols = self._get_local_symbol_map()

            if self._cancelled:
                return

            log.log_info(f"Found {len(local_symbols)} local symbols")

            # Step 3: Build conflict entries with confidence filtering
            self.progress.emit("Building conflict list...")
            conflicts = symgraph_service.build_conflict_entries(
                local_symbols, all_remote_symbols, min_confidence
            )

            if self._cancelled:
                return

            log.log_info(f"Built {len(conflicts)} conflict entries")
            self.preview_complete.emit(conflicts, graph_export, graph_stats)

        except SymGraphAuthError as e:
            self.preview_error.emit(f"Authentication required: {e}")
        except SymGraphNetworkError as e:
            self.preview_error.emit(f"Network error: {e}")
        except Exception as e:
            log.log_error(f"Pull preview error: {e}")
            self.preview_error.emit(str(e))

    def _get_local_symbol_map(self) -> Dict[int, str]:
        """Get a map of address -> name for local function symbols."""
        local_symbols = {}

        if not self.bv:
            return local_symbols

        try:
            for func in self.bv.functions:
                # Include all functions, even auto-named ones
                local_symbols[func.start] = func.name
        except Exception as e:
            log.log_error(f"Error getting local symbols: {e}")

        return local_symbols

    @staticmethod
    def _get_graph_stats(graph_export) -> Dict[str, int]:
        metadata = graph_export.metadata or {}
        communities = metadata.get("community_count")
        if communities is None:
            communities = len(metadata.get("communities", [])) if isinstance(metadata.get("communities"), list) else 0
        return {
            "nodes": len(graph_export.nodes),
            "edges": len(graph_export.edges),
            "communities": communities,
        }


class ApplySymbolsWorker(QThread):
    """Worker thread for applying symbols to Binary Ninja."""

    progress = Signal(int, int, str)  # current, total, message
    apply_complete = Signal(int, int)  # applied, errors
    apply_cancelled = Signal(int)  # applied so far
    apply_error = Signal(str)

    def __init__(self, bv, symbols: List, graph_export=None, merge_policy: str = None, binary_hash: str = None):
        super().__init__()
        self.bv = bv
        self.symbols = symbols
        self.graph_export = graph_export
        self.merge_policy = merge_policy
        self.binary_hash = binary_hash
        self._cancelled = False

    def cancel(self):
        """Request cancellation of the apply operation."""
        self._cancelled = True

    def run(self):
        try:
            import binaryninja

            # Calculate total work items (nodes + edges + symbols)
            num_nodes = len(self.graph_export.nodes) if self.graph_export else 0
            num_edges = len(self.graph_export.edges) if self.graph_export else 0
            num_symbols = len(self.symbols)
            total = num_nodes + num_edges + num_symbols

            progress_count = 0
            applied = 0
            errors = 0

            # Phase 1: Merge graph data (with progress)
            if self.graph_export and self.binary_hash:
                progress_count = self._merge_graph_data(progress_count, total, num_nodes, num_edges)
                if self._cancelled:
                    self.apply_cancelled.emit(applied)
                    return

            # Phase 2: Apply symbols (with progress)
            for i, symbol in enumerate(self.symbols):
                if self._cancelled:
                    self.apply_cancelled.emit(applied)
                    return

                # Handle both Symbol objects and ConflictEntry objects
                if hasattr(symbol, 'remote_symbol'):
                    # It's a ConflictEntry
                    addr = symbol.address
                    remote_sym = symbol.remote_symbol
                    name = remote_sym.name if remote_sym else None
                    symbol_type = getattr(remote_sym, 'symbol_type', 'function') if remote_sym else 'function'
                    metadata = getattr(remote_sym, 'metadata', {}) if remote_sym else {}
                else:
                    # It's a Symbol object
                    addr = symbol.address
                    name = symbol.name
                    symbol_type = getattr(symbol, 'symbol_type', 'function')
                    metadata = getattr(symbol, 'metadata', {})

                if name:
                    try:
                        if symbol_type == 'variable':
                            # Use storage-aware variable application
                            symbol_data = {
                                'name': name,
                                'metadata': metadata
                            }
                            if self._apply_variable(addr, symbol_data):
                                applied += 1
                        else:
                            # Existing function/symbol application
                            self._apply_symbol(addr, name)
                            applied += 1
                    except Exception as e:
                        log.log_error(f"Error applying symbol at 0x{addr:x}: {e}")
                        errors += 1

                progress_count += 1
                self.progress.emit(progress_count, total,
                    f"Applying symbol {i + 1}/{num_symbols}...")

            self.apply_complete.emit(applied, errors)

        except Exception as e:
            log.log_error(f"Apply symbols error: {e}")
            self.apply_error.emit(str(e))

    def _apply_symbol(self, addr: int, name: str):
        """Apply a single symbol to Binary Ninja."""
        import binaryninja

        # Check if it's a function
        funcs = self.bv.get_functions_containing(addr)
        if funcs:
            for func in funcs:
                if func.start == addr:
                    func.name = name
                    log.log_debug(f"Renamed function at 0x{addr:x} to {name}")
                    return

        # Otherwise try to create/update a symbol
        self.bv.define_user_symbol(
            binaryninja.Symbol(binaryninja.SymbolType.FunctionSymbol, addr, name)
        )
        log.log_debug(f"Created symbol at 0x{addr:x}: {name}")

    def _apply_variable(self, func_addr: int, symbol_data: dict) -> bool:
        """Apply a variable symbol to Binary Ninja using storage matching."""
        from binaryninja.enums import VariableSourceType

        funcs = self.bv.get_functions_at(func_addr)
        if not funcs:
            return False

        func = funcs[0]
        metadata = symbol_data.get('metadata', {})
        target_name = symbol_data.get('name')
        storage_class = metadata.get('storage_class')

        if not target_name or not storage_class:
            return False

        try:
            if storage_class == 'parameter':
                param_idx = metadata.get('parameter_index')
                if param_idx is not None and param_idx < len(func.parameter_vars):
                    func.parameter_vars[param_idx].name = target_name
                    log.log_debug(f"Renamed parameter {param_idx} at 0x{func_addr:x} to {target_name}")
                    return True

            elif storage_class == 'stack':
                stack_offset = metadata.get('stack_offset')
                if stack_offset is not None:
                    for var in func.vars:
                        if (var.source_type == VariableSourceType.StackVariableSourceType
                            and var.storage == stack_offset):
                            var.name = target_name
                            log.log_debug(f"Renamed stack var at offset {stack_offset} to {target_name}")
                            return True

            elif storage_class == 'register':
                reg_name = metadata.get('register')
                if reg_name:
                    for var in func.vars:
                        if var.source_type == VariableSourceType.RegisterVariableSourceType:
                            try:
                                var_reg = self.bv.arch._regs_by_index.get(var.storage)
                                if var_reg == reg_name:
                                    var.name = target_name
                                    log.log_debug(f"Renamed register var {reg_name} to {target_name}")
                                    return True
                            except:
                                pass
        except Exception as e:
            log.log_error(f"Error applying variable: {e}")

        return False

    def _merge_graph_data(self, progress_count: int, total: int, num_nodes: int, num_edges: int) -> int:
        """Merge graph data with progress updates.

        Returns the updated progress count after processing nodes and edges.
        """
        if not self.graph_export or not self.binary_hash:
            return progress_count

        graph_store = GraphStore(analysis_db_service)
        merge_policy = self.merge_policy or "upsert"

        if merge_policy == "replace":
            graph_store.delete_graph(self.binary_hash)
            graph_store.delete_communities(self.binary_hash)

        address_to_id: Dict[int, str] = {}
        for i, node in enumerate(self.graph_export.nodes):
            if self._cancelled:
                return progress_count

            node_type_str = (node.node_type or "FUNCTION").upper()
            node_type = NodeType.from_string(node_type_str) or NodeType.FUNCTION
            existing = graph_store.get_node_by_address(self.binary_hash, node_type.value, node.address)

            if merge_policy == "prefer_local" and existing:
                address_to_id[node.address] = existing.id
            else:
                props = node.properties or {}
                node_id = existing.id if existing else node.id

                local_node = LocalGraphNode(
                    id=node_id,
                    binary_hash=self.binary_hash,
                    node_type=node_type,
                    address=node.address,
                    name=node.name,
                    raw_code=props.get("raw_code") or props.get("raw_content"),
                    llm_summary=node.summary or props.get("llm_summary"),
                    confidence=float(props.get("confidence", 0.0) or 0.0),
                    security_flags=self._coerce_list(props.get("security_flags")),
                    network_apis=self._coerce_list(props.get("network_apis")),
                    file_io_apis=self._coerce_list(props.get("file_io_apis")),
                    ip_addresses=self._coerce_list(props.get("ip_addresses")),
                    urls=self._coerce_list(props.get("urls")),
                    file_paths=self._coerce_list(props.get("file_paths")),
                    domains=self._coerce_list(props.get("domains")),
                    registry_keys=self._coerce_list(props.get("registry_keys")),
                    risk_level=props.get("risk_level"),
                    activity_profile=props.get("activity_profile"),
                    analysis_depth=int(props.get("analysis_depth", 0) or 0),
                    is_stale=bool(props.get("is_stale", False)),
                    user_edited=bool(props.get("user_edited", False))
                )
                graph_store.upsert_node(local_node)
                address_to_id[node.address] = local_node.id

            progress_count += 1
            self.progress.emit(progress_count, total,
                f"Merging node {i + 1}/{num_nodes}...")

        for i, edge in enumerate(self.graph_export.edges):
            if self._cancelled:
                return progress_count

            source_id = address_to_id.get(edge.source_address)
            target_id = address_to_id.get(edge.target_address)
            if not source_id or not target_id:
                progress_count += 1
                self.progress.emit(progress_count, total,
                    f"Merging edge {i + 1}/{num_edges}...")
                continue

            edge_type = EdgeType.from_string(edge.edge_type) or EdgeType.CALLS
            metadata = edge.properties or {}
            weight = float(metadata.get("weight", 1.0) or 1.0)
            metadata_json = json.dumps(metadata) if metadata else None

            graph_store.add_edge(LocalGraphEdge(
                binary_hash=self.binary_hash,
                source_id=source_id,
                target_id=target_id,
                edge_type=edge_type,
                weight=weight,
                metadata=metadata_json
            ))

            progress_count += 1
            self.progress.emit(progress_count, total,
                f"Merging edge {i + 1}/{num_edges}...")

        return progress_count

    @staticmethod
    def _coerce_list(value):
        return value if isinstance(value, list) else []


class SymGraphController(QObject):
    """Controller for the SymGraph tab functionality."""

    def __init__(self, view: SymGraphTabView, binary_view=None, data=None, frame=None):
        super().__init__()
        self.view = view
        self.bv = binary_view  # Binary Ninja binary view
        self.data = data       # BinAssist data object
        self.frame = frame     # BinAssist frame

        # Worker threads
        self.query_worker = None
        self.push_worker = None
        self.pull_worker = None
        self.apply_worker = None

        self._graph_export = None
        self._graph_stats = None

        # Connect view signals
        self._connect_signals()

        # Update binary info if available
        self._update_binary_info()

    def _connect_signals(self):
        """Connect view signals to controller methods."""
        self.view.query_requested.connect(self.handle_query)
        self.view.push_requested.connect(self.handle_push)
        self.view.pull_preview_requested.connect(self.handle_pull_preview)
        self.view.apply_selected_requested.connect(self.handle_apply_selected)
        self.view.apply_all_new_requested.connect(self.handle_apply_all_new)

    def set_binary_view(self, bv):
        """Update the binary view reference."""
        self.bv = bv
        self._update_binary_info()

    def _update_binary_info(self):
        """Update binary info display from current binary view."""
        if self.bv:
            try:
                # Get the original filename (not the bndb path)
                name = "Unknown"
                if self.bv.file:
                    # original_filename gives the path to the actual binary
                    if hasattr(self.bv.file, 'original_filename') and self.bv.file.original_filename:
                        name = self.bv.file.original_filename
                    else:
                        name = self.bv.file.filename
                    name = name.split('/')[-1].split('\\')[-1]  # Get filename only
                    # Remove .bndb extension if present
                    if name.endswith('.bndb'):
                        name = name[:-5]

                sha256 = self._get_sha256()
                self.view.set_binary_info(name, sha256)
            except Exception as e:
                log.log_error(f"Error getting binary info: {e}")
                self.view.set_binary_info("<error>", None)
        else:
            self.view.set_binary_info("<no binary loaded>", None)

    def _get_sha256(self) -> Optional[str]:
        """Get SHA256 hash of the original binary (not the bndb file)."""
        if not self.bv:
            return None

        try:
            import hashlib

            # Get the raw view which contains the original binary data
            raw_view = None
            if self.bv.file and hasattr(self.bv.file, 'raw') and self.bv.file.raw:
                raw_view = self.bv.file.raw

            if raw_view:
                # Read all bytes from the raw view - this is the original binary
                # Use .end property to get the size (BinaryView doesn't support len())
                size = raw_view.end
                data = raw_view.read(0, size)
                return hashlib.sha256(data).hexdigest()

            # Fallback: try parent_view chain to find the raw data
            view = self.bv
            while view and hasattr(view, 'parent_view') and view.parent_view:
                view = view.parent_view
            if view and view != self.bv:
                size = view.end
                data = view.read(0, size)
                return hashlib.sha256(data).hexdigest()

        except Exception as e:
            log.log_error(f"Error computing SHA256: {e}")

        return None

    def handle_query(self):
        """Handle query request."""
        sha256 = self._get_sha256()
        if not sha256:
            self._show_error("No Binary", "No binary loaded or unable to compute hash.")
            return

        log.log_info(f"Querying SymGraph for: {sha256}")
        self.view.set_query_status("Checking...")
        self.view.hide_stats()
        self.view.set_buttons_enabled(False)

        # Start query worker
        self.query_worker = QueryWorker(sha256)
        self.query_worker.query_complete.connect(self._on_query_complete)
        self.query_worker.query_error.connect(self._on_query_error)
        self.query_worker.finished.connect(lambda: self.view.set_buttons_enabled(True))
        self.query_worker.start()

    def _on_query_complete(self, result: QueryResult):
        """Handle query completion."""
        self.view.set_buttons_enabled(True)

        if result.error:
            self.view.set_query_status(f"Error: {result.error}", found=False)
            return

        if result.exists:
            self.view.set_query_status("Found in SymGraph", found=True)
            if result.stats:
                self.view.set_stats(
                    symbols=result.stats.symbol_count,
                    functions=result.stats.function_count,
                    nodes=result.stats.graph_node_count,
                    last_updated=result.stats.last_queried_at
                )
        else:
            self.view.set_query_status("Not found in SymGraph", found=False)
            self.view.hide_stats()

    def _on_query_error(self, error_msg: str):
        """Handle query error."""
        self.view.set_buttons_enabled(True)
        self.view.set_query_status(f"Error: {error_msg}", found=False)
        log.log_error(f"Query error: {error_msg}")

    def handle_push(self, scope: str, push_symbols: bool, push_graph: bool):
        """Handle push request."""
        sha256 = self._get_sha256()
        if not sha256:
            self._show_error("No Binary", "No binary loaded or unable to compute hash.")
            return

        if not symgraph_service.has_api_key:
            self._show_error("API Key Required",
                "Push requires a SymGraph API key.\n\n"
                "Add your API key in Settings > SymGraph")
            return

        log.log_info(f"Pushing to SymGraph: scope={scope}, symbols={push_symbols}, graph={push_graph}")
        self.view.set_push_status("Pushing...", success=None)
        self.view.set_buttons_enabled(False)

        # Collect data to push
        symbols_data = []
        graph_data = None

        if push_symbols:
            symbols_data = self._collect_local_symbols(scope)
            log.log_info(f"Collected {len(symbols_data)} symbols to push")

        if push_graph:
            graph_data = self._collect_local_graph(scope)
            if graph_data:
                log.log_info(f"Collected graph data: {len(graph_data.get('nodes', []))} nodes")

        if not symbols_data and not graph_data:
            self.view.set_push_status("No data to push", success=False)
            self.view.set_buttons_enabled(True)
            return

        # Collect fingerprints for matching (BuildID for ELF, PDB GUID for PE)
        fingerprints = self._collect_fingerprints()

        # Start push worker
        self.push_worker = PushWorker(sha256, symbols_data, graph_data, fingerprints)
        self.push_worker.push_complete.connect(self._on_push_complete)
        self.push_worker.push_error.connect(self._on_push_error)
        self.push_worker.finished.connect(lambda: self.view.set_buttons_enabled(True))
        self.push_worker.start()

    def _on_push_complete(self, result: PushResult):
        """Handle push completion."""
        self.view.set_buttons_enabled(True)

        if result.success:
            msg_parts = []
            if result.symbols_pushed > 0:
                msg_parts.append(f"{result.symbols_pushed} symbols")
            if result.nodes_pushed > 0:
                msg_parts.append(f"{result.nodes_pushed} nodes")
            if result.edges_pushed > 0:
                msg_parts.append(f"{result.edges_pushed} edges")

            msg = "Pushed: " + ", ".join(msg_parts) if msg_parts else "Push complete"
            self.view.set_push_status(msg, success=True)
        else:
            self.view.set_push_status(f"Failed: {result.error or 'Unknown error'}", success=False)

    def _on_push_error(self, error_msg: str):
        """Handle push error."""
        self.view.set_buttons_enabled(True)
        self.view.set_push_status(f"Error: {error_msg}", success=False)
        log.log_error(f"Push error: {error_msg}")

    def handle_pull_preview(self):
        """Handle pull preview request."""
        sha256 = self._get_sha256()
        if not sha256:
            self._show_error("No Binary", "No binary loaded or unable to compute hash.")
            return

        if not self.bv:
            self._show_error("No Binary", "No binary view available.")
            return

        if not symgraph_service.has_api_key:
            self._show_error("API Key Required",
                "Pull requires a SymGraph API key.\n\n"
                "Add your API key in Settings > SymGraph")
            return

        # If worker is running, cancel it
        if self.pull_worker and self.pull_worker.isRunning():
            self.pull_worker.cancel()
            self.view.set_pull_status("Stopping...", success=None)
            return

        # Get pull configuration from view
        pull_config = self.view.get_pull_config()
        symbol_types = pull_config.get('symbol_types', [])

        if not symbol_types:
            self._show_error("No Types Selected", "Select at least one symbol type to pull.")
            return

        log.log_info(f"Fetching symbols from SymGraph: {sha256} (types: {symbol_types})")
        self._graph_export = None
        self._graph_stats = None
        self.view.clear_graph_preview_data()
        self.view.set_pull_status("Fetching...", success=None)
        self.view.clear_conflicts()
        self.view.set_buttons_enabled(False)
        self.view.set_pull_button_text("Stop")

        # Start pull preview worker with binary view and pull configuration
        self.pull_worker = PullPreviewWorker(sha256, self.bv, pull_config)
        self.pull_worker.progress.connect(self._on_pull_preview_progress)
        self.pull_worker.preview_complete.connect(self._on_pull_preview_complete)
        self.pull_worker.preview_error.connect(self._on_pull_preview_error)
        self.pull_worker.finished.connect(self._on_pull_preview_finished)
        self.pull_worker.start()

    def _on_pull_preview_progress(self, status: str):
        """Handle pull preview progress update."""
        self.view.set_pull_status(status, success=None)

    def _on_pull_preview_complete(self, conflicts: List[ConflictEntry], graph_export=None, graph_stats=None):
        """Handle pull preview completion."""
        if graph_export is not None and not graph_stats:
            graph_stats = PullPreviewWorker._get_graph_stats(graph_export)
        self._graph_export = graph_export
        self._graph_stats = graph_stats
        self.view.set_graph_preview_data(graph_export, graph_stats)

        if not conflicts and not graph_export:
            self.view.set_pull_status("No symbols found", success=False)
            return

        # Populate the conflict resolution table
        self.view.populate_conflicts(conflicts)

        # Calculate counts for status message
        conflict_count = sum(1 for c in conflicts if c.action == ConflictAction.CONFLICT)
        new_count = sum(1 for c in conflicts if c.action == ConflictAction.NEW)
        same_count = sum(1 for c in conflicts if c.action == ConflictAction.SAME)

        status_msg = f"Found {len(conflicts)} symbols ({conflict_count} conflicts, {new_count} new, {same_count} same)"
        if graph_stats:
            status_msg += (
                f" | Graph: {graph_stats.get('nodes', 0)} nodes, "
                f"{graph_stats.get('edges', 0)} edges, {graph_stats.get('communities', 0)} communities"
            )

        if not conflicts and graph_export:
            status_msg = "No symbols found (graph data available)"

        self.view.set_pull_status(status_msg, success=True)

    def _on_pull_preview_finished(self):
        """Handle pull preview worker finished (cleanup)."""
        self.view.set_buttons_enabled(True)
        self.view.set_pull_button_text("Pull & Preview")

    def _on_pull_preview_error(self, error_msg: str):
        """Handle pull preview error."""
        self._graph_export = None
        self._graph_stats = None
        self.view.clear_graph_preview_data()
        self.view.set_buttons_enabled(True)
        self.view.set_pull_status(f"Error: {error_msg}", success=False)
        log.log_error(f"Pull preview error: {error_msg}")

    def handle_apply_selected(self, addresses: List[int]):
        """Handle applying selected symbols."""
        # If worker is running, cancel it (check first to allow Stop button)
        if self.apply_worker and self.apply_worker.isRunning():
            self.apply_worker.cancel()
            self.view.set_pull_status("Stopping...", success=None)
            return

        if not addresses and not self._graph_export:
            self.view.set_pull_status("No items selected", success=False)
            return

        if not self.bv:
            self._show_error("No Binary", "No binary loaded.")
            return

        # Get the selected items (Symbol or ConflictEntry objects)
        selected_items = self.view.get_selected_conflicts()
        if not selected_items and not self._graph_export:
            self.view.set_pull_status("No items selected", success=False)
            return

        log.log_info(f"Applying {len(selected_items)} selected symbols in background")
        self.view.set_pull_status(f"Applying 0/{len(selected_items)}...", success=None)
        self.view.set_buttons_enabled(False)
        self.view.set_apply_button_text("Stop")

        # Start apply worker
        merge_policy = self.view.get_graph_merge_policy() if self._graph_export else "upsert"
        binary_hash = self._get_sha256() or ""
        self.apply_worker = ApplySymbolsWorker(
            self.bv,
            selected_items,
            graph_export=self._graph_export,
            merge_policy=merge_policy,
            binary_hash=binary_hash
        )
        self.apply_worker.progress.connect(self._on_apply_progress)
        self.apply_worker.apply_complete.connect(self._on_apply_complete)
        self.apply_worker.apply_cancelled.connect(self._on_apply_cancelled)
        self.apply_worker.apply_error.connect(self._on_apply_error)
        self.apply_worker.finished.connect(self._on_apply_finished)
        self.apply_worker.start()

    def handle_apply_all_new(self):
        """Handle applying all NEW symbols (wizard shortcut)."""
        if not self.bv:
            self._show_error("No Binary", "No binary loaded.")
            return

        # Get all NEW conflict entries
        new_items = self.view.get_all_new_conflicts()
        if not new_items and not self._graph_export:
            self.view.set_pull_status("No new items to apply", success=False)
            return

        log.log_info(f"Applying all {len(new_items)} new symbols")
        apply_message = f"Applying {len(new_items)} new symbols..."
        if not new_items and self._graph_export:
            apply_message = "Applying graph data..."
        self.view.show_applying_page(apply_message)
        self.view.set_buttons_enabled(False)

        # Start apply worker
        merge_policy = self.view.get_graph_merge_policy() if self._graph_export else "upsert"
        binary_hash = self._get_sha256() or ""
        self.apply_worker = ApplySymbolsWorker(
            self.bv,
            new_items,
            graph_export=self._graph_export,
            merge_policy=merge_policy,
            binary_hash=binary_hash
        )
        self.apply_worker.progress.connect(self._on_wizard_apply_progress)
        self.apply_worker.apply_complete.connect(self._on_wizard_apply_complete)
        self.apply_worker.apply_cancelled.connect(self._on_wizard_apply_cancelled)
        self.apply_worker.apply_error.connect(self._on_apply_error)
        self.apply_worker.finished.connect(self._on_apply_finished)
        self.apply_worker.start()

    def _on_wizard_apply_progress(self, current: int, total: int, message: str):
        """Handle apply progress update for wizard mode."""
        self.view.update_apply_progress(current, total, message)

    def _on_wizard_apply_complete(self, applied: int, errors: int):
        """Handle apply completion for wizard mode."""
        self.view.show_complete_page(applied, errors)
        log.log_info(f"Applied {applied} symbols, {errors} errors")

    def _on_wizard_apply_cancelled(self, applied: int):
        """Handle apply cancellation for wizard mode."""
        self.view.show_complete_page(applied, 0)
        log.log_info(f"Apply cancelled after {applied} symbols")

    def _on_apply_progress(self, current: int, total: int, message: str):
        """Handle apply progress update."""
        self.view.set_pull_status(message, success=None)

    def _on_apply_complete(self, applied: int, errors: int):
        """Handle apply completion."""
        if errors > 0:
            self.view.set_pull_status(f"Applied {applied} symbols ({errors} errors)", success=True)
        else:
            self.view.set_pull_status(f"Applied {applied} symbols", success=True)
        log.log_info(f"Applied {applied} symbols, {errors} errors")

    def _on_apply_cancelled(self, applied: int):
        """Handle apply cancellation."""
        self.view.set_pull_status(f"Stopped ({applied} symbols applied)", success=None)
        log.log_info(f"Apply cancelled after {applied} symbols")

    def _on_apply_error(self, error_msg: str):
        """Handle apply error."""
        self.view.set_pull_status(f"Error: {error_msg}", success=False)
        log.log_error(f"Apply error: {error_msg}")

    def _on_apply_finished(self):
        """Handle worker finished (cleanup)."""
        self.view.set_buttons_enabled(True)
        self.view.set_apply_button_text("Apply Selected")

    # === Helper methods for data collection ===

    def _collect_fingerprints(self) -> List[Dict[str, str]]:
        """
        Collect fingerprints from the binary for debug symbol matching.

        Returns:
            List of fingerprint dicts with 'type' and 'value' keys.
            - For ELF: BuildID (build_id)
            - For PE: PDB GUID (pdb_guid)
        """
        fingerprints = []

        if not self.bv:
            return fingerprints

        try:
            # Check binary format
            view_type = self.bv.view_type

            if view_type == 'ELF':
                # Extract BuildID from ELF binary
                build_id = self._extract_elf_build_id()
                if build_id:
                    fingerprints.append({'type': 'build_id', 'value': build_id})
                    log.log_info(f"Extracted ELF BuildID: {build_id}")

            elif view_type == 'PE':
                # Extract PDB GUID from PE binary
                pdb_info = self._extract_pe_pdb_info()
                if pdb_info:
                    fingerprints.append({'type': 'pdb_guid', 'value': pdb_info})
                    log.log_info(f"Extracted PDB GUID: {pdb_info}")

        except Exception as e:
            log.log_warn(f"Error collecting fingerprints: {e}")

        return fingerprints

    def _extract_elf_build_id(self) -> Optional[str]:
        """Extract GNU BuildID from ELF binary."""
        if not self.bv:
            return None

        try:
            # Try to find .note.gnu.build-id section
            for section in self.bv.sections.values():
                if section.name == '.note.gnu.build-id':
                    # Read the note section
                    data = self.bv.read(section.start, section.length)
                    if len(data) >= 16:
                        # GNU note format: namesz (4), descsz (4), type (4), name, desc
                        # name = "GNU\0", type = NT_GNU_BUILD_ID (3)
                        namesz = int.from_bytes(data[0:4], 'little')
                        descsz = int.from_bytes(data[4:8], 'little')
                        note_type = int.from_bytes(data[8:12], 'little')

                        if note_type == 3:  # NT_GNU_BUILD_ID
                            # Name is padded to 4-byte boundary
                            name_end = 12 + ((namesz + 3) & ~3)
                            if len(data) >= name_end + descsz:
                                build_id_bytes = data[name_end:name_end + descsz]
                                return build_id_bytes.hex()

            # Fallback: try raw file for build-id
            if hasattr(self.bv, 'file') and hasattr(self.bv.file, 'raw'):
                raw = self.bv.file.raw
                # Search for GNU build-id note in raw data
                # This is a simplified search - production code should parse ELF properly
                marker = b'GNU\x00'
                for section in self.bv.sections.values():
                    if 'note' in section.name.lower() or 'build' in section.name.lower():
                        data = self.bv.read(section.start, min(section.length, 256))
                        if marker in data:
                            idx = data.index(marker)
                            if idx >= 12:
                                # Read descsz from before the marker
                                descsz = int.from_bytes(data[idx-8:idx-4], 'little')
                                if descsz > 0 and descsz <= 64:
                                    name_end = idx + 4  # Skip "GNU\0"
                                    # Align to 4 bytes
                                    name_end = (name_end + 3) & ~3
                                    if len(data) >= name_end + descsz:
                                        return data[name_end:name_end + descsz].hex()

        except Exception as e:
            log.log_debug(f"Error extracting ELF BuildID: {e}")

        return None

    def _extract_pe_pdb_info(self) -> Optional[str]:
        """Extract PDB GUID from PE binary."""
        if not self.bv:
            return None

        try:
            # Binary Ninja stores debug info in metadata
            # Try to get PDB path/GUID from the binary view
            if hasattr(self.bv, 'get_metadata'):
                metadata = self.bv.get_metadata('pdb')
                if metadata:
                    return str(metadata)

            # Alternative: check for CodeView debug directory
            # This would require parsing PE debug directory - complex
            # For now, we'll rely on metadata if available

        except Exception as e:
            log.log_debug(f"Error extracting PE PDB info: {e}")

        return None

    def _collect_local_symbols(self, scope: str) -> List[Dict[str, Any]]:
        """Collect all symbol types from Binary Ninja based on scope."""
        symbols = []

        if not self.bv:
            return symbols

        try:
            if scope == PushScope.CURRENT_FUNCTION.value:
                # Get current function if available
                current_func = self._get_current_function()
                if current_func:
                    symbols.append(self._function_to_symbol_dict(current_func))
                    # Also collect comments within this function
                    symbols.extend(self._collect_function_comments(current_func))
                    # Collect local variables for this function
                    symbols.extend(self._collect_function_variables(current_func))
            else:
                # Full binary - all symbol types
                # 1. Functions
                for func in self.bv.functions:
                    symbols.append(self._function_to_symbol_dict(func))

                # 2. Data variables (global variables)
                symbols.extend(self._collect_data_variables())

                # 3. Types and enums
                symbols.extend(self._collect_types_and_enums())

                # 4. Comments (address-level comments)
                symbols.extend(self._collect_comments())

        except Exception as e:
            log.log_error(f"Error collecting symbols: {e}")

        return symbols

    def _function_to_symbol_dict(self, func) -> Dict[str, Any]:
        """Convert a Binary Ninja function to a symbol dictionary."""
        # Get function type/signature if available
        data_type = None
        try:
            if func.type:
                data_type = str(func.type)
        except Exception:
            pass

        is_auto = is_default_name(func.name)
        return {
            'address': f"0x{func.start:x}",
            'symbol_type': 'function',
            'name': func.name,
            'data_type': data_type,
            'confidence': 0.5 if is_auto else 0.9,
            'provenance': 'decompiler' if is_auto else 'user'
        }

    def _collect_data_variables(self) -> List[Dict[str, Any]]:
        """Collect global data variables from Binary Ninja."""
        symbols = []
        try:
            for addr, var in self.bv.data_vars.items():
                # Get symbol name if available
                sym = self.bv.get_symbol_at(addr)
                name = sym.name if sym else None

                # Skip variables without names
                if not name:
                    continue

                # Check for auto-generated names
                is_auto = is_default_name(name)
                if is_auto:
                    confidence = 0.3
                    provenance = 'decompiler'
                else:
                    confidence = 0.85
                    provenance = 'user'

                symbols.append({
                    'address': f"0x{addr:x}",
                    'symbol_type': 'variable',
                    'name': name,
                    'data_type': str(var.type) if var.type else None,
                    'confidence': confidence,
                    'provenance': provenance
                })
        except Exception as e:
            log.log_error(f"Error collecting data variables: {e}")
        return symbols

    def _collect_function_variables(self, func) -> List[Dict[str, Any]]:
        """Collect local variables from a function with storage identification."""
        from binaryninja.enums import VariableSourceType

        symbols = []
        try:
            # Function parameters - use enumeration index for ordering
            for param_idx, param in enumerate(func.parameter_vars):
                if param.name:
                    is_auto = is_default_name(param.name)

                    metadata = {
                        'scope': 'parameter',
                        'function': func.name,
                        'storage_class': 'parameter',
                        'parameter_index': param_idx
                    }

                    # Also capture storage location for full info
                    if param.source_type == VariableSourceType.RegisterVariableSourceType:
                        try:
                            reg_name = self.bv.arch._regs_by_index.get(param.storage)
                            if reg_name:
                                metadata['register'] = reg_name
                        except:
                            pass
                    elif param.source_type == VariableSourceType.StackVariableSourceType:
                        metadata['stack_offset'] = param.storage

                    symbols.append({
                        'address': f"0x{func.start:x}",
                        'symbol_type': 'variable',
                        'name': param.name,
                        'data_type': str(param.type) if param.type else None,
                        'confidence': 0.3 if is_auto else 0.8,
                        'provenance': 'decompiler' if is_auto else 'user',
                        'metadata': metadata
                    })

            # Local variables (stack and register)
            for var in func.vars:
                if var.name:
                    is_auto = is_default_name(var.name)

                    metadata = {
                        'scope': 'local',
                        'function': func.name
                    }

                    # Determine storage class and identifier
                    if var.source_type == VariableSourceType.StackVariableSourceType:
                        metadata['storage_class'] = 'stack'
                        metadata['stack_offset'] = var.storage
                    elif var.source_type == VariableSourceType.RegisterVariableSourceType:
                        metadata['storage_class'] = 'register'
                        try:
                            reg_name = self.bv.arch._regs_by_index.get(var.storage)
                            metadata['register'] = reg_name if reg_name else f"reg_{var.storage}"
                        except:
                            metadata['register'] = f"reg_{var.storage}"
                    else:
                        metadata['storage_class'] = 'other'
                        metadata['storage_id'] = var.storage

                    symbols.append({
                        'address': f"0x{func.start:x}",
                        'symbol_type': 'variable',
                        'name': var.name,
                        'data_type': str(var.type) if var.type else None,
                        'confidence': 0.3 if is_auto else 0.75,
                        'provenance': 'decompiler' if is_auto else 'user',
                        'metadata': metadata
                    })
        except Exception as e:
            log.log_error(f"Error collecting function variables: {e}")
        return symbols

    def _collect_types_and_enums(self) -> List[Dict[str, Any]]:
        """Collect user-defined types and enums from Binary Ninja."""
        symbols = []
        enum_count = 0
        struct_count = 0
        try:
            import binaryninja

            # Iterate through all types in the binary
            for name, type_obj in self.bv.types.items():
                type_str = str(type_obj)
                symbol_type = 'type'
                metadata = None
                should_append = True

                try:
                    # Use integer comparison for type_class (more reliable)
                    # EnumerationTypeClass = 5, StructureTypeClass = 4
                    type_class_int = int(type_obj.type_class)

                    # Check type class for enum (EnumerationTypeClass = 5)
                    # For EnumerationType, .members is directly on type_obj
                    if type_class_int == 5:
                        symbol_type = 'enum'
                        enum_count += 1
                        # Access members directly on the type object
                        members_list = getattr(type_obj, 'members', None)
                        if members_list is not None:
                            members = {}
                            content_lines = [f"enum {name} {{"]
                            for member in members_list:
                                # EnumerationMember has .name and .value properties
                                member_name = getattr(member, 'name', None)
                                member_value = getattr(member, 'value', None)
                                if member_name is not None and member_value is not None:
                                    members[member_name] = member_value
                                    content_lines.append(f"    {member_name} = 0x{member_value:x},")
                            content_lines.append("}")
                            if members:
                                metadata = {'members': members}
                                # Store human-readable content
                                type_str = "\n".join(content_lines)
                            else:
                                # Empty enum - still push but with empty members
                                metadata = {'members': {}}
                        else:
                            log.log_warn(f"Skipping enum '{name}': could not access members")
                            should_append = False

                    # Check type class for struct (StructureTypeClass = 4)
                    # For StructureType, .members is directly on type_obj
                    elif type_class_int == 4:
                        symbol_type = 'struct'
                        struct_count += 1
                        # Access members directly on the type object
                        members_list = getattr(type_obj, 'members', None)
                        if members_list is not None:
                            fields = []
                            content_lines = [f"struct {name} {{"]
                            for member in members_list:
                                # StructureMember has .name, .type, .offset properties
                                field_name = getattr(member, 'name', None)
                                field_type_obj = getattr(member, 'type', None)
                                field_type = str(field_type_obj) if field_type_obj else 'unknown'
                                field_offset = getattr(member, 'offset', 0)
                                fields.append({
                                    'name': field_name,
                                    'type': field_type,
                                    'offset': field_offset
                                })
                                content_lines.append(f"    /* 0x{field_offset:02x} */ {field_type} {field_name};")
                            content_lines.append("}")
                            metadata = {'fields': fields}
                            # Store human-readable content
                            type_str = "\n".join(content_lines)
                        else:
                            log.log_warn(f"Skipping struct '{name}': could not access members")
                            should_append = False

                    # For other types (typedefs, pointers, etc.), push as 'type' without metadata
                    # This is fine - they don't have members

                except Exception as e:
                    log.log_warn(f"Could not get type details for {name}: {e}")
                    should_append = False

                if should_append:
                    symbol_data = {
                        'address': '0x0',  # Types don't have addresses
                        'symbol_type': symbol_type,
                        'name': str(name),
                        'data_type': type_str,
                        'confidence': 0.9,
                        'provenance': 'user',
                        'metadata': metadata
                    }
                    # For structs and enums, also populate the content field
                    # with the human-readable definition (same as data_type)
                    if symbol_type in ('struct', 'enum'):
                        symbol_data['content'] = type_str
                    symbols.append(symbol_data)
        except Exception as e:
            log.log_error(f"Error collecting types and enums: {e}")

        log.log_info(f"Collected {len(symbols)} types ({enum_count} enums, {struct_count} structs)")
        return symbols

    def _collect_comments(self) -> List[Dict[str, Any]]:
        """Collect address-level comments from Binary Ninja."""
        symbols = []
        try:
            # Collect function comments
            for func in self.bv.functions:
                # Function-level comment
                if func.comment:
                    symbols.append({
                        'address': f"0x{func.start:x}",
                        'symbol_type': 'comment',
                        'name': None,
                        'content': func.comment,
                        'confidence': 1.0,
                        'provenance': 'user',
                        'metadata': {'type': 'function'}
                    })

                # Address comments within the function
                for addr, comment in func.comments.items():
                    if comment:
                        symbols.append({
                            'address': f"0x{addr:x}",
                            'symbol_type': 'comment',
                            'name': None,
                            'content': comment,
                            'confidence': 1.0,
                            'provenance': 'user',
                            'metadata': {'type': 'address', 'function': func.name}
                        })
        except Exception as e:
            log.log_error(f"Error collecting comments: {e}")
        return symbols

    def _collect_function_comments(self, func) -> List[Dict[str, Any]]:
        """Collect comments within a specific function."""
        symbols = []
        try:
            # Function-level comment
            if func.comment:
                symbols.append({
                    'address': f"0x{func.start:x}",
                    'symbol_type': 'comment',
                    'name': None,
                    'content': func.comment,
                    'confidence': 1.0,
                    'provenance': 'user',
                    'metadata': {'type': 'function'}
                })

            # Address comments within the function
            for addr, comment in func.comments.items():
                if comment:
                    symbols.append({
                        'address': f"0x{addr:x}",
                        'symbol_type': 'comment',
                        'name': None,
                        'content': comment,
                        'confidence': 1.0,
                        'provenance': 'user',
                        'metadata': {'type': 'address', 'function': func.name}
                    })
        except Exception as e:
            log.log_error(f"Error collecting function comments: {e}")
        return symbols

    def _collect_local_graph(self, scope: str) -> Optional[Dict[str, Any]]:
        """Collect graph data from local graph store (with rich metadata) or fallback to Binary Ninja."""
        if not self.bv:
            return None

        try:
            nodes = []
            edges = []

            # Get binary hash for graph store queries
            binary_hash = self._get_sha256()
            if not binary_hash:
                return self._collect_minimal_graph(scope)

            # Try to read from local graph store first
            graph_store = GraphStore(analysis_db_service)

            if scope == PushScope.CURRENT_FUNCTION.value:
                current_func = self._get_current_function()
                if current_func:
                    # Try to get rich node data from graph store
                    local_node = graph_store.get_node_by_address(
                        binary_hash, "FUNCTION", current_func.start
                    )
                    if local_node:
                        nodes.append(self._local_node_to_push_dict(local_node))
                        # Get edges with weights from graph store
                        graph_edges = graph_store.get_edges_for_node(binary_hash, local_node.id)
                        for edge in graph_edges:
                            edge_dict = self._local_edge_to_push_dict(edge, graph_store)
                            if edge_dict:
                                edges.append(edge_dict)
                    else:
                        # Fallback: create minimal node from Binary Ninja
                        nodes.append(self._function_to_node_dict(current_func))
                        for callee in current_func.callees:
                            edges.append({
                                'source_address': f"0x{current_func.start:x}",
                                'target_address': f"0x{callee.start:x}",
                                'edge_type': 'calls',
                                'weight': 1.0
                            })
            else:
                # Full binary scope
                local_nodes = graph_store.get_nodes_by_type(binary_hash, "FUNCTION")

                if local_nodes:
                    # Build node ID to address mapping for edge resolution
                    node_id_to_address = {}
                    for local_node in local_nodes:
                        nodes.append(self._local_node_to_push_dict(local_node))
                        node_id_to_address[local_node.id] = local_node.address

                    # Get all edges from graph store
                    all_edges = graph_store.get_edges_by_types(
                        binary_hash,
                        ["calls", "calls_vulnerable", "network_send", "network_recv",
                         "taint_flows_to", "similar_purpose", "references"]
                    )
                    for edge in all_edges:
                        edge_dict = self._local_edge_to_push_dict(edge, graph_store, node_id_to_address)
                        if edge_dict:
                            edges.append(edge_dict)
                else:
                    # Fallback: create minimal graph from Binary Ninja
                    for func in self.bv.functions:
                        nodes.append(self._function_to_node_dict(func))
                        for callee in func.callees:
                            edges.append({
                                'source_address': f"0x{func.start:x}",
                                'target_address': f"0x{callee.start:x}",
                                'edge_type': 'calls',
                                'weight': 1.0
                            })

            if nodes:
                return {'nodes': nodes, 'edges': edges}

        except Exception as e:
            log.log_error(f"Error collecting graph: {e}")

        return None

    def _collect_minimal_graph(self, scope: str) -> Optional[Dict[str, Any]]:
        """Fallback: Collect minimal graph data directly from Binary Ninja."""
        if not self.bv:
            return None

        try:
            nodes = []
            edges = []

            if scope == PushScope.CURRENT_FUNCTION.value:
                current_func = self._get_current_function()
                if current_func:
                    nodes.append(self._function_to_node_dict(current_func))
                    for callee in current_func.callees:
                        edges.append({
                            'source_address': f"0x{current_func.start:x}",
                            'target_address': f"0x{callee.start:x}",
                            'edge_type': 'calls',
                            'weight': 1.0
                        })
            else:
                for func in self.bv.functions:
                    nodes.append(self._function_to_node_dict(func))
                    for callee in func.callees:
                        edges.append({
                            'source_address': f"0x{func.start:x}",
                            'target_address': f"0x{callee.start:x}",
                            'edge_type': 'calls',
                            'weight': 1.0
                        })

            if nodes:
                return {'nodes': nodes, 'edges': edges}

        except Exception as e:
            log.log_error(f"Error collecting minimal graph: {e}")

        return None

    def _function_to_node_dict(self, func) -> Dict[str, Any]:
        """Convert a Binary Ninja function to a minimal graph node dictionary (fallback)."""
        return {
            'address': f"0x{func.start:x}",
            'node_type': 'function',
            'name': func.name,
            'raw_content': None,
            'llm_summary': None,
            'confidence': 0.0,
            'provenance': 'decompiler',
            'is_stale': True,
            'user_edited': False
        }

    def _local_node_to_push_dict(self, node: LocalGraphNode) -> Dict[str, Any]:
        """Convert a local GraphNode to push format with all rich metadata."""
        # Determine confidence - fix up old nodes that were analyzed before confidence was set
        confidence = node.confidence or 0.0
        if node.llm_summary and confidence == 0.0:
            # Node was analyzed with old code that didn't set confidence
            # Assign reasonable defaults based on provenance
            confidence = 0.95 if node.user_edited else 0.85
            log.log_debug(f"Fixed up confidence for {node.name}: {confidence}")

        result = {
            'address': f"0x{node.address:x}" if node.address else "0x0",
            'node_type': node.get_node_type_str().lower(),
            'name': node.name,
            'raw_content': node.raw_code,
            'llm_summary': node.llm_summary,
            'confidence': confidence,
            'provenance': 'user' if node.user_edited else 'decompiler',
        }

        # Add security-related fields if present
        if node.security_flags:
            result['security_flags'] = list(node.security_flags)
        if node.network_apis:
            result['network_apis'] = list(node.network_apis)
        if node.file_io_apis:
            result['file_io_apis'] = list(node.file_io_apis)
        if node.ip_addresses:
            result['ip_addresses'] = list(node.ip_addresses)
        if node.urls:
            result['urls'] = list(node.urls)
        if node.file_paths:
            result['file_paths'] = list(node.file_paths)
        if node.domains:
            result['domains'] = list(node.domains)
        if node.registry_keys:
            result['registry_keys'] = list(node.registry_keys)

        # Add analysis metadata
        if node.risk_level:
            result['risk_level'] = node.risk_level
        if node.activity_profile:
            result['activity_profile'] = node.activity_profile
        if node.analysis_depth:
            result['analysis_depth'] = node.analysis_depth

        # Add state flags
        result['is_stale'] = node.is_stale
        result['user_edited'] = node.user_edited

        return result

    def _local_edge_to_push_dict(
        self,
        edge: LocalGraphEdge,
        graph_store: GraphStore,
        node_id_to_address: Optional[Dict[str, int]] = None
    ) -> Optional[Dict[str, Any]]:
        """Convert a local GraphEdge to push format with weight."""
        source_addr = None
        target_addr = None

        if node_id_to_address:
            # Use cached mapping for efficiency
            source_addr = node_id_to_address.get(edge.source_id)
            target_addr = node_id_to_address.get(edge.target_id)
        else:
            # Look up nodes by ID
            source_node = graph_store.get_node_by_id(edge.source_id)
            target_node = graph_store.get_node_by_id(edge.target_id)
            if source_node:
                source_addr = source_node.address
            if target_node:
                target_addr = target_node.address

        if source_addr is None or target_addr is None:
            return None

        return {
            'source_address': f"0x{source_addr:x}",
            'target_address': f"0x{target_addr:x}",
            'edge_type': edge.get_edge_type_str(),
            'weight': edge.weight or 1.0
        }

    def _get_current_function(self):
        """Get the current function from the frame if available."""
        if self.frame and hasattr(self.frame, 'getCurrentFunction'):
            return self.frame.getCurrentFunction()
        elif self.bv and self.data and hasattr(self.data, 'current_address'):
            # Try to get function at current address
            funcs = self.bv.get_functions_containing(self.data.current_address)
            if funcs:
                return funcs[0]
        return None

    # === Utility methods ===

    def _show_error(self, title: str, message: str):
        """Show error message dialog."""
        QMessageBox.critical(self.view, title, message)

    def _show_info(self, title: str, message: str):
        """Show info message dialog."""
        QMessageBox.information(self.view, title, message)

    def _show_warning(self, title: str, message: str):
        """Show warning message dialog."""
        QMessageBox.warning(self.view, title, message)
