#!/usr/bin/env python3
"""
SymGraph.ai Controller for BinAssist.

This controller manages the SymGraph.ai tab functionality including
querying, pushing, and pulling symbols and graph data.
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable
from PySide6.QtWidgets import QMessageBox
from PySide6.QtCore import QThread, Signal, QObject

from ..services.symgraph_service import (
    symgraph_service, SymGraphServiceError, SymGraphAuthError,
    SymGraphNetworkError, SymGraphAPIError
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
    """Worker thread for querying SymGraph.ai."""

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
    """Worker thread for pushing to SymGraph.ai."""

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
    """Worker thread for pulling symbols from SymGraph.ai and building conflicts."""

    progress = Signal(str)  # status message
    preview_complete = Signal(list)  # List[ConflictEntry]
    preview_error = Signal(str)

    def __init__(self, sha256: str, bv):
        super().__init__()
        self.sha256 = sha256
        self.bv = bv  # Binary Ninja binary view for getting local symbols
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            # Step 1: Fetch remote symbols from API
            self.progress.emit("Fetching function symbols...")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Use server-side filtering to only fetch function symbols
                remote_symbols = loop.run_until_complete(
                    symgraph_service.get_symbols(self.sha256, symbol_type='function')
                )
            finally:
                loop.close()

            if self._cancelled:
                return

            log.log_info(f"Fetched {len(remote_symbols)} function symbols from API")

            if not remote_symbols:
                self.preview_complete.emit([])
                return

            # Step 2: Get local symbols from Binary Ninja
            self.progress.emit("Collecting local symbols...")
            local_symbols = self._get_local_symbol_map()

            if self._cancelled:
                return

            log.log_info(f"Found {len(local_symbols)} local function symbols")

            # Step 3: Build conflict entries
            self.progress.emit("Building conflict list...")
            conflicts = symgraph_service.build_conflict_entries(local_symbols, remote_symbols)

            if self._cancelled:
                return

            log.log_info(f"Built {len(conflicts)} conflict entries")
            self.preview_complete.emit(conflicts)

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


class ApplySymbolsWorker(QThread):
    """Worker thread for applying symbols to Binary Ninja."""

    progress = Signal(int, int)  # current, total
    apply_complete = Signal(int, int)  # applied, errors
    apply_cancelled = Signal(int)  # applied so far
    apply_error = Signal(str)

    def __init__(self, bv, symbols: List):
        super().__init__()
        self.bv = bv
        self.symbols = symbols
        self._cancelled = False

    def cancel(self):
        """Request cancellation of the apply operation."""
        self._cancelled = True

    def run(self):
        try:
            import binaryninja
            total = len(self.symbols)
            applied = 0
            errors = 0

            for i, symbol in enumerate(self.symbols):
                if self._cancelled:
                    self.apply_cancelled.emit(applied)
                    return

                # Handle both Symbol objects and ConflictEntry objects
                if hasattr(symbol, 'remote_symbol'):
                    # It's a ConflictEntry
                    addr = symbol.address
                    name = symbol.remote_symbol.name if symbol.remote_symbol else None
                else:
                    # It's a Symbol object
                    addr = symbol.address
                    name = symbol.name

                if name:
                    try:
                        self._apply_symbol(addr, name)
                        applied += 1
                    except Exception as e:
                        log.log_error(f"Error applying symbol at 0x{addr:x}: {e}")
                        errors += 1

                self.progress.emit(i + 1, total)

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


class SymGraphController(QObject):
    """Controller for the SymGraph.ai tab functionality."""

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

        log.log_info(f"Querying SymGraph.ai for: {sha256}")
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
            self.view.set_query_status("Found in SymGraph.ai", found=True)
            if result.stats:
                self.view.set_stats(
                    symbols=result.stats.symbol_count,
                    functions=result.stats.function_count,
                    nodes=result.stats.graph_node_count,
                    last_updated=result.stats.last_queried_at
                )
        else:
            self.view.set_query_status("Not found in SymGraph.ai", found=False)
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
                "Push requires a SymGraph.ai API key.\n\n"
                "Add your API key in Settings > SymGraph.ai")
            return

        log.log_info(f"Pushing to SymGraph.ai: scope={scope}, symbols={push_symbols}, graph={push_graph}")
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
                "Pull requires a SymGraph.ai API key.\n\n"
                "Add your API key in Settings > SymGraph.ai")
            return

        # If worker is running, cancel it
        if self.pull_worker and self.pull_worker.isRunning():
            self.pull_worker.cancel()
            self.view.set_pull_status("Stopping...", success=None)
            return

        log.log_info(f"Fetching symbols from SymGraph.ai: {sha256}")
        self.view.set_pull_status("Fetching...", success=None)
        self.view.clear_conflicts()
        self.view.set_buttons_enabled(False)
        self.view.set_pull_button_text("Stop")

        # Start pull preview worker with binary view for local symbol collection
        self.pull_worker = PullPreviewWorker(sha256, self.bv)
        self.pull_worker.progress.connect(self._on_pull_preview_progress)
        self.pull_worker.preview_complete.connect(self._on_pull_preview_complete)
        self.pull_worker.preview_error.connect(self._on_pull_preview_error)
        self.pull_worker.finished.connect(self._on_pull_preview_finished)
        self.pull_worker.start()

    def _on_pull_preview_progress(self, status: str):
        """Handle pull preview progress update."""
        self.view.set_pull_status(status, success=None)

    def _on_pull_preview_complete(self, conflicts: List[ConflictEntry]):
        """Handle pull preview completion."""
        if not conflicts:
            self.view.set_pull_status("No symbols found", success=False)
            return

        # Populate the conflict resolution table
        self.view.populate_conflicts(conflicts)

        # Calculate counts for status message
        conflict_count = sum(1 for c in conflicts if c.action == ConflictAction.CONFLICT)
        new_count = sum(1 for c in conflicts if c.action == ConflictAction.NEW)
        same_count = sum(1 for c in conflicts if c.action == ConflictAction.SAME)

        self.view.set_pull_status(
            f"Found {len(conflicts)} symbols ({conflict_count} conflicts, {new_count} new, {same_count} same)",
            success=True
        )

    def _on_pull_preview_finished(self):
        """Handle pull preview worker finished (cleanup)."""
        self.view.set_buttons_enabled(True)
        self.view.set_pull_button_text("Pull & Preview")

    def _on_pull_preview_error(self, error_msg: str):
        """Handle pull preview error."""
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

        if not addresses:
            self.view.set_pull_status("No items selected", success=False)
            return

        if not self.bv:
            self._show_error("No Binary", "No binary loaded.")
            return

        # Get the selected items (Symbol or ConflictEntry objects)
        selected_items = self.view.get_selected_conflicts()
        if not selected_items:
            self.view.set_pull_status("No items selected", success=False)
            return

        log.log_info(f"Applying {len(selected_items)} selected symbols in background")
        self.view.set_pull_status(f"Applying 0/{len(selected_items)}...", success=None)
        self.view.set_buttons_enabled(False)
        self.view.set_apply_button_text("Stop")

        # Start apply worker
        self.apply_worker = ApplySymbolsWorker(self.bv, selected_items)
        self.apply_worker.progress.connect(self._on_apply_progress)
        self.apply_worker.apply_complete.connect(self._on_apply_complete)
        self.apply_worker.apply_cancelled.connect(self._on_apply_cancelled)
        self.apply_worker.apply_error.connect(self._on_apply_error)
        self.apply_worker.finished.connect(self._on_apply_finished)
        self.apply_worker.start()

    def _on_apply_progress(self, current: int, total: int):
        """Handle apply progress update."""
        self.view.set_pull_status(f"Applying {current}/{total}...", success=None)

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

        return {
            'address': f"0x{func.start:x}",
            'symbol_type': 'function',
            'name': func.name,
            'data_type': data_type,
            'confidence': 0.9 if not func.name.startswith('sub_') else 0.5,
            'provenance': 'user' if not func.name.startswith('sub_') else 'decompiler'
        }

    def _collect_data_variables(self) -> List[Dict[str, Any]]:
        """Collect global data variables from Binary Ninja."""
        symbols = []
        try:
            for addr, var in self.bv.data_vars.items():
                # Get symbol name if available
                sym = self.bv.get_symbol_at(addr)
                name = sym.name if sym else None

                # Skip auto-generated names
                if name and (name.startswith('data_') or name.startswith('byte_')):
                    confidence = 0.3
                    provenance = 'decompiler'
                else:
                    confidence = 0.85 if name else 0.3
                    provenance = 'user' if name else 'decompiler'

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
        """Collect local variables from a function."""
        symbols = []
        try:
            # Function parameters
            for param in func.parameter_vars:
                if param.name and not param.name.startswith('arg'):
                    symbols.append({
                        'address': f"0x{func.start:x}",
                        'symbol_type': 'variable',
                        'name': param.name,
                        'data_type': str(param.type) if param.type else None,
                        'confidence': 0.8,
                        'provenance': 'user',
                        'metadata': {'scope': 'parameter', 'function': func.name}
                    })

            # Local variables (stack and register)
            for var in func.vars:
                if var.name and not var.name.startswith('var_'):
                    symbols.append({
                        'address': f"0x{func.start:x}",
                        'symbol_type': 'variable',
                        'name': var.name,
                        'data_type': str(var.type) if var.type else None,
                        'confidence': 0.75,
                        'provenance': 'user',
                        'metadata': {'scope': 'local', 'function': func.name}
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
        """Collect graph data from Binary Ninja based on scope."""
        if not self.bv:
            return None

        try:
            nodes = []
            edges = []

            if scope == PushScope.CURRENT_FUNCTION.value:
                current_func = self._get_current_function()
                if current_func:
                    nodes.append(self._function_to_node_dict(current_func))
                    # Add edges for callees
                    for callee in current_func.callees:
                        edges.append({
                            'source_address': f"0x{current_func.start:x}",
                            'target_address': f"0x{callee.start:x}",
                            'edge_type': 'calls'
                        })
            else:
                # Full binary
                for func in self.bv.functions:
                    nodes.append(self._function_to_node_dict(func))
                    for callee in func.callees:
                        edges.append({
                            'source_address': f"0x{func.start:x}",
                            'target_address': f"0x{callee.start:x}",
                            'edge_type': 'calls'
                        })

            if nodes:
                return {'nodes': nodes, 'edges': edges}

        except Exception as e:
            log.log_error(f"Error collecting graph: {e}")

        return None

    def _function_to_node_dict(self, func) -> Dict[str, Any]:
        """Convert a Binary Ninja function to a graph node dictionary."""
        return {
            'address': f"0x{func.start:x}",
            'node_type': 'function',
            'name': func.name,
            'properties': {}
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
