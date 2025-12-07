#!/usr/bin/env python3

"""
Debounced Rendering Service for BinAssist

This module provides debounced HTML/Markdown rendering for streaming LLM responses.
Instead of rendering on every delta update (which can happen hundreds of times per response),
it accumulates deltas in a buffer and renders at fixed intervals (default: 1 second).

Performance benefits:
- 10-40x reduction in rendering calls
- Smooth visual experience (no stuttering/flickering)
- Lower CPU usage during streaming
- Better UI responsiveness

Based on the GhidrAssist implementation documented in Debounced_Rendering_v1.md
"""

from typing import Callable, Optional
from PySide6.QtCore import QTimer

try:
    import binaryninja
    log = binaryninja.log.Logger(0, "BinAssist")
except ImportError:
    # Fallback for testing outside Binary Ninja
    class MockLog:
        @staticmethod
        def log_info(msg): print(f"[BinAssist] INFO: {msg}")
        @staticmethod
        def log_debug(msg): print(f"[BinAssist] DEBUG: {msg}")
        @staticmethod
        def log_warn(msg): print(f"[BinAssist] WARN: {msg}")
    log = MockLog()


class DebouncedRenderer:
    """
    Debounced renderer for streaming LLM responses.

    Accumulates streaming deltas in a buffer and renders periodically at fixed
    intervals (default: 1 second) to avoid overwhelming the UI thread.

    IMPORTANT: All rendering happens on the Qt main thread to comply with
    Binary Ninja's threading requirements. The debouncing alone provides
    10-40x reduction in UI updates.

    Example usage:
        # In controller __init__:
        self.debouncer = DebouncedRenderer(
            update_callback=self._update_ui_with_content
        )

        # When streaming starts:
        self.debouncer.start()

        # On each delta:
        self.debouncer.on_delta(chunk)

        # When streaming completes:
        self.debouncer.complete(final_response)

        # On error or cancellation:
        self.debouncer.cancel()
    """

    RENDER_INTERVAL_MS = 1000  # 1 second - matches GhidrAssist proven performance

    def __init__(self, update_callback: Callable[[str], None],
                 render_interval_ms: Optional[int] = None):
        """
        Initialize the debounced renderer.

        Args:
            update_callback: Function to call with accumulated content.
                            This function should handle the actual UI update.
                            It will be called on the Qt main thread via QTimer.
            render_interval_ms: Optional custom render interval in milliseconds.
                              Default is 1000ms (1 second).
        """
        self.update_callback = update_callback
        self.render_interval_ms = render_interval_ms or self.RENDER_INTERVAL_MS

        # Buffer for accumulating deltas
        self.response_buffer = []

        # Qt timer for periodic rendering
        self.render_timer = QTimer()
        self.render_timer.timeout.connect(self._render_current_content)

        # State tracking
        self.is_active = False
        self._render_count = 0

        log.log_debug(f"DebouncedRenderer initialized with {self.render_interval_ms}ms interval")

    def start(self):
        """
        Start debounced rendering.

        Clears the buffer and starts the periodic render timer.
        Call this when LLM streaming begins.
        """
        self.is_active = True
        self.response_buffer.clear()
        self._render_count = 0

        # Start periodic rendering
        self.render_timer.start(self.render_interval_ms)

        log.log_debug(f"DebouncedRenderer started (interval: {self.render_interval_ms}ms)")

    def on_delta(self, delta: str):
        """
        Handle a new delta from the LLM stream.

        Appends the delta to the buffer. The actual rendering will happen
        on the next timer tick.

        Args:
            delta: The new content delta to accumulate
        """
        if not delta or not self.is_active:
            return

        self.response_buffer.append(delta)

    def _render_current_content(self):
        """
        Periodic render task executed by QTimer.

        This method is called every RENDER_INTERVAL_MS milliseconds.
        It snapshots the current buffer and calls the update callback.
        """
        # Early return if buffer is empty
        if not self.response_buffer:
            return

        # Snapshot buffer content (fast operation)
        content = ''.join(self.response_buffer)

        # Call the update callback with accumulated content
        # Note: QTimer.timeout is already on the Qt main thread, so this is thread-safe
        try:
            self.update_callback(content)
            self._render_count += 1
            log.log_debug(f"DebouncedRenderer: periodic render #{self._render_count} ({len(content)} chars)")
        except Exception as e:
            log.log_warn(f"DebouncedRenderer update callback failed: {e}")

    def complete(self, final_response: Optional[str] = None):
        """
        Complete streaming and perform final render.

        Stops the periodic timer and does one final render with either
        the provided final_response or the accumulated buffer.

        Args:
            final_response: Optional complete response to render.
                          If provided, this is used instead of the buffer.
                          Useful when the LLM provider sends a complete
                          response in the completion callback.
        """
        self.is_active = False
        self.render_timer.stop()

        # Determine final content
        if final_response:
            content = final_response
        else:
            content = ''.join(self.response_buffer)

        # Final render if we have content
        if content:
            try:
                self.update_callback(content)
                log.log_debug(f"DebouncedRenderer: final render ({len(content)} chars, {self._render_count} periodic renders)")
            except Exception as e:
                log.log_warn(f"DebouncedRenderer final update callback failed: {e}")

        # Clean up
        self.response_buffer.clear()

    def cancel(self):
        """
        Cancel debounced rendering.

        Stops the periodic timer and clears the buffer without performing
        a final render. Use this when the user cancels the query or an
        error occurs.
        """
        self.is_active = False
        self.render_timer.stop()
        self.response_buffer.clear()

        log.log_debug(f"DebouncedRenderer cancelled after {self._render_count} renders")

    def set_render_interval(self, interval_ms: int):
        """
        Update the render interval.

        This can be used for performance tuning:
        - Lower values (500ms): More responsive, higher CPU usage
        - Higher values (2000ms): Less CPU usage, chunkier updates
        - Recommended: 1000ms (1 second) - proven sweet spot

        Args:
            interval_ms: New interval in milliseconds
        """
        self.render_interval_ms = interval_ms

        # Update timer if active
        if self.is_active:
            self.render_timer.stop()
            self.render_timer.start(self.render_interval_ms)

        log.log_info(f"DebouncedRenderer interval updated to {interval_ms}ms")
