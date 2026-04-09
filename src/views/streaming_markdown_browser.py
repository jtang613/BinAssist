#!/usr/bin/env python3

from PySide6.QtWidgets import QTextBrowser, QAbstractScrollArea, QApplication
from PySide6.QtGui import QTextCursor, QKeySequence

from ..services.streaming.render_update import RenderUpdate, UpdateType
from ..services.streaming.streaming_renderer import MARKDOWN_CSS

_BOTTOM_THRESHOLD = 50
_DEFAULT_STREAMING_STYLESHEET = (
    MARKDOWN_CSS.replace("<style>", "").replace("</style>", "")
    + """
    body { font-family: sans-serif; background: #31363b; color: #d8dee9; margin: 0; padding: 8px; }
    a { color: #64b5f6; text-decoration: none; }
    .ga-chat-title { font-size: 13px; font-weight: 600; margin: 0 0 6px 0; color: #d7dde4; }
    .ga-empty { color: #9aa5b1; font-size: 12px; padding: 16px; border: 1px dashed #546e7a; }
    .ga-card { width: 100%; border: 1px solid #5a636c; background: #3a4046; margin: 0 0 8px 0; border-collapse: collapse; table-layout: fixed; }
    .ga-accent { width: 4px; min-width: 4px; padding: 0; }
    .ga-body { padding: 5px 8px 6px 8px; vertical-align: top; }
    .ga-heading { font-size: 11px; font-weight: 600; color: #eef2f6; margin-bottom: 1px; }
    .ga-subheading { font-weight: 400; color: #d7dde4; }
    .ga-meta { font-size: 10px; color: #9fb0bf; margin-bottom: 4px; }
    .ga-compact, .ga-tool-preview { font-size: 11px; color: #d8dee9; white-space: pre-wrap; word-wrap: break-word; line-height: 1.25; }
    .ga-markdown { font-size: 12px; line-height: 1.25; white-space: normal; overflow-wrap: anywhere; }
    .ga-markdown p { font-size: inherit; line-height: inherit; margin: 0 0 8px 0; }
    .ga-markdown ul, .ga-markdown ol { margin: 6px 0 6px 18px; padding: 0; }
    .ga-markdown li { margin: 2px 0; }
    .ga-markdown h1 { font-size: 18px; margin: 12px 0 8px 0; }
    .ga-markdown h2 { font-size: 16px; margin: 10px 0 6px 0; }
    .ga-markdown h3 { font-size: 14px; margin: 8px 0 4px 0; }
    .ga-markdown h4, .ga-markdown h5, .ga-markdown h6 { font-size: 12px; margin: 6px 0 4px 0; }
    .ga-markdown pre { white-space: pre-wrap; overflow-wrap: anywhere; background: #2b3035; padding: 6px; margin: 5px 0; border: 1px solid #4c565f; font-size: 11px; }
    .ga-markdown code { font-size: 11px; }
    .ga-code { white-space: pre-wrap; overflow-wrap: anywhere; font-size: 11px; background: #2b3035; padding: 6px; margin: 4px 0; border: 1px solid #4c565f; }
    """
)


class ScrollPositionTracker:
    """Track scroll position to preserve user's reading position"""

    def __init__(self, scroll_area: QAbstractScrollArea):
        self._scrollbar = scroll_area.verticalScrollBar()

    def is_at_bottom(self) -> bool:
        return (self._scrollbar.maximum() - self._scrollbar.value() <= _BOTTOM_THRESHOLD)

    def scroll_to_bottom(self) -> None:
        self._scrollbar.setValue(self._scrollbar.maximum())

    def get_scroll_value(self) -> int:
        return self._scrollbar.value()

    def set_scroll_value(self, value: int) -> None:
        self._scrollbar.setValue(value)


class StreamingMarkdownBrowser(QTextBrowser):
    """
    QTextBrowser with hybrid cursor-based rendering for reduced flicker.

    Features:
    - Copies markdown source on Ctrl+C (not rendered HTML)
    - Uses QTextCursor.insertHtml() when content is appended
    - Falls back to setHtml() when content changes earlier
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._markdown_source = ""
        self._scroll_tracker = ScrollPositionTracker(self)

        # Track previously rendered HTML for incremental updates
        self._last_rendered_html = ""
        self._committed_html = MARKDOWN_CSS  # Track committed HTML separately for RenderUpdate
        self._pending_start = 0  # Track character position where pending content starts

        # QTextCursor.insertHtml() does not reliably inherit styles declared in
        # the previous setHtml() document. Install a default stylesheet so
        # streaming fragments render like the final transcript cards.
        self.document().setDefaultStyleSheet(_DEFAULT_STREAMING_STYLESHEET)

        # Initialize with CSS so content is styled from the start
        self.setHtml(MARKDOWN_CSS)

    # --- Markdown Copy Functionality (from MarkdownCopyBrowser) ---

    def set_markdown_source(self, markdown_text: str):
        """Store the markdown source for copy operations"""
        self._markdown_source = markdown_text

    def get_markdown_source(self) -> str:
        """Get the markdown source"""
        return self._markdown_source

    def _copy_markdown(self):
        """Copy the markdown source to clipboard"""
        if self._markdown_source:
            QApplication.clipboard().setText(self._markdown_source)

    def keyPressEvent(self, event):
        """Intercept Ctrl+C to copy markdown"""
        if event.matches(QKeySequence.Copy):
            self._copy_markdown()
        else:
            super().keyPressEvent(event)

    def contextMenuEvent(self, event):
        """Override context menu to replace Copy action with markdown copy"""
        menu = self.createStandardContextMenu()
        for action in menu.actions():
            if action.text().replace('&', '') == 'Copy':
                action.triggered.disconnect()
                action.triggered.connect(self._copy_markdown)
                break
        menu.exec(event.globalPos())
        menu.deleteLater()

    # --- Streaming Functionality ---

    def set_streaming_content(self, markdown_text: str, html_content: str) -> None:
        """
        Update content with hybrid rendering for reduced flicker.

        Args:
            markdown_text: The markdown source (for copy operations)
            html_content: The pre-rendered HTML content
        """
        self.set_markdown_source(markdown_text)

        was_at_bottom = self._scroll_tracker.is_at_bottom()
        saved_scroll = self._scroll_tracker.get_scroll_value()

        # Check if we can do incremental append
        if (self._last_rendered_html and
            html_content.startswith(self._last_rendered_html)):
            # Incremental update - just append the delta
            delta_html = html_content[len(self._last_rendered_html):]
            if delta_html:
                cursor = self.textCursor()
                cursor.movePosition(QTextCursor.MoveOperation.End)
                cursor.insertHtml(delta_html)
                self._force_layout()
        else:
            # Full replace needed
            self.setHtml(html_content)
            self._force_layout()

        self._last_rendered_html = html_content

        # Restore scroll position
        if was_at_bottom:
            self._scroll_tracker.scroll_to_bottom()
        else:
            self._scroll_tracker.set_scroll_value(saved_scroll)

    def apply_render_update(self, update: RenderUpdate) -> None:
        """Apply RenderUpdate using cursor-based operations for minimal reflow."""
        was_at_bottom = self._scroll_tracker.is_at_bottom()
        saved_scroll = self._scroll_tracker.get_scroll_value()

        if update.update_type == UpdateType.FULL_REPLACE:
            self.setHtml(update.full_html or "")
            self._committed_html = update.full_html or ""
            self._pending_start = self.document().characterCount()
        else:
            # Incremental update
            if update.committed_html:
                # New committed content - rebuild committed portion with setHtml
                self._committed_html += update.committed_html
                self.setHtml(self._committed_html)
                self._force_layout()
                self._pending_start = self.document().characterCount()

                # Append pending via cursor (avoids second setHtml)
                if update.pending_html:
                    cursor = self.textCursor()
                    cursor.movePosition(QTextCursor.MoveOperation.End)
                    cursor.insertHtml(update.pending_html)
                    self._force_layout()
            else:
                # Only pending changed - pure cursor update (no flicker)
                cursor = self.textCursor()

                # Delete existing pending content
                if self._pending_start < self.document().characterCount():
                    cursor.setPosition(self._pending_start)
                    cursor.movePosition(QTextCursor.MoveOperation.End,
                                       QTextCursor.MoveMode.KeepAnchor)
                    cursor.removeSelectedText()

                # Insert new pending content
                if update.pending_html:
                    cursor.movePosition(QTextCursor.MoveOperation.End)
                    cursor.insertHtml(update.pending_html)

                self._force_layout()

        if was_at_bottom:
            self._scroll_tracker.scroll_to_bottom()
        else:
            self._scroll_tracker.set_scroll_value(saved_scroll)

    def reset_streaming(self) -> None:
        """Reset streaming state for new content"""
        self._last_rendered_html = ""
        self._committed_html = MARKDOWN_CSS
        self._pending_start = 0
        # Initialize document with CSS so streaming content is styled
        self.setHtml(MARKDOWN_CSS)

    def _force_layout(self) -> None:
        """Force Qt to complete document layout for accurate scrollbar"""
        doc = self.document()
        last_block = doc.lastBlock()
        if last_block.isValid():
            doc.documentLayout().blockBoundingRect(last_block)
