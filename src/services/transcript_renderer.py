#!/usr/bin/env python3

import html
import json
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Set

from .models.transcript_models import ContextSnapshot, TranscriptEvent, TranscriptEventKind
from .streaming.streaming_renderer import render_markdown_to_html


class TranscriptRenderer:
    def __init__(self, product_label: str):
        self.product_label = product_label

    def render(self, chat_title: str, events: Iterable[TranscriptEvent],
               streaming_assistant_markdown: Optional[str] = None,
               expanded_keys: Optional[Set[str]] = None,
               context_snapshot: Optional[ContextSnapshot] = None) -> str:
        expanded_keys = expanded_keys or set()
        event_list = list(events)
        grouped = self._group_events(event_list)
        cards = [self._render_title(chat_title)]
        for item in grouped:
            cards.append(self._render_item(item, expanded_keys))
        if streaming_assistant_markdown:
            cards.append(self._render_message_card("Assistant", streaming_assistant_markdown, "#4db6ac"))
        if not grouped and not streaming_assistant_markdown:
            cards.append("<div class='ga-empty'>No messages yet.</div>")
        return f"""
        <html><head><style>{self._css()}</style></head><body>
        {''.join(cards)}
        </body></html>
        """

    def render_streaming_assistant_card(self, markdown: str) -> str:
        return self._render_message_card("Assistant", markdown, "#4db6ac")

    def _group_events(self, events: List[TranscriptEvent]) -> List[Dict]:
        groups: List[Dict] = []
        tool_groups: "OrderedDict[str, Dict]" = OrderedDict()
        for event in events:
            if event.event_kind in (
                TranscriptEventKind.TOOL_CALL_REQUESTED,
                TranscriptEventKind.TOOL_CALL_STARTED,
                TranscriptEventKind.TOOL_CALL_COMPLETED,
                TranscriptEventKind.TOOL_CALL_FAILED,
                TranscriptEventKind.APPROVAL_REQUESTED,
                TranscriptEventKind.APPROVAL_DECISION,
            ) and event.correlation_id:
                group = tool_groups.setdefault(event.correlation_id, {
                    "type": "tool",
                    "correlation_id": event.correlation_id,
                    "events": [],
                })
                group["events"].append(event)
                if group not in groups:
                    groups.append(group)
                continue
            groups.append({"type": "event", "event": event})
        return groups

    def _render_item(self, item: Dict, expanded_keys: Set[str]) -> str:
        if item["type"] == "tool":
            return self._render_tool_group(item, expanded_keys)
        event = item["event"]
        if event.event_kind == TranscriptEventKind.USER_MESSAGE:
            return self._render_message_card("User", event.content_text or "", "#64b5f6", event.created_at)
        if event.event_kind == TranscriptEventKind.ASSISTANT_MESSAGE:
            return self._render_message_card("Assistant", event.content_text or "", "#4db6ac", event.created_at)
        if event.event_kind == TranscriptEventKind.TODO_UPDATED:
            return self._render_task_card(event, expanded_keys)
        if event.event_kind == TranscriptEventKind.FINDING_ADDED:
            return self._render_note_card("Finding", event.content_text or "", "#ffb74d", event.created_at)
        if event.event_kind == TranscriptEventKind.ITERATION_NOTICE:
            return self._render_note_card("Agent", event.content_text or "", "#9575cd", event.created_at)
        if event.event_kind == TranscriptEventKind.CONTEXT_COMPACTED:
            return self._render_note_card("Context", event.content_text or "", "#ef5350", event.created_at)
        if event.event_kind == TranscriptEventKind.DOCUMENT_SNAPSHOT:
            return self._render_message_card("Assistant", event.content_text or "", "#4db6ac", event.created_at)
        return self._render_note_card("System", event.content_text or "", "#90a4ae", event.created_at)

    def _render_tool_group(self, item: Dict, expanded_keys: Set[str]) -> str:
        events = item["events"]
        first = events[0]
        metadata = {}
        result_text = ""
        status = "Pending"
        approval_line = ""
        for event in events:
            metadata.update(event.metadata or {})
            if event.event_kind == TranscriptEventKind.APPROVAL_REQUESTED:
                approval_line = f"Approval required: {html.escape(metadata.get('risk_tier', 'unknown'))}"
            elif event.event_kind == TranscriptEventKind.APPROVAL_DECISION:
                approval_line = f"Approval: {html.escape(event.content_text or '')}"
            elif event.event_kind == TranscriptEventKind.TOOL_CALL_STARTED:
                status = "Executing"
            elif event.event_kind == TranscriptEventKind.TOOL_CALL_COMPLETED:
                status = "Completed"
                result_text = event.content_text or ""
            elif event.event_kind == TranscriptEventKind.TOOL_CALL_FAILED:
                status = "Failed"
                result_text = event.content_text or ""
        tool_name = metadata.get("tool_name", "tool")
        tool_source = metadata.get("tool_source", "")
        arguments = metadata.get("arguments") or {}
        expanded = item["correlation_id"] in expanded_keys
        toggle = "Hide details" if expanded else "Show details"
        action = "tool-collapse" if expanded else "tool-expand"
        details_html = ""
        if expanded:
            args_html = self._code_block(json.dumps(arguments, indent=2, sort_keys=True))
            result_source = result_text
            if events[-1].artifact_ref:
                try:
                    with open(events[-1].artifact_ref.file_path, "r", encoding="utf-8") as handle:
                        result_source = handle.read()
                except OSError:
                    result_source = result_text
            details_html = f"""
            <div class='ga-tool-section'><div class='ga-label'>Arguments</div>{args_html}</div>
            <div class='ga-tool-section'><div class='ga-label'>Result</div>{self._code_block(result_source or '')}</div>
            """
        else:
            preview = html.escape((result_text or "").strip().replace("\n", " ")[:220])
            details_html = f"<div class='ga-tool-preview'>{preview}</div>" if preview else ""
        meta_line = f"{html.escape(first.created_at or '')} | {html.escape(tool_source)}"
        return f"""
        <table class='ga-card ga-tool' cellspacing='0' cellpadding='0'>
            <tr>
            <td class='ga-accent' style='background:#d8a24b;'>&nbsp;</td>
            <td class='ga-body'>
                <div class='ga-heading'>Tool <span class='ga-subheading'>{html.escape(tool_name)}</span></div>
                <div class='ga-meta'>{meta_line}</div>
                <div class='ga-compact'>Status: {html.escape(status)} {html.escape(approval_line)}</div>
                {details_html}
                <div class='ga-toggle'><a href='{action}:{item["correlation_id"]}'>
                    {toggle}
                </a></div>
            </td>
            </tr>
        </table>
        """

    def _render_task_card(self, event: TranscriptEvent, expanded_keys: Set[str]) -> str:
        key = f"tasks:{event.id}"
        expanded = key in expanded_keys
        metadata = event.metadata or {}
        tasks = metadata.get("todos") or metadata.get("tasks") or []
        summary_text = metadata.get("summary") or event.content_text or "Task list updated"
        progress_text = self._format_task_counts(metadata, tasks)
        active_task = self._find_active_task(tasks)
        toggle = "Hide tasks" if expanded else "Show tasks"
        action = "tasks-collapse" if expanded else "tasks-expand"
        details = ""
        if expanded:
            lines = "".join(
                f"<li>{self._format_task_detail(task)}</li>"
                for task in tasks
            )
            details = f"<ul class='ga-task-list'>{lines}</ul>"
        summary_block = f"<div><b>Summary:</b> {html.escape(summary_text)}</div>"
        if progress_text:
            summary_block += f"<div><b>Progress:</b> {html.escape(progress_text)}</div>"
        if active_task:
            summary_block += f"<div><b>Active:</b> {html.escape(active_task)}</div>"
        return f"""
        <table class='ga-card ga-agent' cellspacing='0' cellpadding='0'>
            <tr>
            <td class='ga-accent' style='background:#8d79c9;'>&nbsp;</td>
            <td class='ga-body'>
                <div class='ga-heading'>Tasks</div>
                <div class='ga-meta'>{html.escape(event.created_at or '')}</div>
                <div class='ga-compact'>{summary_block}</div>
                {details}
                <div class='ga-toggle'><a href='{action}:{event.id}'>{toggle}</a></div>
            </td>
            </tr>
        </table>
        """

    def _render_note_card(self, heading: str, content: str, accent: str, created_at: Optional[str]) -> str:
        return f"""
        <table class='ga-card ga-note' cellspacing='0' cellpadding='0'>
            <tr>
            <td class='ga-accent' style='background:{accent};'>&nbsp;</td>
            <td class='ga-body'>
                <div class='ga-heading'>{html.escape(heading)}</div>
                <div class='ga-meta'>{html.escape(created_at or '')}</div>
                <div class='ga-compact'>{html.escape(content)}</div>
            </td>
            </tr>
        </table>
        """

    def _format_task_counts(self, metadata: Dict, tasks: List[Dict]) -> str:
        total = self._coerce_int(metadata.get("total_count"))
        complete = self._coerce_int(metadata.get("complete_count"))
        active = self._coerce_int(metadata.get("in_progress_count"))
        pending = self._coerce_int(metadata.get("pending_count"))
        if total <= 0 and tasks:
            normalized = [self._normalize_task_status(task.get("status")) for task in tasks]
            total = len(tasks)
            complete = sum(1 for status in normalized if status == "COMPLETE")
            active = sum(1 for status in normalized if status == "IN_PROGRESS")
            pending = max(0, total - complete - active)
        if total <= 0:
            return ""
        return f"{complete}/{total} complete | {active} active | {pending} pending"

    def _find_active_task(self, tasks: List[Dict]) -> Optional[str]:
        for task in tasks:
            if self._normalize_task_status(task.get("status")) == "IN_PROGRESS":
                return task.get("task") or None
        return None

    def _format_task_detail(self, task: Dict) -> str:
        status = self._normalize_task_status(task.get("status"))
        icon = "[ ]"
        if status == "COMPLETE":
            icon = "[x]"
        elif status == "IN_PROGRESS":
            icon = "[->]"
        line = f"{icon} {html.escape(task.get('task', '') or 'Unnamed task')}"
        evidence = task.get("evidence")
        if evidence:
            line += f"<br>&nbsp;&nbsp;&nbsp;&nbsp;evidence: {html.escape(evidence)}"
        return line

    def _normalize_task_status(self, status: Optional[str]) -> str:
        value = (status or "PENDING").strip().upper()
        if value in ("COMPLETE", "COMPLETED"):
            return "COMPLETE"
        if value in ("IN_PROGRESS", "IN PROGRESS", "ACTIVE", ">"):
            return "IN_PROGRESS"
        return "PENDING"

    def _coerce_int(self, value) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _render_message_card(self, heading: str, markdown: str, accent: str, created_at: Optional[str] = None) -> str:
        html_body = render_markdown_to_html(markdown or "", include_css=False)
        return f"""
        <table class='ga-card ga-message' cellspacing='0' cellpadding='0'>
            <tr>
            <td class='ga-accent' style='background:{accent};'>&nbsp;</td>
            <td class='ga-body'>
                <div class='ga-heading'>{html.escape(heading)}</div>
                <div class='ga-meta'>{html.escape(created_at or '')}</div>
                <div class='ga-markdown'>{html_body}</div>
            </td>
            </tr>
        </table>
        """

    def _render_title(self, title: str) -> str:
        return f"<div class='ga-chat-title'>{html.escape(title or 'Chat')}</div>"

    def _render_status(self, snapshot: Optional[ContextSnapshot]) -> str:
        if not snapshot:
            return ""
        token_text = "No active context window data"
        if snapshot.token_limit:
            token_text = (
                f"context {snapshot.token_count}/{snapshot.token_limit} tokens | "
                f"{snapshot.message_count} msgs | peak {snapshot.peak_token_count} tokens / {snapshot.peak_message_count} msgs | "
                f"threshold {snapshot.threshold}"
            )
            if snapshot.compaction_summary:
                token_text += f" | compacted {html.escape(snapshot.compaction_summary)}"
        return f"<div class='ga-status'>Model: {html.escape(snapshot.provider_label)} / {html.escape(snapshot.model_name)} | {token_text}</div>"

    def _code_block(self, text: str) -> str:
        return f"<pre class='ga-code'>{html.escape(text or '')}</pre>"

    def _css(self) -> str:
        return """
        body { font-family: sans-serif; background: #31363b; color: #d8dee9; margin: 0; padding: 8px; }
        a { color: #64b5f6; text-decoration: none; }
        .ga-status { font-size: 11px; color: #b0bec5; margin-bottom: 8px; }
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
        .ga-markdown pre { white-space: pre-wrap; overflow-wrap: anywhere; background: #2b3035; padding: 6px; margin: 5px 0; border: 1px solid #4c565f; }
        .ga-code { white-space: pre-wrap; overflow-wrap: anywhere; font-size: 11px; background: #2b3035; padding: 6px; margin: 4px 0; border: 1px solid #4c565f; }
        .ga-label { font-size: 10px; color: #b0bec5; margin: 4px 0 2px 0; }
        .ga-toggle { margin-top: 4px; font-size: 10px; }
        .ga-task-list { margin: 6px 0 0 16px; padding: 0; font-size: 11px; }
        .ga-tool-section { margin-top: 4px; }
        """
