#!/usr/bin/env python3

"""
Document Chat Tool

Provides an LLM-callable tool that creates a new chat session populated with
custom markdown content. Used by the ReAct agent to output clean reports and
as groundwork for SymGraph document sync.
"""

DOCUMENT_CHAT_TOOL_NAME = "add_document_to_chat"

DOCUMENT_CHAT_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "add_document_to_chat",
        "description": "Create a new chat document with custom markdown content. Use this to produce standalone analysis reports, summaries, or findings separate from the current conversation.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Title for the new chat document"},
                "content": {"type": "string", "description": "Markdown content for the document"}
            },
            "required": ["title", "content"]
        }
    }
}

# Callback registry
_handler = None


def set_document_chat_handler(handler):
    """Register a callback for creating document chats.

    Args:
        handler: Callable(title: str, content: str) -> int (chat_id)
    """
    global _handler
    _handler = handler


def execute_document_chat_tool(arguments):
    """Execute the add_document_to_chat tool.

    Args:
        arguments: dict with 'title' and 'content' keys

    Returns:
        Result string for the LLM
    """
    if _handler is None:
        return "Error: document chat handler not registered"
    title = arguments.get("title", "Untitled Document")
    content = arguments.get("content", "")
    try:
        chat_id = _handler(title, content)
        return f"Document '{title}' created as Chat {chat_id}"
    except Exception as e:
        return f"Error creating document: {e}"
