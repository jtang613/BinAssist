# Workflow: Interactive Queries and the ReAct Agent

This guide walks you through using the Query tab for interactive conversations with the LLM, including how to leverage context macros, MCP tools, and the autonomous ReAct agent.

## Overview

The Query tab provides a chat interface for asking questions about your binary. It supports:

- Free-form questions and follow-ups
- Context macros that reference binary elements
- MCP tool calling for interactive analysis
- ReAct autonomous agent for complex investigations

## Basic Query Workflow

### Asking Simple Questions

1. Open the Query tab
2. Type your question in the input field
3. Press Enter or click Send
4. Watch the response stream in

Questions can be about:
- General reverse engineering concepts
- Specific aspects of your binary
- Analysis techniques and approaches
- Anything you'd ask an expert analyst

### Using Context Macros

Context macros let you reference specific binary elements. The macro is replaced with actual data before sending to the LLM.

| Macro | Description |
|-------|-------------|
| `#func` | Current function's decompiled code |
| `#addr` | Data at the current address |
| `#line` | Current instruction |
| `#range(start, end)` | Data in an address range |

**Example**:
```
What security vulnerabilities exist in #func?
```

The LLM receives the full decompiled code along with your question.

### Managing Conversations

The Query tab maintains conversation history:

- **Left panel**: List of saved conversations
- **New button**: Start a fresh conversation
- **Delete button**: Remove selected conversation

Conversations persist across sessions, so you can return to previous analyses.

## MCP Tool Integration

When MCP is enabled, the LLM can call tools to gather information during the conversation.

### Enabling MCP

1. Check the **MCP** checkbox in the Query tab
2. Ensure MCP servers are configured in [Settings](../tabs/settings-tab.md)

### How Tool Calling Works

1. You ask a question
2. The LLM decides it needs more information
3. It calls relevant tools (decompile, get xrefs, etc.)
4. Tool results appear in the conversation
5. The LLM uses the results to answer your question

### Common Tools

Tools available depend on your MCP server configuration. Common tools include:

- **Decompile**: Get decompiled code for a function
- **Disassemble**: Get assembly for an address range
- **Get XRefs**: Find cross-references to/from an address
- **Navigate**: Jump to a location in Binary Ninja
- **Query Graph**: Search the semantic graph

## ReAct Agent Workflow

For complex investigations, the ReAct agent conducts autonomous multi-step analysis.

### When to Use ReAct

ReAct is ideal for:

- **Exploratory questions**: "What does this binary do?"
- **Security auditing**: "Find vulnerabilities in this code"
- **Data flow analysis**: "Trace user input through the program"
- **Comprehensive analysis**: "Analyze the authentication mechanism"

For simple questions, basic queries are faster.

### Enabling ReAct

1. Check the **Agentic** checkbox
2. Also enable **MCP** (ReAct needs tools)
3. Ask your question

### The ReAct Process

The agent follows four phases:

#### 1. Planning

The agent creates an investigation plan based on your question:

```
Based on your question, I'll investigate:
- [ ] Identify the entry point and main function
- [ ] Trace the authentication flow
- [ ] Examine credential validation
- [ ] Check for bypass vulnerabilities
```

#### 2. Investigation

For each planned step, the agent:
- Calls relevant tools
- Gathers information
- Summarizes findings
- Marks the step complete

```
- [x] Identify the entry point and main function
- [->] Trace the authentication flow
- [ ] Examine credential validation
- [ ] Check for bypass vulnerabilities
```

The `[->]` marker shows the current focus.

#### 3. Reflection

After each step, the agent reflects:
- Have I gathered enough information?
- Should I add new investigation steps?
- Should I remove obsolete steps?
- Am I ready to answer?

The plan may adapt based on discoveries.

#### 4. Synthesis

When ready, the agent synthesizes a comprehensive answer:
- Summarizes all findings
- References specific evidence
- Provides actionable conclusions
- Suggests next steps

### Monitoring Progress

During a ReAct investigation:
- Watch the investigation plan update
- See tool calls and results
- The current step is highlighted
- You can stop at any time

### Stopping Early

Click **Stop** to cancel a ReAct investigation. The agent will provide a partial summary based on findings so far.

## Extended Thinking

For models that support it, extended thinking improves reasoning quality.

### Configuring Reasoning Effort

In [Settings](../tabs/settings-tab.md), set the **Reasoning Effort**:

| Level | Best For |
|-------|----------|
| None | Quick queries |
| Low | Light reasoning |
| Medium | Moderate complexity |
| High | Deep analysis |

Higher levels increase latency and cost but improve quality for complex questions.

### Supported Models

- Anthropic: Claude Sonnet 4+, Claude Opus
- OpenAI: o1, o3, gpt-5 models
- Ollama: gpt-oss and compatible models

## Combining Approaches

For thorough analysis, combine different approaches:

1. **Start simple**: Ask basic questions to orient yourself
2. **Add context**: Use `#func` to focus on specific code
3. **Enable MCP**: Let the LLM explore with tools
4. **Use ReAct**: For complex investigations
5. **Increase reasoning**: For difficult problems

## Tips for Effective Queries

### Be Specific

Instead of: "What does this do?"
Try: "What does #func do with the network socket after receiving data?"

### Provide Context

Instead of: "Is this vulnerable?"
Try: "Does #func have any buffer overflow vulnerabilities when processing user input from the network?"

### Iterate

Don't expect one query to answer everything:
1. Start broad to understand the landscape
2. Narrow down to specific areas
3. Deep dive with ReAct for thorough analysis

### Check the Semantic Graph

If you've built a semantic graph, enable MCP so the LLM can query it for function summaries and relationships.

## Related Documentation

- [Query Tab Reference](../tabs/query-tab.md) - UI element details
- [Explain Workflow](explain-workflow.md) - Structured function analysis
- [Semantic Graph Workflow](semantic-graph-workflow.md) - Building context for queries
- [Settings Tab](../tabs/settings-tab.md) - Configuring MCP and reasoning effort
