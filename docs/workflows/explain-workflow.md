# Workflow: Building Context with the Explain Tab

This guide walks you through using the Explain tab to systematically document your understanding of a binary, building context that persists across analysis sessions.

## Overview

The Explain tab generates LLM-powered explanations of functions and instructions. Unlike one-off queries, these explanations are stored and associated with specific functions, allowing you to:

- Build up documentation as you analyze
- Return to previous explanations later
- Edit and refine the LLM's output
- Track security properties across functions

## When to Use the Explain Tab

Use the Explain tab when you want to:

- Understand what a function does at a high level
- Document your findings for later reference
- Get security assessment of a function
- Create a foundation for deeper analysis

For one-off questions or multi-step investigations, consider the [Query tab](../tabs/query-tab.md) instead.

## Step-by-Step Workflow

### Step 1: Select a Function

Navigate to a function in Binary Ninja:

1. Use the Functions list in the left panel
2. Or jump to an address with **Go to Address** (G key)
3. Or follow a cross-reference from another function

The Explain tab automatically detects your current function and displays its name and address.

### Step 2: Generate an Explanation

Click **Explain Function** to generate a comprehensive analysis.

The LLM analyzes the function at multiple IL levels:
- Low-Level IL (LLIL) - closest to assembly
- Medium-Level IL (MLIL) - simplified operations
- High-Level IL (HLIL) - closest to source code

Watch as the explanation streams in, covering:
- Function purpose and behavior
- Parameters and return values
- Notable operations and patterns
- Potential concerns or issues

### Step 3: Review Security Analysis

Expand the Security Analysis panel below the explanation to see:

| Field | What It Tells You |
|-------|-------------------|
| **Risk Level** | Overall security assessment (Low/Medium/High/Critical) |
| **Activity Profile** | What the function does (e.g., "Network Communication", "Data Processing") |
| **Security Flags** | Detected patterns (e.g., "uses dangerous functions", "handles user input") |
| **Network APIs** | Network-related calls found |
| **File I/O APIs** | File operations detected |

This information helps you quickly identify security-relevant functions and prioritize your analysis.

### Step 4: Edit and Refine (Optional)

If the explanation needs corrections or additions:

1. Click the **Edit** button
2. Modify the text as needed
3. Click **Save** to store your changes

Your edits are saved to the analysis database and will appear the next time you view this function.

### Step 5: Provide Feedback (Optional)

Use the thumbs up/thumbs down buttons to indicate whether the explanation was helpful. This feedback is stored and can be used for model improvement.

## Enhancing Explanations

### Using RAG

If you have relevant documentation indexed in the [RAG tab](../tabs/rag-tab.md):

1. Enable the **RAG** checkbox before clicking Explain Function
2. BinAssist searches your indexed documents for relevant context
3. Relevant excerpts are included in the LLM prompt
4. The explanation may reference your documentation

This is particularly useful when analyzing binaries that use specific libraries or protocols.

### Using MCP

If you have MCP servers configured:

1. Enable the **MCP** checkbox before clicking Explain Function
2. The LLM can call tools to gather additional information
3. This may include cross-references, related functions, or other context

## Explaining Individual Instructions

For detailed instruction-level analysis:

1. Position your cursor on a specific instruction in Binary Ninja
2. Click **Explain Line** in the Explain tab
3. Receive a detailed explanation of that instruction

This is useful for:
- Understanding complex or unusual instructions
- Analyzing obfuscated code
- Learning about unfamiliar operations

## Building a Documentation Set

To systematically document a binary:

1. **Start with entry points**: Explain `main`, exported functions, or known entry points
2. **Follow the call graph**: Navigate to called functions and explain them
3. **Prioritize by interest**: Focus on security-relevant or complex functions
4. **Edit as you learn**: Update explanations as your understanding deepens

Your explanations persist in the analysis database, keyed by binary hash and function address.

## Tips for Effective Context Building

### Quality Over Quantity

- Focus on understanding key functions thoroughly
- Don't rush through every function
- Take time to edit and refine explanations

### Combine with Other Tools

- Use **Query** for follow-up questions about an explained function
- Use **Actions** to rename functions based on your understanding
- Use **Semantic Graph** to see the bigger picture

### Use Security Analysis

- Pay attention to the Risk Level
- Functions with Network or File I/O APIs deserve extra scrutiny
- Security flags highlight patterns worth investigating

## Stored Explanations

Explanations are stored in the analysis database at:
```
~/.binaryninja/binassist/analysis.db
```

Each explanation is keyed by:
- Binary SHA256 hash (so different copies of the same binary share analysis)
- Function address

When you return to a previously explained function, the stored explanation is displayed automatically.

## Related Documentation

- [Explain Tab Reference](../tabs/explain-tab.md) - UI element details
- [Query Workflow](query-workflow.md) - For interactive follow-up questions
- [RAG Tab](../tabs/rag-tab.md) - Managing documents for context enhancement
