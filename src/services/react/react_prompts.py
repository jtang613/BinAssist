#!/usr/bin/env python3

"""
ReAct Agent Prompts

Prompt templates for planning, investigation, reflection, and synthesis.
Based on GhidrAssist's proven ReAct implementation.
"""


class ReActPrompts:
    """Prompt templates for ReAct agent"""

    @staticmethod
    def get_system_prompt() -> str:
        """System prompt for ReAct agent"""
        return """You are an expert reverse engineering assistant conducting an autonomous investigation.

You have access to MCP tools for binary analysis. Use these tools to systematically investigate the user's question.

Guidelines:
- Be thorough but efficient - don't make redundant tool calls
- Focus on the most relevant information for the investigation
- Track your progress against the investigation plan
- Summarize findings clearly after each tool use
- Know when you have enough information to provide a comprehensive answer"""

    @staticmethod
    def get_planning_prompt(objective: str, context_preview: str = "") -> str:
        """Generate planning prompt for initial todo list creation"""
        context_section = f"""**Available Context**:
{context_preview}""" if context_preview else """**Available Context**:
[Binary analysis context available via tools]"""

        return f"""## Investigation Planning

**User's Question**: {objective}

{context_section}

Before we start investigating, let's plan the investigation steps.

**Task**: Based on the user's question, propose a list of 3-5 investigation
steps to answer this question thoroughly.

Format your response as a markdown checklist:
- [ ] First investigation step
- [ ] Second investigation step
- [ ] Third investigation step

Focus on specific, actionable steps that use the available MCP tools.
Consider what information you need to gather to provide a comprehensive answer.

Keep steps concrete and achievable in 1-2 tool calls each."""

    @staticmethod
    def get_investigation_prompt(objective: str, todos_formatted: str,
                                  findings_formatted: str, iteration: int,
                                  remaining_iterations: int) -> str:
        """Generate investigation prompt for each iteration"""
        warning = ""
        if remaining_iterations <= 3:
            warning = f"\n\n**Warning:** Only {remaining_iterations} iterations remaining. Consider synthesizing your answer soon if you have enough information."

        return f"""## Investigation Iteration {iteration}

**Your Goal**: {objective}

**Investigation Progress**:
{todos_formatted}

**What You've Discovered**:
{findings_formatted}
{warning}

**Current Task**: Focus on the task marked with [->] in the progress list above.

**Instructions**:
1. Think about what information you still need for the current task
2. Call the appropriate MCP tool(s) to gather that information
3. After receiving results, briefly summarize what you learned

If you believe the current task is complete based on previous findings, you may proceed without additional tool calls."""

    @staticmethod
    def get_reflection_prompt(objective: str, todos_formatted: str,
                              findings_formatted: str) -> str:
        """Generate self-reflection prompt to decide continue or synthesize"""
        return f"""## Self-Reflection

**Original Question**: {objective}

**Investigation Progress**:
{todos_formatted}

**Findings So Far**:
{findings_formatted}

**Reflection Task**: Based on what you've discovered so far:

1. Do you have sufficient information to answer the user's question comprehensively?
2. Are there critical gaps that require more investigation?

Respond with ONLY one of these formats:
- "READY: [brief reason why you can answer now]"
- "CONTINUE: [what critical information is still needed]"

Guidelines:
- Say READY if you have enough evidence to give a useful, well-supported answer
- Say CONTINUE only if there's critical missing information that would significantly improve the answer
- Don't continue just because there are unchecked todos - focus on whether you can answer the question
- Be honest about what you know vs. what would be nice to know"""

    @staticmethod
    def get_synthesis_prompt(objective: str, todos_formatted: str,
                             findings_detailed: str) -> str:
        """Generate final synthesis prompt"""
        return f"""## Time to Synthesize

**Goal**: {objective}

**Completed Investigation**:
{todos_formatted}

**Investigation Summary**:
{findings_detailed}

You've completed your investigation. Based on all the information you've gathered, provide a comprehensive answer to the user's question.

**Instructions**:
Synthesize your findings into a clear, actionable response that:
1. Directly addresses the user's question
2. References specific evidence from your investigation
3. Highlights the most important discoveries
4. Provides recommendations or next steps if applicable

Format your response with clear headings and structure for readability.
Be concise but thorough - include all relevant findings."""

    @staticmethod
    def get_tool_guidance_prompt(available_tools: list) -> str:
        """Generate guidance about available tools"""
        if not available_tools:
            return ""

        tool_names = [t.get('function', {}).get('name', 'unknown') for t in available_tools]
        return f"""
**Available Tools**: {', '.join(tool_names)}

Choose the most appropriate tool for your current investigation step."""
