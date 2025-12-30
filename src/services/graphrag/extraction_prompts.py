#!/usr/bin/env python3

from dataclasses import dataclass
import re
from typing import List


@dataclass
class ComplexityMetrics:
    line_count: int
    branch_count: int
    call_count: int
    loop_count: int
    level: str
    summary_guidance: str

    def __str__(self) -> str:
        return (f"{self.line_count} lines, {self.branch_count} branches, "
                f"{self.call_count} calls, {self.loop_count} loops ({self.level})")


def analyze_complexity(code: str) -> ComplexityMetrics:
    if not code:
        return ComplexityMetrics(0, 0, 0, 0, "simple",
                                 "This is a simple function. Provide a concise summary (2-4 sentences).")

    lines = len(code.splitlines())
    branches = _count_occurrences(code, r"\bif\s*\(")
    branches += _count_occurrences(code, r"\bswitch\s*\(")
    branches += _count_occurrences(code, r"\bcase\s+")
    branches += _count_occurrences(code, r"\?.*:")  # ternary

    loops = _count_occurrences(code, r"\bwhile\s*\(")
    loops += _count_occurrences(code, r"\bfor\s*\(")
    loops += _count_occurrences(code, r"\bdo\s*\{")

    calls = _count_occurrences(code, r"\b[a-zA-Z_][a-zA-Z0-9_]*\s*\(")
    calls = max(0, calls - branches - loops)

    score = 0
    score += 2 if lines > 50 else (1 if lines > 20 else 0)
    score += 2 if branches > 10 else (1 if branches > 5 else 0)
    score += 2 if calls > 8 else (1 if calls > 4 else 0)
    score += 2 if loops > 3 else (1 if loops > 1 else 0)

    if score >= 6:
        level = "very_complex"
        guidance = ("This is a very complex function. Provide a detailed multi-paragraph summary "
                    "(3-5 paragraphs) covering all major code paths, data transformations, and behaviors.")
    elif score >= 4:
        level = "complex"
        guidance = ("This is a complex function. Provide a thorough summary (2-3 paragraphs) explaining "
                    "the main logic, key operations, and notable patterns.")
    elif score >= 2:
        level = "moderate"
        guidance = ("This is a moderately complex function. Provide a detailed summary (1-2 paragraphs) "
                    "explaining its purpose and key operations.")
    else:
        level = "simple"
        guidance = "This is a simple function. Provide a concise summary (2-4 sentences)."

    return ComplexityMetrics(lines, branches, calls, loops, level, guidance)


def function_summary_prompt(function_name: str, decompiled_code: str,
                            callers: List[str], callees: List[str]) -> str:
    complexity = analyze_complexity(decompiled_code)

    prompt = ["Analyze this decompiled function and provide a summary.\n"]
    prompt.append(f"## Function: {function_name}\n")
    prompt.append(f"**Complexity:** {complexity}\n")

    if callers:
        prompt.append(f"**Called by:** {', '.join(callers)}\n")
    if callees:
        prompt.append(f"**Calls:** {', '.join(callees)}\n")

    truncate_limit = 4000 if complexity.level == "very_complex" else 3000 if complexity.level == "complex" else 2000
    prompt.append(f"\n```c\n{_truncate_code(decompiled_code, truncate_limit)}\n```\n")
    prompt.append(f"**Summary Length Guidance:** {complexity.summary_guidance}\n\n")
    prompt.append("Provide a summary in the following format:\n\n")

    if complexity.level in ("very_complex", "complex"):
        prompt.append("**Purpose:** [A thorough description of what this function does, "
                      "its role in the larger system, and its key responsibilities.]\n\n")
        prompt.append("**Behavior:** [Detailed explanation of the function's logic including:\n"
                      "- Main code paths and control flow\n"
                      "- Key data transformations and algorithms\n"
                      "- Important state changes and side effects\n"
                      "- Error handling patterns]\n\n")
    else:
        prompt.append("**Purpose:** [1-3 sentences describing what this function does]\n\n")
        prompt.append("**Behavior:** [Key operations, data transformations, control flow]\n\n")

    prompt.append("**Security Notes:** [Any potential security concerns: buffer handling, "
                  "input validation, crypto usage, privilege operations. Write 'None identified' if none.]\n\n")
    prompt.append("**Category:** [One of: initialization, data_processing, io_operations, "
                  "network, crypto, authentication, error_handling, utility, unknown]\n")
    return "".join(prompt)


def batch_function_summary_prompt(nodes: List[dict]) -> str:
    prompt = ["Summarize each of these functions. Scale summary length based on complexity:\n"
              "- Simple functions: 1-2 sentences\n"
              "- Moderate functions: 3-5 sentences\n"
              "- Complex functions: 1-2 paragraphs\n"
              "- Very complex functions: 2-3 paragraphs\n\n"
              "Format your response as a numbered list matching the input.\n\n"]

    for idx, node in enumerate(nodes, 1):
        name = node.get("name") or "unknown"
        code = node.get("raw_code") or ""
        complexity = analyze_complexity(code)
        truncate_limit = 2000 if complexity.level == "very_complex" else 1500 if complexity.level == "complex" else 1000
        prompt.append(f"{idx}. **{name}** [{complexity.level}]\n```c\n"
                      f"{_truncate_code(code, truncate_limit)}\n```\n\n")

    prompt.append("Summaries:\n")
    return "".join(prompt)


def _count_occurrences(text: str, pattern: str) -> int:
    return len(re.findall(pattern, text))


def _truncate_code(code: str, max_length: int) -> str:
    if not code:
        return ""
    if len(code) <= max_length:
        return code
    cutoff = code.rfind("\n", 0, max_length)
    if cutoff < max_length // 2:
        cutoff = max_length
    return code[:cutoff] + "\n// ... (truncated)"
