from __future__ import annotations

import json
import os
import random
import re
import time
from typing import Any, Dict, List

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from pipeline.diff_utils import annotate_diff_with_line_numbers
from pipeline.merge_utils import (
    category_counts,
    overall_severity_of,
    process_findings,
    verdict_for,
)
from pipeline.state import ReviewState


def _get_llm() -> ChatGroq:
    """70b model, used by the security agent and the merge node."""
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=os.environ.get("GROQ_API_KEY"),
    )


def _get_fast_llm() -> ChatGroq:
    """8b model for the lower-stakes agents (more generous free-tier quota)."""
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=os.environ.get("GROQ_API_KEY"),
    )


def _invoke_with_retry(llm: ChatGroq, messages: list, max_attempts: int = 5) -> Any:
    """Call llm.invoke() with backoff on Groq rate limits.

    Five agents fire in parallel, so we can blow past the free-tier limit in a
    second. The jitter keeps all five from retrying at the same instant.
    """
    for attempt in range(max_attempts):
        try:
            return llm.invoke(messages)
        except Exception as exc:
            msg = str(exc).lower()
            is_rate_limit = any(tok in msg for tok in ("rate limit", "429", "too many", "rate_limit", "ratelimit"))
            if is_rate_limit and attempt < max_attempts - 1:
                base_wait = (2 ** attempt) * 8          # 8s, 16s, 32s, 64s
                jitter = random.uniform(0, base_wait * 0.4)
                wait = base_wait + jitter
                print(
                    f"[rate limit] attempt {attempt + 1}/{max_attempts}, "
                    f"retrying in {wait:.1f}s",
                    flush=True,
                )
                time.sleep(wait)
            else:
                raise


def _parse_json_array(text: str) -> List[Dict[str, Any]]:
    """Pull the first JSON array out of an LLM response, ignoring any prose around it."""
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # Greedy match so the whole array is captured even if prose surrounds it.
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    return []


_FINDING_SCHEMA = """
Return a JSON array of findings. Each finding MUST have exactly these fields:
[
  {
    "line": <integer: the L<n> number shown at the start of the offending line>,
    "line_content": "<the exact code on that line, WITHOUT the 'L<n>:' prefix>",
    "category": "security | performance | correctness | style | test_coverage",
    "severity": "critical | high | medium | low",
    "title": "<short label, max 10 words>",
    "description": "<why this is a real problem and what it leads to>",
    "suggestion": "<concrete fix, including a corrected code snippet>"
  }
]

Severity guide: SQL injection, hardcoded secrets, and plaintext password storage
are critical; missing auth/IDOR, null-deref crashes, and infinite loops with no
timeout are high (or critical); type-safety holes and resource leaks are medium;
pure readability is low.

Rules:
- The diff is shown with every new-file line prefixed by its number, e.g. "L12: ...".
  Copy that exact number into "line". Do not count lines yourself.
- Report a finding ONLY if the issue is clearly visible in the shown code.
  Never speculate about code that is not shown.
- One finding per distinct issue; do not repeat the same issue.
- The diff is a fragment, so do NOT flag a symbol as a missing import just because
  its definition is not shown.
- Return ONLY the JSON array, no markdown fences. If there is nothing to report,
  return [].
"""


def _diff_for_prompt(state: ReviewState) -> str:
    return annotate_diff_with_line_numbers(state["diff"])


def _human_message(state: ReviewState, focus: str) -> HumanMessage:
    return HumanMessage(content=(
        f"Review this {state['language']} diff for {focus} ONLY.\n"
        f"Each line is prefixed with its line number as \"L<n>:\".\n\n"
        f"{_diff_for_prompt(state)}\n\n"
        f"Context: {state.get('context') or 'No additional context provided'}"
    ))


def security_reviewer_node(state: ReviewState) -> dict:
    llm = _get_llm()
    messages = [
        SystemMessage(content=(
            "You are a meticulous, paranoid application security reviewer.\n"
            "Find every real security vulnerability in the diff:\n"
            "- SQL injection: f-strings or string concatenation building SQL from variables.\n"
            "- Hardcoded credentials, API keys, tokens, or secrets committed in source.\n"
            "- Plaintext password storage: passwords written to the DB without bcrypt/argon2 hashing.\n"
            "- XSS: user-controlled data inserted into HTML/templates without escaping.\n"
            "- IDOR / missing authorisation: a function that reads or mutates data for a\n"
            "  user-supplied id or email WITHOUT verifying the caller owns that resource.\n"
            "  Check EVERY such function on its own, including password-reset and other\n"
            "  account-modifying endpoints, not only the obvious ones.\n"
            "- Unvalidated or unsanitised input flowing into a sensitive sink.\n"
            "Report only issues clearly visible in the shown code.\n\n"
            + _FINDING_SCHEMA
        )),
        _human_message(state, "SECURITY vulnerabilities"),
    ]
    response = _invoke_with_retry(llm, messages)
    return {"security_findings": _parse_json_array(response.content)}


def performance_reviewer_node(state: ReviewState) -> dict:
    llm = _get_fast_llm()
    messages = [
        SystemMessage(content=(
            "You are a performance reviewer focused on issues that bite at scale. These\n"
            "categories are distinct, do not confuse them:\n"
            "- N+1 QUERIES: a DB/IO call inside a loop over a finite collection, i.e. any\n"
            "  `for`, `for...of`, `for...in`, `foreach`, `for x in range(...)`, `.map`,\n"
            "  or `.forEach`. A 100k-iteration `for` loop issuing one query each is a\n"
            "  textbook N+1. Title it 'N+1 query', severity high. A finite for-loop is\n"
            "  NEVER an infinite loop.\n"
            "- INFINITE / UNBOUNDED LOOPS: only a `while` / `do-while` / `for(;;)` whose\n"
            "  exit condition may never become false and that has no timeout or\n"
            "  max-attempts guard (e.g. polling a status with only a sleep). Title it\n"
            "  'infinite loop', severity critical. It holds a thread/connection forever.\n"
            "- SEQUENTIAL AWAITS: independent awaits run one-by-one inside a loop that\n"
            "  should run concurrently. Describe as 'sequential; parallelise with\n"
            "  Promise.all / asyncio.gather', severity high. Not an N+1.\n"
            "- Synchronous blocking calls on an async code path.\n"
            "- Repeated expensive work that should be cached.\n"
            "Only call something an N+1 if you can see the loop. Do NOT flag a single,\n"
            "non-looped query as N+1. Do NOT invent loops. Ignore micro-optimisations.\n\n"
            + _FINDING_SCHEMA
        )),
        _human_message(state, "PERFORMANCE issues"),
    ]
    response = _invoke_with_retry(llm, messages)
    return {"performance_findings": _parse_json_array(response.content)}


def correctness_reviewer_node(state: ReviewState) -> dict:
    llm = _get_fast_llm()
    messages = [
        SystemMessage(content=(
            "You are a correctness reviewer hunting bugs that crash or corrupt data in\n"
            "production:\n"
            "- Values read from request input (req.query / req.body / req.params /\n"
            "  request[...]) used WITHOUT a null or type check, e.g. calling .split() on a\n"
            "  query param that may be undefined throws at runtime.\n"
            "- Property access on a value that may be null/None, e.g. the result of a\n"
            "  findById / SELECT that returned nothing.\n"
            "- Unclosed file handles or other resource leaks.\n"
            "- Swallowed exceptions, wrong status codes, silent NaN or undefined results.\n"
            "- References to variables that are never defined in scope (ReferenceError).\n"
            "- Off-by-one errors and inverted boolean logic.\n"
            "Flag bugs that will actually fail in production, not style preferences.\n\n"
            + _FINDING_SCHEMA
        )),
        _human_message(state, "CORRECTNESS bugs and edge cases"),
    ]
    response = _invoke_with_retry(llm, messages)
    return {"correctness_findings": _parse_json_array(response.content)}


def style_reviewer_node(state: ReviewState) -> dict:
    llm = _get_fast_llm()
    messages = [
        SystemMessage(content=(
            "You are a code-quality reviewer. Flag ONLY issues that genuinely hurt\n"
            "readability or maintainability:\n"
            "- Overly broad types (e.g. `any` where a precise type clearly exists).\n"
            "- Magic numbers or strings that clearly should be named constants.\n"
            "- Dead code, genuinely duplicated logic, or oversized functions.\n"
            "Do NOT raise pure preferences (SELECT * , renaming a function, a lone literal),\n"
            "and do NOT restate a security or correctness bug as a style nit.\n\n"
            + _FINDING_SCHEMA
        )),
        _human_message(state, "CODE QUALITY issues"),
    ]
    response = _invoke_with_retry(llm, messages)
    return {"style_findings": _parse_json_array(response.content)}


def test_coverage_reviewer_node(state: ReviewState) -> dict:
    llm = _get_fast_llm()
    messages = [
        SystemMessage(content=(
            "You are a test-coverage reviewer. Name the specific tests the diff should\n"
            "have but appears to lack:\n"
            "- New functions or branches with no corresponding test.\n"
            "- Error and exception paths (e.g. the notification/email call throwing).\n"
            "- Edge cases the new code introduces: empty input, missing record, or a\n"
            "  double action such as cancelling an already-cancelled order.\n"
            "Give at most the 3-4 most valuable missing tests. Do NOT suggest 'testing'\n"
            "that a secret or constant holds a particular value.\n\n"
            + _FINDING_SCHEMA
        )),
        _human_message(state, "TEST COVERAGE gaps"),
    ]
    response = _invoke_with_retry(llm, messages)
    return {"test_coverage_findings": _parse_json_array(response.content)}


def _fallback_missing_tests(findings: List[Dict[str, Any]]) -> List[str]:
    out = []
    for f in findings:
        if f["category"] == "test_coverage":
            out.append(f["title"] if not f.get("suggestion") else f["suggestion"])
    return out[:4]


def merge_findings_node(state: ReviewState) -> dict:
    category_map = {
        "security": state.get("security_findings") or [],
        "performance": state.get("performance_findings") or [],
        "correctness": state.get("correctness_findings") or [],
        "style": state.get("style_findings") or [],
        "test_coverage": state.get("test_coverage_findings") or [],
    }

    # Everything structural (line numbers, FP filtering, dedup, severity, ids) is
    # done deterministically here. No LLM, fully unit-tested.
    deduped = process_findings(category_map, state["diff"])
    overall_severity = overall_severity_of(deduped)
    verdict = verdict_for(overall_severity)
    agent_counts = category_counts(deduped)

    # The LLM only writes the prose fields, never the structured facts above.
    findings_preview = json.dumps(deduped[:8], indent=2)
    meta_messages = [
        SystemMessage(content=(
            "You are a senior code reviewer writing a structured summary.\n"
            "Return ONLY a JSON object with exactly these fields:\n"
            "{\n"
            '  "pr_summary": "<one sentence: what does this PR do?>",\n'
            '  "verdict_reason": "<one sentence: why this verdict given the findings?>",\n'
            '  "positive_observations": ["<genuine strength 1>", "<genuine strength 2>"],\n'
            '  "missing_tests": ["<specific test case 1>", "<specific test case 2>"]\n'
            "}\n"
            "positive_observations must be genuine and at least 2. "
            "Return ONLY the JSON object. No markdown, no extra text."
        )),
        HumanMessage(content=(
            f"Summarise this code review.\n\n"
            f"Language: {state['language']}\n"
            f"Diff (first 1500 chars):\n{state['diff'][:1500]}\n\n"
            f"Key findings:\n{findings_preview}\n\n"
            f"Verdict: {verdict}\n"
            f"Overall severity: {overall_severity}"
        )),
    ]
    try:
        meta_response = _invoke_with_retry(_get_llm(), meta_messages)
        match = re.search(r"\{.*\}", meta_response.content, re.DOTALL)
        meta = json.loads(match.group()) if match else {}
    except Exception:
        meta = {}

    missing_tests = meta.get("missing_tests") or _fallback_missing_tests(deduped)

    return {
        "all_findings": deduped,
        "pr_summary": meta.get("pr_summary", "PR modifies existing functionality."),
        "verdict": verdict,
        "verdict_reason": meta.get(
            "verdict_reason",
            f"Found {len(deduped)} issue(s); highest severity is {overall_severity}.",
        ),
        "overall_severity": overall_severity,
        "positive_observations": meta.get(
            "positive_observations",
            ["Code is logically structured.", "PR scope is focused on a single concern."],
        ),
        "missing_tests": missing_tests,
        "agent_findings_count": agent_counts,
    }
