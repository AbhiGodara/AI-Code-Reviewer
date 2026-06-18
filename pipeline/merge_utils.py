"""Deterministic post-processing for the merge node.

No LLM here, so it can be unit-tested without a Groq key. Takes the raw findings
from each agent plus the diff and turns them into a clean, de-duplicated,
correctly-numbered list. Order: coerce, resolve line numbers, drop false
positives, calibrate severity, deduplicate, cap noise, sort, assign ids.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List

from pipeline.diff_utils import (
    NewLine,
    has_loop_near,
    has_unbounded_loop_near,
    parse_diff_new_lines,
    resolve_line,
)

SEVERITY_RANK = {"critical": 4, "high": 3, "medium": 2, "low": 1}

VALID_CATEGORIES = {"security", "performance", "correctness", "style", "test_coverage"}
VALID_SEVERITIES = {"critical", "high", "medium", "low"}

# When two categories land on the same line, the higher one wins.
CATEGORY_PRIORITY = {
    "security": 5,
    "correctness": 4,
    "performance": 3,
    "test_coverage": 2,
    "style": 1,
}

# Style and test_coverage attract low-value nitpicks, so cap how many we keep.
NOISE_CAPS = {"style": 3, "test_coverage": 3}

_WORD_RE = re.compile(r"[a-z0-9+]+")
_STOPWORDS = {
    "the", "this", "that", "with", "from", "into", "should", "could", "when",
    "which", "missing", "check", "code", "value", "using", "used", "and",
    "for", "not", "are", "has", "have", "would", "will", "can", "may",
}


def coerce_finding(raw: Dict[str, Any], default_category: str) -> Dict[str, Any]:
    """Normalise a raw LLM dict into a valid finding, fixing bad enums/types."""
    severity = str(raw.get("severity", "low")).lower().strip()
    if severity not in VALID_SEVERITIES:
        severity = "low"

    category = str(raw.get("category", default_category)).lower().strip()
    if category not in VALID_CATEGORIES:
        category = default_category

    try:
        line = int(raw.get("line", 0))
    except (TypeError, ValueError):
        line = 0

    return {
        "id": "",
        "line": line,
        "line_content": str(raw.get("line_content", "")),
        "category": category,
        "severity": severity,
        "title": str(raw.get("title", "")).strip()[:120],
        "description": str(raw.get("description", "")).strip(),
        "suggestion": str(raw.get("suggestion", "")).strip(),
    }


def resolve_findings(findings: List[Dict[str, Any]], parsed: List[NewLine]) -> List[Dict[str, Any]]:
    """Fix each finding's line number from the diff and drop any whose
    line_content isn't in the diff at all (a hallucination). Skipped when the
    diff didn't parse, so a malformed diff doesn't hide real bugs."""
    if not parsed:
        return findings

    resolved: List[Dict[str, Any]] = []
    for f in findings:
        line, content = resolve_line(f["line_content"], f["line"], parsed)
        if content is None:
            continue
        f["line"] = line
        f["line_content"] = content
        resolved.append(f)
    return resolved


_PERF_LOOP_CLAIM = (
    "n+1", "n + 1", "in a loop", "inside the loop", "inside a loop",
    "loop that should", "sequential", "should be batched", "batch query",
    "batch quer", "single query", "single batch", "repeated database quer",
    "repeated db quer", "repeated quer",
)

_IMPORT_FP = (
    "missing import", "not imported", "import the necessary",
    "add the import", "import statement is missing",
)

# Phrases that admit the code is already safe; a "SQL injection" finding saying
# this is contradicting itself.
_SAFE_ADMISSIONS = (
    "although", "using a parameterized", "using a parameterised",
    "is parameterized", "is parameterised", "already parameterized",
    "uses parameterized", "uses parameterised", "helps prevent",
    "parameterized query, which", "parameterised query, which", "is safe from",
)


def is_spurious(finding: Dict[str, Any], parsed: List[NewLine]) -> bool:
    """Whether a finding is almost certainly noise. Every rule below is general,
    not tied to any specific diff."""
    title = finding.get("title", "").strip()
    description = finding.get("description", "").strip()
    if not title or not description:
        return True

    category = finding.get("category")
    desc_low = description.lower()
    text = " ".join((title, description, finding.get("suggestion", ""))).lower()

    # An N+1/batch/sequential claim with no loop in sight.
    if category == "performance":
        if any(tok in text for tok in _PERF_LOOP_CLAIM) and not has_loop_near(parsed, finding["line"]):
            return True

    # "Missing import" complaints: the diff is only a fragment.
    if any(tok in text for tok in _IMPORT_FP):
        return True

    # Pointless to test that a secret holds a given value.
    if category == "test_coverage":
        if any(tok in text for tok in ("secret key", "api key", "credential", "key is correct")):
            return True

    # An "injection" finding whose own description says the query is parameterised.
    # Only the description is checked; a real fix legitimately mentions it too.
    if category == "security" and "injection" in (title + " " + desc_low).lower():
        if any(adm in desc_low for adm in _SAFE_ADMISSIONS):
            return True

    # Style nits that are never worth raising.
    if category == "style":
        if "select *" in text:
            return True
        if "magic number" in text and re.search(r"\bnumber\s+[01]\b", text):
            return True

    return False


def filter_false_positives(findings: List[Dict[str, Any]], parsed: List[NewLine]) -> List[Dict[str, Any]]:
    return [f for f in findings if not is_spurious(f, parsed)]


# Patterns that are critical no matter the surrounding code. "hardcoded" is
# scoped to actual secrets; a hardcoded user id or flag is not one.
_CRITICAL_MARKERS = (
    "sql injection", "sqli",
    "hardcoded secret", "hardcoded api key", "hardcoded credential",
    "hardcoded password", "hardcoded token", "hardcoded private key",
    "hard-coded secret", "hard coded secret", "hardcoded key",
    "secret key", "live key", "live secret", "private key committed",
    "api key in source", "api key committed",
    "plain text password", "plaintext password", "plain-text password",
    "password in plain", "without hashing", "no hashing", "unhashed password",
)


def calibrate_severity(finding: Dict[str, Any], parsed: List[NewLine] = None) -> str:
    """Bump well-known critical patterns up to critical (upgrade only, never
    down). The loop-related upgrades need a real unbounded loop near the finding,
    so a finite for...of N+1 is never inflated."""
    text = (finding.get("title", "") + " " + finding.get("description", "")).lower()
    if any(marker in text for marker in _CRITICAL_MARKERS):
        return "critical"

    loop_marker = (
        "infinite loop" in text
        or "unbounded loop" in text
        or "never returns" in text
        or ("no timeout" in text and ("loop" in text or "poll" in text))
    )
    if loop_marker and (parsed is None or has_unbounded_loop_near(parsed, finding.get("line", 0))):
        return "critical"

    # A performance finding inside a while/for(;;) loop is critical even if the
    # model called it an "N+1" (the work in it may never stop).
    if (
        finding.get("category") == "performance"
        and parsed is not None
        and any(w in text for w in ("loop", "poll", "iteration", "iterate", "every time"))
        and has_unbounded_loop_near(parsed, finding.get("line", 0))
    ):
        return "critical"

    return finding.get("severity", "low")


def _keywords(text: str) -> set:
    return {w for w in _WORD_RE.findall(text.lower()) if len(w) >= 3 and w not in _STOPWORDS}


def _richness(finding: Dict[str, Any]) -> tuple:
    return (SEVERITY_RANK.get(finding["severity"], 0), len(finding.get("description", "")))


def _dominated(finding: Dict[str, Any], kept: List[Dict[str, Any]]) -> bool:
    """Whether `finding` should give way to one already kept on the same line.
    `kept` is built highest-priority-first."""
    for k in kept:
        shared = len(_keywords(finding["title"]) & _keywords(k["title"]))
        # Same issue restated, e.g. security SQLi also reported by correctness.
        if shared >= 2:
            return True
        # Style nitpicking on a line that already has a real bug.
        if finding["category"] == "style" and k["category"] in ("security", "correctness", "performance"):
            return True
        # A noise-category finding overlapping a strictly higher-priority one.
        if (
            finding["category"] in ("style", "test_coverage")
            and shared >= 1
            and CATEGORY_PRIORITY.get(k["category"], 0) > CATEGORY_PRIORITY.get(finding["category"], 0)
        ):
            return True
    return False


def deduplicate(findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pass 1 merges equal (line, category) pairs, keeping the richest. Pass 2,
    per line, drops findings dominated by a higher-priority one."""
    best: Dict[tuple, Dict[str, Any]] = {}
    for f in findings:
        key = (f["line"], f["category"])
        current = best.get(key)
        if current is None or _richness(f) > _richness(current):
            best[key] = f

    by_line: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for f in best.values():
        by_line[f["line"]].append(f)

    result: List[Dict[str, Any]] = []
    for group in by_line.values():
        ordered = sorted(
            group,
            key=lambda x: (-CATEGORY_PRIORITY.get(x["category"], 0), -_richness(x)[0], -_richness(x)[1]),
        )
        kept: List[Dict[str, Any]] = []
        for f in ordered:
            if not _dominated(f, kept):
                kept.append(f)
        result.extend(kept)
    return result


def soft_cap(findings: List[Dict[str, Any]], caps: Dict[str, int]) -> List[Dict[str, Any]]:
    """Keep at most caps[category] findings for the capped categories, preferring
    higher severity then richer descriptions."""
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for f in findings:
        buckets[f["category"]].append(f)

    result: List[Dict[str, Any]] = []
    for category, items in buckets.items():
        cap = caps.get(category)
        if cap is not None and len(items) > cap:
            items = sorted(
                items,
                key=lambda x: (-SEVERITY_RANK.get(x["severity"], 0), -len(x.get("description", ""))),
            )[:cap]
        result.extend(items)
    return result


def process_findings(category_map: Dict[str, List[Dict[str, Any]]], diff: str) -> List[Dict[str, Any]]:
    """Run the whole deterministic merge and return final findings, most severe
    first, with sequential F-00n ids."""
    parsed = parse_diff_new_lines(diff)

    combined: List[Dict[str, Any]] = []
    for category, raw_list in category_map.items():
        for raw in raw_list or []:
            if isinstance(raw, dict):
                combined.append(coerce_finding(raw, category))

    combined = resolve_findings(combined, parsed)
    combined = filter_false_positives(combined, parsed)
    for f in combined:
        f["severity"] = calibrate_severity(f, parsed)

    deduped = deduplicate(combined)
    deduped = soft_cap(deduped, NOISE_CAPS)

    deduped.sort(key=lambda x: (-SEVERITY_RANK.get(x["severity"], 0), x["line"]))
    for i, f in enumerate(deduped, 1):
        f["id"] = f"F-{i:03d}"
    return deduped


def overall_severity_of(findings: List[Dict[str, Any]]) -> str:
    if not findings:
        return "clean"
    return max(findings, key=lambda f: SEVERITY_RANK.get(f["severity"], 0))["severity"]


def verdict_for(overall_severity: str) -> str:
    if overall_severity in ("critical", "high"):
        return "request_changes"
    if overall_severity == "medium":
        return "needs_discussion"
    return "approve"


def category_counts(findings: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {c: 0 for c in VALID_CATEGORIES}
    for f in findings:
        counts[f["category"]] = counts.get(f["category"], 0) + 1
    return counts
