"""Unified-diff helpers.

Line numbers come from here, not from the LLM. A model reading a diff guesses
line numbers and drifts further down the file, so instead we compute every
number from the @@ hunk headers and anchor findings to the real new-file line.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Matches: @@ -<old_start>[,<old_len>] +<new_start>[,<new_len>] @@
_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")

# A leading "L42:" / "42:" prefix the model sometimes echoes back into line_content.
_LNUM_PREFIX_RE = re.compile(r"^\s*[lL]?\s*\d+\s*[:\-]\s*")

# Loop constructs, used to sanity-check "N+1 / inside a loop" claims.
_LOOP_TOKENS = (
    "for ", "for(", "while ", "while(", "foreach", ".map(", ".foreach(",
    ".filter(", ".reduce(", " in range(", "do {",
)

# Loops that can run forever, unlike a finite for...of. Used to decide whether an
# "infinite loop" finding really sits on an unbounded loop before bumping it to critical.
_UNBOUNDED_LOOP_TOKENS = ("while ", "while(", "do {", "do{", "for (;;)", "for(;;)")


@dataclass(frozen=True)
class NewLine:
    number: int      # 1-based line number in the new file
    content: str     # line text with the diff +/space marker stripped


def parse_diff_new_lines(diff: str) -> List[NewLine]:
    """Return every line present in the new file (added and context lines), each
    with its real new-file line number from the @@ headers. Removed lines belong
    to the old file only and don't advance the counter.
    """
    out: List[NewLine] = []
    new_no = 0
    in_hunk = False

    for raw in diff.splitlines():
        if raw.startswith("+++") or raw.startswith("---"):
            continue

        hunk = _HUNK_RE.match(raw)
        if hunk:
            new_no = int(hunk.group(1))
            in_hunk = True
            continue

        if not in_hunk:
            continue

        if raw.startswith("+"):
            out.append(NewLine(new_no, raw[1:]))
            new_no += 1
        elif raw.startswith("-"):
            continue  # old file only
        elif raw.startswith("\\"):
            continue  # e.g. "\ No newline at end of file"
        else:
            # Context line: starts with a single space, or is a bare blank line.
            content = raw[1:] if raw.startswith(" ") else raw
            out.append(NewLine(new_no, content))
            new_no += 1

    return out


def annotate_diff_with_line_numbers(diff: str) -> str:
    """Render the new-file view with an explicit 'Ln:' prefix on each line so the
    model can copy line numbers instead of counting. Falls back to the raw diff.
    """
    lines = parse_diff_new_lines(diff)
    if not lines:
        return diff
    return "\n".join(f"L{nl.number}: {nl.content}" for nl in lines)


def _normalise(text: str) -> str:
    text = text.strip()
    if text[:1] in "+-":
        text = text[1:]
    text = _LNUM_PREFIX_RE.sub("", text)
    return text.strip()


def resolve_line(
    line_content: str,
    claimed_line: int,
    parsed: List[NewLine],
) -> Tuple[int, Optional[str]]:
    """Find the true new-file line for a finding by matching its line_content
    against the diff: exact, then substring, then fuzzy (closest to the claimed
    line on ties). Returns (line, canonical_content), with content None if nothing
    matched so the caller can tell a real line from a hallucinated one.
    """
    target = _normalise(line_content)
    if not target or not parsed:
        return claimed_line, None

    exact = [nl for nl in parsed if nl.content.strip() == target]
    if len(exact) == 1:
        return exact[0].number, exact[0].content
    if len(exact) > 1:
        best = min(exact, key=lambda nl: abs(nl.number - claimed_line))
        return best.number, best.content

    sub = [
        nl for nl in parsed
        if nl.content.strip()
        and (target in nl.content.strip() or nl.content.strip() in target)
    ]
    if len(sub) == 1:
        return sub[0].number, sub[0].content
    if len(sub) > 1:
        best = min(sub, key=lambda nl: abs(nl.number - claimed_line))
        return best.number, best.content

    best_nl: Optional[NewLine] = None
    best_ratio = 0.0
    for nl in parsed:
        candidate = nl.content.strip()
        if not candidate:
            continue
        ratio = difflib.SequenceMatcher(None, target, candidate).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_nl = nl
    if best_nl is not None and best_ratio >= 0.6:
        return best_nl.number, best_nl.content

    return claimed_line, None


def has_loop_near(parsed: List[NewLine], line: int, window: int = 2) -> bool:
    """Whether any loop appears within `window` lines. Used to drop N+1 claims
    where there's no loop in sight."""
    return _token_near(parsed, line, window, _LOOP_TOKENS)


def has_unbounded_loop_near(parsed: List[NewLine], line: int, window: int = 4) -> bool:
    """Whether a possibly non-terminating loop (while / do / for(;;)) is nearby.
    A finite for...of does not count."""
    return _token_near(parsed, line, window, _UNBOUNDED_LOOP_TOKENS)


def _token_near(parsed: List[NewLine], line: int, window: int, tokens: tuple) -> bool:
    for nl in parsed:
        if abs(nl.number - line) <= window:
            low = nl.content.lower()
            if any(tok in low for tok in tokens):
                return True
    return False
