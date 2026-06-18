"""
Adversarial tests for realistic diffs the assignment samples don't cover:
multiple hunks, context lines, removed lines, section headings on @@ lines,
duplicate lines, and empty input.  This is the shape the unseen 4th interview
diff is most likely to take.
"""

from __future__ import annotations

from pipeline.diff_utils import has_loop_near, parse_diff_new_lines, resolve_line
from pipeline.merge_utils import (
    calibrate_severity,
    overall_severity_of,
    process_findings,
    verdict_for,
)

MULTI_HUNK = """\
--- a/foo.py
+++ b/foo.py
@@ -1,6 +1,7 @@
 import os
 import sys
-def old():
+def new(x):
+    return x
     pass

 CONST = 1
@@ -20,3 +21,4 @@ def tail():
     a = 1
     b = 2
+    c = 3
     return a
"""


def test_multi_hunk_context_and_removed_lines():
    by_no = {nl.number: nl.content for nl in parse_diff_new_lines(MULTI_HUNK)}
    # First hunk: removed '-def old()' must NOT advance the new-file counter.
    assert by_no[1].strip() == "import os"
    assert by_no[2].strip() == "import sys"
    assert by_no[3].strip() == "def new(x):"
    assert by_no[4].strip() == "return x"
    assert by_no[5].strip() == "pass"
    assert by_no[7].strip() == "CONST = 1"
    # Second hunk jumps to new-file line 21 per the @@ header (with section heading).
    assert by_no[21].strip() == "a = 1"
    assert by_no[23].strip() == "c = 3"
    assert by_no[24].strip() == "return a"


def test_resolve_in_second_hunk():
    parsed = parse_diff_new_lines(MULTI_HUNK)
    line, content = resolve_line("c = 3", 1, parsed)  # claimed wrongly as line 1
    assert line == 23
    assert content.strip() == "c = 3"


DUP_LINES = """\
--- a/dup.py
+++ b/dup.py
@@ -1,0 +1,6 @@
+def f():
+    x = compute()
+    log(x)
+def g():
+    x = compute()
+    log(x)
"""


def test_duplicate_line_resolves_to_closest_claim():
    parsed = parse_diff_new_lines(DUP_LINES)
    # 'x = compute()' appears on lines 2 and 5; claim near 5 should pick 5.
    line, _ = resolve_line("x = compute()", 5, parsed)
    assert line == 5
    line2, _ = resolve_line("x = compute()", 2, parsed)
    assert line2 == 2


def test_empty_diff_is_safe():
    assert parse_diff_new_lines("") == []
    assert process_findings({"security": []}, "") == []


def test_no_findings_yields_clean_approve():
    findings = process_findings({c: [] for c in ("security", "performance", "correctness")}, MULTI_HUNK)
    assert findings == []
    assert overall_severity_of(findings) == "clean"
    assert verdict_for(overall_severity_of(findings)) == "approve"


def test_loop_guard_does_not_drop_legit_perf_without_loop_words():
    """A real perf issue that doesn't claim a loop (e.g. sync blocking call) is kept."""
    cat_map = {
        "performance": [
            {
                "line": 4, "line_content": "return x",
                "category": "performance", "severity": "medium",
                "title": "Synchronous blocking call",
                "description": "Blocking IO on the event loop thread stalls all requests.",
                "suggestion": "Use an async client.",
            }
        ]
    }
    findings = process_findings(cat_map, MULTI_HUNK)
    assert len(findings) == 1
    assert findings[0]["category"] == "performance"


def test_infinite_loop_calibrated_to_critical():
    f = {"title": "Infinite polling loop", "description": "while loop with no timeout, never returns",
         "severity": "high"}
    assert calibrate_severity(f) == "critical"


def test_blank_added_line_counts_as_a_line():
    diff = "--- a/x\n+++ b/x\n@@ -0,0 +1,3 @@\n+a = 1\n+\n+b = 2\n"
    by_no = {nl.number: nl.content for nl in parse_diff_new_lines(diff)}
    assert by_no[1] == "a = 1"
    assert by_no[2] == ""      # the added blank line
    assert by_no[3] == "b = 2"
