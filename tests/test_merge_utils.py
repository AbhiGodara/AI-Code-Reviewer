"""
Tests for the deterministic merge pipeline.

The synthetic findings below are deliberately modelled on the *real* messy
output observed in reviews/diff1_review.json and diff2_review.json: wrong line
numbers, an N+1 hallucination on a non-loop, a duplicate SQL finding split
across security/style, a nonsensical "test the secret key" finding, and a
"missing import" complaint.  The pipeline must clean all of that up.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.merge_utils import (
    calibrate_severity,
    category_counts,
    deduplicate,
    overall_severity_of,
    process_findings,
    soft_cap,
    verdict_for,
)

DIFFS_DIR = Path(__file__).resolve().parent.parent / "diffs"


@pytest.fixture(scope="module")
def diff1() -> str:
    return (DIFFS_DIR / "diff1_python.txt").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def diff2() -> str:
    return (DIFFS_DIR / "diff2_javascript.txt").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def diff3() -> str:
    return (DIFFS_DIR / "diff3_typescript.txt").read_text(encoding="utf-8")


@pytest.fixture
def messy_diff1_findings() -> dict:
    return {
        "security": [
            {
                "line": 4,
                "line_content": 'query = f"SELECT * FROM transactions WHERE user_id = {user_id}"',
                "category": "security", "severity": "critical",
                "title": "SQL Injection", "description": "f-string interpolation into raw SQL.",
                "suggestion": "Use parameterised queries.",
            },
            {
                "line": 11,  # WRONG: real line is 12
                "line_content": "db.execute(\"UPDATE transactions SET status='refunded' WHERE id=\" + str(transaction['id']))",
                "category": "security", "severity": "high",  # should be upgraded to critical
                "title": "SQL Injection", "description": "String concatenation into SQL UPDATE.",
                "suggestion": "Parameterise.",
            },
            {
                "line": 20,  # WRONG: real line is 27
                "line_content": "STRIPE_SECRET_KEY = 'sk_live_YOUR_STRIPE_KEY_HERE'",
                "category": "security", "severity": "high",  # should be upgraded to critical
                "title": "Hardcoded Secret Key", "description": "Live Stripe secret key committed.",
                "suggestion": "Move to env var.",
            },
        ],
        "performance": [
            {
                "line": 10,  # resolves to 12 (the UPDATE); there is no loop here
                "line_content": "db.execute(\"UPDATE transactions SET status='refunded' WHERE id=\" + str(transaction['id']))",
                "category": "performance", "severity": "high",
                "title": "N+1 query pattern", "description": "Inside the loop a query runs per row.",
                "suggestion": "Batch it.",
            },
        ],
        "correctness": [
            {
                "line": 10,
                "line_content": "if transaction['status'] == 'completed':",
                "category": "correctness", "severity": "high",
                "title": "Missing null check for transaction",
                "description": "transaction may be None after get_transaction().",
                "suggestion": "Guard with if transaction is None.",
            },
        ],
        "style": [
            {
                "line": 10,  # resolves to 12, same line as the security SQLi -> should be dropped
                "line_content": "db.execute(\"UPDATE transactions SET status='refunded' WHERE id=\" + str(transaction['id']))",
                "category": "style", "severity": "high",
                "title": "SQL query concatenation",
                "description": "SQL built by concatenation; parameterise instead.",
                "suggestion": "Use placeholders.",
            },
            {
                "line": 15,
                "line_content": "smtp.send(email, body)",
                "category": "style", "severity": "medium",
                "title": "Missing import or module",
                "description": "The smtp module is not imported or defined.",
                "suggestion": "import smtplib.",
            },
        ],
        "test_coverage": [
            {
                "line": 20,
                "line_content": "STRIPE_SECRET_KEY = 'sk_live_YOUR_STRIPE_KEY_HERE'",
                "category": "test_coverage", "severity": "low",
                "title": "Missing test for secret key",
                "description": "No test verifies the Stripe secret key is correct.",
                "suggestion": "Add a test.",
            },
        ],
    }


def test_line_numbers_resolved_against_diff(messy_diff1_findings, diff1):
    findings = process_findings(messy_diff1_findings, diff1)
    by_title_line = {(f["title"], f["line"]) for f in findings}
    # UPDATE SQLi resolved 11 -> 12, Stripe key resolved 20 -> 27
    assert ("SQL Injection", 12) in by_title_line
    assert ("Hardcoded Secret Key", 27) in by_title_line


def test_n_plus_one_hallucination_dropped(messy_diff1_findings, diff1):
    findings = process_findings(messy_diff1_findings, diff1)
    # The only performance finding was an N+1 claim on a line with no loop.
    assert all(f["category"] != "performance" for f in findings)


def test_missing_import_finding_dropped(messy_diff1_findings, diff1):
    findings = process_findings(messy_diff1_findings, diff1)
    assert all("import" not in f["title"].lower() for f in findings)


def test_secret_key_test_finding_dropped(messy_diff1_findings, diff1):
    findings = process_findings(messy_diff1_findings, diff1)
    assert all(f["category"] != "test_coverage" for f in findings)


def test_cross_category_sql_duplicate_collapsed(messy_diff1_findings, diff1):
    findings = process_findings(messy_diff1_findings, diff1)
    on_line_12 = [f for f in findings if f["line"] == 12]
    # Security SQLi stays; the style "SQL query concatenation" on the same line is dropped.
    assert len(on_line_12) == 1
    assert on_line_12[0]["category"] == "security"


def test_severity_calibrated_up_to_critical(messy_diff1_findings, diff1):
    findings = process_findings(messy_diff1_findings, diff1)
    sev_by_title_line = {(f["title"], f["line"]): f["severity"] for f in findings}
    assert sev_by_title_line[("SQL Injection", 12)] == "critical"
    assert sev_by_title_line[("Hardcoded Secret Key", 27)] == "critical"


def test_ids_sequential_and_sorted_by_severity(messy_diff1_findings, diff1):
    findings = process_findings(messy_diff1_findings, diff1)
    assert [f["id"] for f in findings] == [f"F-{i:03d}" for i in range(1, len(findings) + 1)]
    ranks = [{"critical": 4, "high": 3, "medium": 2, "low": 1}[f["severity"]] for f in findings]
    assert ranks == sorted(ranks, reverse=True)


def test_overall_severity_and_verdict(messy_diff1_findings, diff1):
    findings = process_findings(messy_diff1_findings, diff1)
    assert overall_severity_of(findings) == "critical"
    assert verdict_for("critical") == "request_changes"
    assert verdict_for("medium") == "needs_discussion"
    assert verdict_for("clean") == "approve"


def test_counts_match_final_findings(messy_diff1_findings, diff1):
    findings = process_findings(messy_diff1_findings, diff1)
    counts = category_counts(findings)
    assert sum(counts.values()) == len(findings)


def test_real_n_plus_one_with_loop_is_kept(diff2):
    """A genuine N+1 (db.query inside a for-loop) must survive the filter."""
    category_map = {
        "performance": [
            {
                "line": 14,  # resolves to 8, which IS inside the for-loop
                "line_content": "const user = await db.query('SELECT * FROM users WHERE id = ?', [id]);",
                "category": "performance", "severity": "high",
                "title": "N+1 query pattern", "description": "One DB call per id inside the loop.",
                "suggestion": "Use WHERE id IN (?).",
            }
        ]
    }
    findings = process_findings(category_map, diff2)
    assert len(findings) == 1
    assert findings[0]["line"] == 8
    assert findings[0]["category"] == "performance"


# focused unit tests

def test_calibrate_only_upgrades():
    low_sqli = {"title": "SQL Injection", "description": "raw sql", "severity": "low"}
    assert calibrate_severity(low_sqli) == "critical"
    benign = {"title": "Magic number", "description": "use a constant", "severity": "medium"}
    assert calibrate_severity(benign) == "medium"


def test_infinite_loop_label_on_finite_for_loop_not_inflated(diff2):
    """The getUsers for...of N+1, even if mislabelled 'infinite loop', is a finite
    loop, so calibration must NOT push it to critical."""
    from pipeline.diff_utils import parse_diff_new_lines
    parsed = parse_diff_new_lines(diff2)
    mislabelled = {"title": "Infinite Loop: Sequential DB Queries",
                   "description": "db query inside the for loop", "severity": "high", "line": 8}
    assert calibrate_severity(mislabelled, parsed) == "high"


def test_real_infinite_while_loop_calibrated_critical(diff3):
    from pipeline.diff_utils import parse_diff_new_lines
    parsed = parse_diff_new_lines(diff3)
    real = {"title": "Infinite Loop with No Timeout",
            "description": "while poll never returns", "severity": "high", "line": 16}
    assert calibrate_severity(real, parsed) == "critical"


def test_n_plus_one_label_inside_while_loop_upgraded(diff3):
    """The polling-loop query, even if the model calls it 'N+1', is critical
    because it sits in an unbounded while loop."""
    from pipeline.diff_utils import parse_diff_new_lines
    parsed = parse_diff_new_lines(diff3)
    finding = {"title": "N+1 query", "category": "performance",
               "description": "a db query runs every iteration of the polling loop",
               "severity": "high", "line": 14}
    assert calibrate_severity(finding, parsed) == "critical"


def test_hardcoded_userid_not_calibrated_critical(diff3):
    """Passing a hardcoded 'system' user id is not a hardcoded *secret*."""
    from pipeline.diff_utils import parse_diff_new_lines
    parsed = parse_diff_new_lines(diff3)
    finding = {"title": "Hardcoded UserId", "category": "security",
               "description": "The userId 'system' is hardcoded.", "severity": "high", "line": 23}
    assert calibrate_severity(finding, parsed) == "high"


def test_hardcoded_secret_key_still_critical():
    finding = {"title": "Hardcoded Secret Key",
               "description": "A live Stripe secret key is committed.", "severity": "high"}
    assert calibrate_severity(finding) == "critical"


def test_soft_cap_trims_only_noisy_categories():
    findings = [
        {"category": "style", "severity": "low", "description": str(i), "line": i,
         "title": f"s{i}", "line_content": "", "suggestion": "", "id": ""}
        for i in range(6)
    ] + [
        {"category": "security", "severity": "high", "description": "x", "line": 99,
         "title": "sec", "line_content": "", "suggestion": "", "id": ""}
    ]
    capped = soft_cap(findings, {"style": 4, "test_coverage": 4})
    assert sum(1 for f in capped if f["category"] == "style") == 4
    assert sum(1 for f in capped if f["category"] == "security") == 1


def test_parameterized_sql_injection_fp_dropped(diff2):
    """A 'SQL injection' finding that admits the query is parameterised is noise."""
    cat_map = {
        "security": [
            {
                "line": 8,
                "line_content": "const user = await db.query('SELECT * FROM users WHERE id = ?', [id]);",
                "category": "security", "severity": "high",
                "title": "Potential SQL Injection",
                "description": "Although the code is using a parameterized query, which helps prevent SQL injection, validate the input.",
                "suggestion": "Validate the id.",
            }
        ]
    }
    assert process_findings(cat_map, diff2) == []


def test_real_sql_injection_with_parameterised_suggestion_kept(diff1):
    """A genuine injection whose *suggestion* mentions parameterisation must survive."""
    cat_map = {
        "security": [
            {
                "line": 4,
                "line_content": 'query = f"SELECT * FROM transactions WHERE user_id = {user_id}"',
                "category": "security", "severity": "critical",
                "title": "SQL Injection",
                "description": "An f-string interpolates user_id directly into raw SQL, so an attacker can inject SQL.",
                "suggestion": "Use parameterized queries instead.",
            }
        ]
    }
    findings = process_findings(cat_map, diff1)
    assert len(findings) == 1
    assert findings[0]["severity"] == "critical"


def test_style_select_star_and_magic_one_dropped(diff2):
    cat_map = {
        "style": [
            {
                "line": 8, "line_content": "const user = await db.query('SELECT * FROM users WHERE id = ?', [id]);",
                "category": "style", "severity": "low",
                "title": "SELECT * is not necessary",
                "description": "Specify exact columns instead of SELECT *.", "suggestion": "SELECT id, email",
            },
            {
                "line": 5, "line_content": "const userIds = ids.split(',');",
                "category": "style", "severity": "low",
                "title": "Magic number in calculation",
                "description": "The number 1 is used as a magic number.", "suggestion": "const x = 1;",
            },
        ]
    }
    assert process_findings(cat_map, diff2) == []


def test_style_dropped_when_real_bug_on_same_line(diff1):
    cat_map = {
        "security": [
            {
                "line": 4, "line_content": 'query = f"SELECT * FROM transactions WHERE user_id = {user_id}"',
                "category": "security", "severity": "critical",
                "title": "SQL Injection", "description": "f-string into raw SQL.", "suggestion": "Parameterise.",
            }
        ],
        "style": [
            {
                "line": 4, "line_content": 'query = f"SELECT * FROM transactions WHERE user_id = {user_id}"',
                "category": "style", "severity": "low",
                "title": "Unnecessary string concatenation",
                "description": "Use one f-string.", "suggestion": "merge.",
            }
        ],
    }
    findings = process_findings(cat_map, diff1)
    assert len(findings) == 1
    assert findings[0]["category"] == "security"


def test_planted_any_style_finding_survives_next_to_test_coverage(diff3):
    """The legit `any`-type style bug shares its line only with a test_coverage
    finding, so it must NOT be dropped."""
    full_line = "  const discounts: any = { SAVE10: 0.1, SAVE20: 0.2, SAVE50: 0.5 };"
    cat_map = {
        "test_coverage": [
            {
                "line": 28, "line_content": full_line,
                "category": "test_coverage", "severity": "medium",
                "title": "Missing test for invalid discount code",
                "description": "No test covers an unknown code.", "suggestion": "add test",
            }
        ],
        "style": [
            {
                "line": 28, "line_content": full_line,
                "category": "style", "severity": "medium",
                "title": "Overly broad any type",
                "description": "any defeats TypeScript type safety.", "suggestion": "Record<string, number>",
            }
        ],
    }
    findings = process_findings(cat_map, diff3)
    cats = {f["category"] for f in findings}
    assert "style" in cats and "test_coverage" in cats


def test_finding_referencing_absent_code_is_dropped(diff1):
    """A finding whose line_content is nowhere in the diff is a hallucination."""
    cat_map = {
        "security": [
            {
                "line": 4,
                "line_content": 'query = f"SELECT * FROM transactions WHERE user_id = {user_id}"',
                "category": "security", "severity": "critical", "title": "SQL Injection",
                "description": "f-string into SQL.", "suggestion": "parameterise",
            },
            {
                "line": 99,
                "line_content": "subprocess.call(user_supplied, shell=True)  # not in this diff at all",
                "category": "security", "severity": "high", "title": "Command Injection",
                "description": "shell=True with user input.", "suggestion": "avoid shell",
            },
        ]
    }
    titles = {f["title"] for f in process_findings(cat_map, diff1)}
    assert "SQL Injection" in titles
    assert "Command Injection" not in titles


def test_deduplicate_collapses_same_line_and_category():
    findings = [
        {"line": 5, "category": "correctness", "severity": "medium", "title": "Null check",
         "description": "short", "line_content": "", "suggestion": "", "id": ""},
        {"line": 5, "category": "correctness", "severity": "high", "title": "Null check",
         "description": "a much longer and richer description", "line_content": "", "suggestion": "", "id": ""},
    ]
    deduped = deduplicate(findings)
    assert len(deduped) == 1
    assert deduped[0]["severity"] == "high"  # richest kept
