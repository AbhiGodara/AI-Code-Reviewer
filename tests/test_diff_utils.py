"""Tests for deterministic diff parsing and line resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.diff_utils import (
    annotate_diff_with_line_numbers,
    has_loop_near,
    parse_diff_new_lines,
    resolve_line,
)

DIFFS_DIR = Path(__file__).resolve().parent.parent / "diffs"


def _load(name: str) -> str:
    return (DIFFS_DIR / name).read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def diff1() -> str:
    return _load("diff1_python.txt")


@pytest.fixture(scope="module")
def diff2() -> str:
    return _load("diff2_javascript.txt")


@pytest.fixture(scope="module")
def diff3() -> str:
    return _load("diff3_typescript.txt")


# parse_diff_new_lines

def test_diff1_line_numbers_are_exact(diff1):
    parsed = parse_diff_new_lines(diff1)
    by_no = {nl.number: nl.content for nl in parsed}

    assert by_no[1].startswith("# payments_service.py")
    assert by_no[2] == ""  # added blank line
    assert by_no[3] == "def get_transaction(user_id, transaction_id):"
    assert by_no[4].strip() == 'query = f"SELECT * FROM transactions WHERE user_id = {user_id}"'
    assert by_no[5].strip() == 'query += f" AND transaction_id = {transaction_id}"'
    assert "UPDATE transactions" in by_no[12]
    assert by_no[17].strip() == "template = open('templates/email.html').read()"
    assert by_no[27].strip() == "STRIPE_SECRET_KEY = 'sk_live_YOUR_STRIPE_KEY_HERE'"
    assert by_no[28].strip() == "MAX_RETRIES = 3"


def test_diff2_loop_line_numbers(diff2):
    by_no = {nl.number: nl.content for nl in parse_diff_new_lines(diff2)}
    assert by_no[7].strip() == "for (const id of userIds) {"
    assert "db.query('SELECT * FROM users WHERE id = ?', [id])" in by_no[8]
    assert by_no[5].strip() == "const userIds = ids.split(',');"


def test_diff3_line_numbers(diff3):
    by_no = {nl.number: nl.content for nl in parse_diff_new_lines(diff3)}
    assert by_no[13].strip() == "while (status === 'pending') {"
    assert "const discounts: any" in by_no[28]
    assert "return price * (1 - discounts[discountCode]);" in by_no[29]


def test_numbers_are_contiguous_from_one(diff1):
    numbers = [nl.number for nl in parse_diff_new_lines(diff1)]
    assert numbers == list(range(1, len(numbers) + 1))


# annotate

def test_annotate_prefixes_each_line(diff1):
    annotated = annotate_diff_with_line_numbers(diff1)
    assert "L27: STRIPE_SECRET_KEY = 'sk_live_YOUR_STRIPE_KEY_HERE'" in annotated
    assert annotated.splitlines()[0].startswith("L1: ")


# resolve_line: the core fix for hallucinated line numbers

def test_resolve_corrects_a_wrong_claimed_line(diff1):
    parsed = parse_diff_new_lines(diff1)
    # The LLM claimed line 20 for the Stripe key; the truth is line 27.
    line, content = resolve_line("STRIPE_SECRET_KEY = 'sk_live_YOUR_STRIPE_KEY_HERE'", 20, parsed)
    assert line == 27
    assert content.strip().startswith("STRIPE_SECRET_KEY")


def test_resolve_update_statement(diff1):
    parsed = parse_diff_new_lines(diff1)
    update = "        db.execute(\"UPDATE transactions SET status='refunded' WHERE id=\" + str(transaction['id']))"
    line, _ = resolve_line(update, 11, parsed)
    assert line == 12


def test_resolve_tolerates_lnn_prefix_and_substring(diff1):
    parsed = parse_diff_new_lines(diff1)
    # Model echoed a "L99:" prefix and only a fragment of the line.
    line, content = resolve_line("L99: query += f\" AND transaction_id", 99, parsed)
    assert line == 5
    assert content.strip().startswith("query +=")


def test_resolve_falls_back_when_no_match(diff1):
    parsed = parse_diff_new_lines(diff1)
    line, content = resolve_line("this code does not exist anywhere", 42, parsed)
    assert line == 42
    assert content is None


# has_loop_near: backstop for N+1 hallucinations

def test_loop_detected_inside_getusers(diff2):
    parsed = parse_diff_new_lines(diff2)
    assert has_loop_near(parsed, 8) is True   # db.query inside the for-loop


def test_no_loop_near_single_update(diff1):
    parsed = parse_diff_new_lines(diff1)
    assert has_loop_near(parsed, 12) is False  # UPDATE is not in any loop


def test_loop_detected_near_while(diff3):
    parsed = parse_diff_new_lines(diff3)
    assert has_loop_near(parsed, 14) is True   # findById inside the while-loop
