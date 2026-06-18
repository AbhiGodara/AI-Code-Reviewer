from __future__ import annotations

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class ReviewState(TypedDict):
    # Inputs
    diff: str
    language: str
    context: Optional[str]

    # Per-agent findings (parallel nodes each write to their own field)
    security_findings: List[Dict[str, Any]]
    performance_findings: List[Dict[str, Any]]
    correctness_findings: List[Dict[str, Any]]
    style_findings: List[Dict[str, Any]]
    test_coverage_findings: List[Dict[str, Any]]

    # Merged output (written by merge_findings node)
    all_findings: List[Dict[str, Any]]
    pr_summary: str
    verdict: str
    verdict_reason: str
    overall_severity: str
    positive_observations: List[str]
    missing_tests: List[str]
    agent_findings_count: Dict[str, int]
