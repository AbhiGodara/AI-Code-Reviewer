from __future__ import annotations

import asyncio
import time
from typing import Dict, List

from fastapi import APIRouter, HTTPException

from models.schemas import (
    AgentFindingsCount,
    Finding,
    ReviewReport,
    ReviewRequest,
    ReviewSummary,
)
from pipeline.graph import build_review_graph

router = APIRouter()

# In-memory review store (session-scoped)
_store: Dict[str, ReviewReport] = {}

# Build the graph once at import time so it's reused across requests
_graph = build_review_graph()


def _build_initial_state(request: ReviewRequest) -> dict:
    return {
        "diff": request.diff,
        "language": request.language,
        "context": request.context,
        "security_findings": [],
        "performance_findings": [],
        "correctness_findings": [],
        "style_findings": [],
        "test_coverage_findings": [],
        "all_findings": [],
        "pr_summary": "",
        "verdict": "",
        "verdict_reason": "",
        "overall_severity": "",
        "positive_observations": [],
        "missing_tests": [],
        "agent_findings_count": {},
    }


def _coerce_literal(value: str, allowed: tuple, default: str) -> str:
    return value if value in allowed else default


@router.post("/review", response_model=ReviewReport)
async def create_review(request: ReviewRequest) -> ReviewReport:
    start_ms = time.time()

    # Nothing to review, so short-circuit before burning LLM calls. Without this
    # the agents tend to hallucinate findings about code that isn't there.
    if not request.diff.strip():
        report = ReviewReport(
            pr_summary="Empty diff, no code changes to review.",
            verdict="approve",
            verdict_reason="No changes were submitted.",
            overall_severity="clean",
            findings=[],
            positive_observations=["No changes to review.", "Nothing to flag."],
            missing_tests=[],
            agent_findings_count=AgentFindingsCount(),
            processing_time_ms=int((time.time() - start_ms) * 1000),
        )
        _store[report.review_id] = report
        return report

    initial_state = _build_initial_state(request)

    # ainvoke runs sync node functions in a thread pool automatically
    result = await _graph.ainvoke(initial_state)

    processing_time_ms = int((time.time() - start_ms) * 1000)

    # Build the typed Finding list, coercing any invalid literals from the LLM.
    valid_categories = ("security", "performance", "correctness", "style", "test_coverage")
    valid_severities = ("critical", "high", "medium", "low")

    findings: List[Finding] = []
    for raw in result.get("all_findings", []):
        try:
            findings.append(
                Finding(
                    id=raw.get("id", ""),
                    line=int(raw.get("line", 0)),
                    line_content=str(raw.get("line_content", "")),
                    category=_coerce_literal(raw.get("category", ""), valid_categories, "correctness"),
                    severity=_coerce_literal(raw.get("severity", ""), valid_severities, "low"),
                    title=str(raw.get("title", "")),
                    description=str(raw.get("description", "")),
                    suggestion=str(raw.get("suggestion", "")),
                )
            )
        except Exception:
            continue

    counts = result.get("agent_findings_count", {})
    valid_verdicts = ("approve", "request_changes", "needs_discussion")
    valid_overall = ("critical", "high", "medium", "low", "clean")

    report = ReviewReport(
        pr_summary=result.get("pr_summary", "PR modifies existing functionality."),
        verdict=_coerce_literal(result.get("verdict", ""), valid_verdicts, "needs_discussion"),
        verdict_reason=result.get("verdict_reason", "Review complete."),
        overall_severity=_coerce_literal(result.get("overall_severity", ""), valid_overall, "medium"),
        findings=findings,
        positive_observations=result.get("positive_observations", []),
        missing_tests=result.get("missing_tests", []),
        agent_findings_count=AgentFindingsCount(
            security=counts.get("security", 0),
            performance=counts.get("performance", 0),
            correctness=counts.get("correctness", 0),
            style=counts.get("style", 0),
            test_coverage=counts.get("test_coverage", 0),
        ),
        processing_time_ms=processing_time_ms,
    )

    _store[report.review_id] = report
    return report


@router.get("/review/{review_id}", response_model=ReviewReport)
async def get_review(review_id: str) -> ReviewReport:
    if review_id not in _store:
        raise HTTPException(status_code=404, detail=f"Review '{review_id}' not found.")
    return _store[review_id]


@router.get("/reviews", response_model=List[ReviewSummary])
async def list_reviews() -> List[ReviewSummary]:
    return [
        ReviewSummary(
            review_id=r.review_id,
            pr_summary=r.pr_summary,
            verdict=r.verdict,
            overall_severity=r.overall_severity,
            created_at=r.created_at,
        )
        for r in _store.values()
    ]


@router.get("/health")
async def health() -> dict:
    import os

    groq_key = os.environ.get("GROQ_API_KEY", "")
    groq_configured = bool(groq_key and groq_key.startswith("gsk_"))

    groq_connected = False
    if groq_configured:
        try:
            from langchain_groq import ChatGroq
            # Same model the fast agents use, so the check reflects real availability.
            llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, groq_api_key=groq_key)
            await asyncio.to_thread(llm.invoke, [{"role": "user", "content": "ping"}])
            groq_connected = True
        except Exception:
            groq_connected = False

    return {
        "status": "ok",
        "groq_configured": groq_configured,
        "groq_connected": groq_connected,
        "reviews_in_session": len(_store),
    }
