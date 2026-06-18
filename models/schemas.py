from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class Finding(BaseModel):
    id: str = ""
    line: int
    line_content: str
    category: Literal["security", "performance", "correctness", "style", "test_coverage"]
    severity: Literal["critical", "high", "medium", "low"]
    title: str
    description: str
    suggestion: str


class AgentFindingsCount(BaseModel):
    security: int = 0
    performance: int = 0
    correctness: int = 0
    style: int = 0
    test_coverage: int = 0


class ReviewReport(BaseModel):
    review_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    pr_summary: str
    verdict: Literal["approve", "request_changes", "needs_discussion"]
    verdict_reason: str
    overall_severity: Literal["critical", "high", "medium", "low", "clean"]
    findings: List[Finding]
    positive_observations: List[str]
    missing_tests: List[str]
    agent_findings_count: AgentFindingsCount
    processing_time_ms: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ReviewRequest(BaseModel):
    diff: str
    language: str
    context: Optional[str] = None


class ReviewSummary(BaseModel):
    review_id: str
    pr_summary: str
    verdict: str
    overall_severity: str
    created_at: datetime
