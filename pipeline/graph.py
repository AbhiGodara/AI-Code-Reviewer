from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from pipeline.agents import (
    correctness_reviewer_node,
    merge_findings_node,
    performance_reviewer_node,
    security_reviewer_node,
    style_reviewer_node,
    test_coverage_reviewer_node,
)
from pipeline.state import ReviewState


def build_review_graph():
    """Build and compile the review pipeline.

    The five specialist agents fan out from START and run in parallel; LangGraph
    waits for all of them before running merge_findings, which consolidates and
    deduplicates their output.
    """
    workflow = StateGraph(ReviewState)

    workflow.add_node("security_reviewer", security_reviewer_node)
    workflow.add_node("performance_reviewer", performance_reviewer_node)
    workflow.add_node("correctness_reviewer", correctness_reviewer_node)
    workflow.add_node("style_reviewer", style_reviewer_node)
    workflow.add_node("test_coverage_reviewer", test_coverage_reviewer_node)
    workflow.add_node("merge_findings", merge_findings_node)

    # Fan out from START to all 5 agents.
    workflow.add_edge(START, "security_reviewer")
    workflow.add_edge(START, "performance_reviewer")
    workflow.add_edge(START, "correctness_reviewer")
    workflow.add_edge(START, "style_reviewer")
    workflow.add_edge(START, "test_coverage_reviewer")

    # Fan back in to the merge node.
    workflow.add_edge("security_reviewer", "merge_findings")
    workflow.add_edge("performance_reviewer", "merge_findings")
    workflow.add_edge("correctness_reviewer", "merge_findings")
    workflow.add_edge("style_reviewer", "merge_findings")
    workflow.add_edge("test_coverage_reviewer", "merge_findings")

    workflow.add_edge("merge_findings", END)

    return workflow.compile()
