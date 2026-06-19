# AI Code Reviewer

A FastAPI service backed by a LangGraph multi-agent pipeline that reviews a raw
GitHub PR diff and returns a structured, line-level `ReviewReport` covering
security, performance, correctness, style, and test coverage, with real line
numbers, per-finding severity, and an overall verdict.

Built for the Infravox AI Backend Intern assignment. LLM: Groq free tier
(`llama-3.3-70b-versatile` and `llama-3.1-8b-instant`).

## Quickstart

Requires Python 3.10+.

```bash
pip install -r requirements.txt

# add your Groq key (free at console.groq.com, no card needed)
cp .env.example .env          # then edit .env and paste your GROQ_API_KEY

# start the API
uvicorn app.main:app --reload
```

In a second terminal, generate the three review reports from the sample diffs:

```bash
python run_reviews.py         # reads diffs/, POSTs to /review, writes reviews/
```

Run the tests (no API key needed; the deterministic core runs offline):

```bash
pytest -q
```

## API

| Method & path        | Description |
|----------------------|-------------|
| `POST /review`       | Body `{ diff, language, context? }`, returns a full `ReviewReport`. |
| `GET /review/{id}`   | Fetch a previously generated review by `review_id`. |
| `GET /reviews`       | List this session's reviews (id, summary, verdict, severity, created_at). |
| `GET /health`        | Service status plus a live Groq connectivity check. |

Reviews live in an in-memory, session-scoped store.

## Architecture

A LangGraph `StateGraph` fans out from `START` to five specialist agents that run
in parallel (security, performance, correctness, style, test_coverage). Each one
writes to its own key in the state, so the parallel writes never collide. The
graph then fans back in to a single `merge_findings` node.

The merge node is split into two halves on purpose:

1. **A deterministic core (no LLM, fully unit-tested).** This is the spine of the
   report and where most of the work went:
   - *Real line numbers.* Rather than trust the model's line numbers (which drift
     as it reads down a file), the diff's `@@` hunk headers are parsed to map
     every new-file line, and each finding's `line_content` is matched back to
     that map (`pipeline/diff_utils.py`). The `line` field is computed, not guessed.
   - *False-positive filtering.* Drops N+1 claims with no loop nearby, "missing
     import" complaints (the diff is a fragment), self-contradicting "it's
     parameterised but still SQL injection" findings, and low-value style nits.
     A finding whose `line_content` isn't in the diff at all is dropped as a
     hallucination.
   - *Deduplication.* Collapses the same issue reported by several agents (now
     reliable because line numbers are exact) and cross-category restatements.
   - *Severity calibration (upgrade only).* Pins canonical criticals: SQL
     injection, hardcoded secrets, plaintext passwords, and unbounded polling
     loops with no timeout. The infinite-loop upgrade requires an actual `while`
     loop near the finding, so a finite `for...of` N+1 is never inflated.
2. **One LLM call** for the prose fields only (`pr_summary`, `verdict_reason`,
   `positive_observations`, `missing_tests`), never for the structured facts.

### Model choice

The security agent and the merge summary use `llama-3.3-70b-versatile` for the
highest-stakes work; the other agents use `llama-3.1-8b-instant`, which has a far
more generous free-tier quota. All five agents fire at once, so a transient
rate-limit spike is absorbed by retries with exponential backoff and jitter.

## Project structure

```
app/        FastAPI app and routes
models/     Pydantic schemas (ReviewReport, Finding, ReviewRequest, ...)
pipeline/   agents.py (5 specialists + merge), graph.py (LangGraph wiring),
            state.py, diff_utils.py + merge_utils.py (the deterministic core)
diffs/      the three sample PR diffs
reviews/    the generated ReviewReport JSON for each diff
tests/      44 offline unit tests for the deterministic core
run_reviews.py   reads diffs/, POSTs them, writes reviews/
```

## Design decisions

**What I'm happiest with:** pulling line numbering, dedup, FP filtering, and
severity out of the LLM and into a deterministic, unit-tested core. Models are
good at spotting and explaining a bug and unreliable at counting lines and not
repeating themselves, so splitting the work along that line made the output
stable and the logic testable without spending API calls (the 44 tests run in
under 0.2s with no key).

**What I'm least happy with:** the false positives that still live purely in the
LLM agents. A `.map()` with no DB call inside can still draw a "parallelise the
queries" comment, and the small model occasionally calls a polling loop an "N+1"
(the deterministic layer fixes its severity but not its title). With another day
I'd add a lightweight LLM critic pass that scores each finding's confidence
before it lands in the report, and give the merge layer enough scope awareness to
know whether a loop body actually contains the IO it's accused of.

## Note on AI assistance

This was built with significant AI help (Claude): scaffolding the FastAPI and
LangGraph structure, drafting the agent prompts, and most of all designing and
testing the deterministic merge core. Every prompt and merge rule was then
checked against the three sample diffs and the test suite and tuned from the
actual model output rather than accepted blind.
