from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()  # Must happen before any module that reads env vars is imported

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import router

app = FastAPI(
    title="AI Code Reviewer",
    description=(
        "Agentic FastAPI service powered by a LangGraph multi-agent pipeline. "
        "Accepts a raw GitHub PR diff and returns a structured, line-level ReviewReport "
        "covering security, performance, correctness, style, and test coverage."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
