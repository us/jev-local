"""FastAPI app: POST /v1/systemone, Jev-compatible contract."""

import functools
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .models import SystemOneRequest, SystemOneResponse, Usage
from .scorer import DeterministicStubScorer, ScorerError

app = FastAPI(title="jev-local")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["POST", "GET"],
                   allow_headers=["Content-Type", "Authorization"])


@functools.lru_cache(maxsize=1)
def _get_scorer():
    if os.getenv("JEVLOCAL_SCORER") == "hf":
        from .scorer import HfLogprobScorer
        return HfLogprobScorer(model_id=os.getenv("JEVLOCAL_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"))
    return DeterministicStubScorer()


def _state_str(state) -> str:
    return state if isinstance(state, str) else str(state)


@app.post("/v1/systemone", response_model=SystemOneResponse)
def system_one(req: SystemOneRequest):
    scorer = _get_scorer()
    answers: dict = {}
    # usage fields are estimates (whitespace token count + scoring-token
    # count), not tokenizer counts; documented, not billed.
    prompt_tokens = len(_state_str(req.state).split())
    try:
        for qid, q in req.questions.items():
            prompt_tokens += len(str(q.instructions).split())
            if q.type == "noul":
                if q.criteria is not None and set(q.criteria) - {"true", "false"}:
                    raise HTTPException(status_code=422, detail=f"{qid}: noul criteria keys must be true/false")
                answers[qid] = scorer.noul(_state_str(req.state), q)
            elif q.type == "choice":
                if len(q.criteria) == 0:
                    raise HTTPException(status_code=422, detail=f"{qid}: choice criteria must not be empty")
                if len(q.criteria) > 255:
                    raise HTTPException(status_code=422, detail=f"{qid}: choice accepts up to 255 options")
                answers[qid] = scorer.choice(_state_str(req.state), q)
            elif q.type == "score":
                if not 2 <= len(q.criteria) <= 10:
                    raise HTTPException(status_code=422, detail=f"{qid}: score needs 2..10 levels")
                answers[qid] = scorer.score(_state_str(req.state), q)
    except ScorerError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    scoring_tokens = len(req.questions)
    return SystemOneResponse(
        model=getattr(scorer, "model_name", req.model),
        answers=answers,
        usage=Usage(input_tokens=prompt_tokens, output_tokens=scoring_tokens),
    )


@app.get("/health")
def health():
    return {"ok": True}
