"""Pydantic models mirroring the TypeSafe /v1/systemone contract.

Source: docs.typesafe.ai/api (+ primitives guides for limits).
Structured fields accept string | object | array (| null where documented).
"""

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field

Structured = Union[str, dict, list]
StructuredOrNull = Union[str, dict, list, None]


class NoulQuestion(BaseModel):
    type: Literal["noul"]
    instructions: Structured
    criteria: dict[str, StructuredOrNull] | None = None


class ChoiceQuestion(BaseModel):
    type: Literal["choice"]
    instructions: Structured
    criteria: dict[str, StructuredOrNull]


class ScoreQuestion(BaseModel):
    type: Literal["score"]
    instructions: Structured
    criteria: list[StructuredOrNull]


Question = Annotated[Union[NoulQuestion, ChoiceQuestion, ScoreQuestion], Field(discriminator="type")]


class SystemOneRequest(BaseModel):
    state: Structured
    model: str
    questions: dict[str, Question]


class NoulAnswer(BaseModel):
    type: Literal["noul"]
    noul: float


class ChoiceAnswer(BaseModel):
    type: Literal["choice"]
    choice: str
    probabilities: dict[str, float]
    confidence: float


class ScoreAnswer(BaseModel):
    type: Literal["score"]
    score: float
    legend: dict[str, Any]
    probabilities: dict[str, float]
    confidence: float


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class SystemOneResponse(BaseModel):
    model: str
    answers: dict[str, NoulAnswer | ChoiceAnswer | ScoreAnswer]
    usage: Usage
