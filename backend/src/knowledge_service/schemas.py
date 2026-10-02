from typing import Literal

from pydantic import BaseModel, Field


class Login(BaseModel):
    login: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=1, max_length=200)


class KnowledgeBaseInput(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    domain: Literal["education", "listening", "travel", "parenting", "finance"]
    allow_parser_egress: bool = False
    allow_llm_egress: bool = False


class MembershipInput(BaseModel):
    role: Literal["reader", "maintainer", "reviewer", "admin"]


class KnowledgeBasePatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    allow_parser_egress: bool | None = None
    allow_llm_egress: bool | None = None


class AclInput(BaseModel):
    reader_ids: list[str] | None = Field(default=None, max_length=1000)


class RollbackInput(BaseModel):
    version_id: str


class FeedbackPatch(BaseModel):
    status: Literal["open", "in_progress", "resolved", "dismissed"]
    resolution: str = Field(default="", max_length=2000)


class ReviewInput(BaseModel):
    decision: Literal["approved", "rejected"]
    reason: str = Field(default="", max_length=1000)


class SearchInput(BaseModel):
    kb_ids: list[str] = Field(min_length=1, max_length=20)
    question: str = Field(min_length=1, max_length=2000)
    mode: Literal["dense", "sparse", "bm25", "hybrid"] = "hybrid"
    rerank: bool = True
    limit: int = Field(default=5, ge=1, le=20)


class ConversationInput(BaseModel):
    kb_ids: list[str] = Field(min_length=1, max_length=20)


class AnswerInput(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class FeedbackInput(BaseModel):
    message_id: str
    category: Literal["answer_error", "citation_error", "expired", "missing", "irrelevant"]
    detail: str = Field(default="", max_length=2000)
