from typing import List
from pydantic import BaseModel


class RagRequest(BaseModel):
    question: str
    top_k: int = 5


class RagResponse(BaseModel):
    answer: str
    context_documents: List[str]


class GraphRagRequest(BaseModel):
    question: str


class GraphRagResponse(BaseModel):
    answer: str
    context_documents: List[str]