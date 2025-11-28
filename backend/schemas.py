from typing import List
from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str
    top_k: int = 5


class ChatResponse(BaseModel):
    answer: str
    context_documents: List[str]