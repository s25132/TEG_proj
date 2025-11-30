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
    top_k: int = 5  # ile najistotniejszych elementów grafu (np. ścieżek / trójek) chcesz


class GraphRagResponse(BaseModel):
    answer: str
    # np. listy “zflattenowanych” ścieżek lub trójek grafowych w formie tekstu
    context_subgraphs: List[str]