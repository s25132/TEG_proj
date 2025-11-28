import os

import chromadb
from chromadb import PersistentClient
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from dotenv import load_dotenv
from schemas import ChatRequest, ChatResponse
from chroma import build_context_from_chroma, call_llm_with_rag

load_dotenv(override=True)

# ---------- Konfiguracja ----------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Brak OPENAI_API_KEY w zmiennych środowiskowych / .env")

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "projects_and_cvs")

# Klient OpenAI
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Klient Chroma (Persistent – dane na dysku)
chroma_client: PersistentClient = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION_NAME)

# ---------- FastAPI ----------

app = FastAPI(title="API", version="1.0.0")

@app.post("/ask_rag", response_model=ChatResponse)
def chat_rag(request: ChatRequest):
    # 1. Query do Chroma
    try:
       chroma_result = build_context_from_chroma(
        openai_client=openai_client,
        collection=collection,
        question=request.question,
        top_k=request.top_k
    )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chroma query error: {e}")

    docs = chroma_result["documents"]
    metas = chroma_result["metadatas"]

    # 2. Call LLM z kontekstem
    try:
        answer = call_llm_with_rag(openai_client, request.question, docs, metas)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    return ChatResponse(
        answer=answer,
        context_documents=docs,
    )
