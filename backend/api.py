import os

import chromadb
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from chromadb import PersistentClient
from fastapi import FastAPI, HTTPException, UploadFile, File
from dotenv import load_dotenv
from schemas import ChatRequest, ChatResponse
from chroma import build_context_from_chroma, call_llm_with_rag, load_rfps_into_collection

load_dotenv(override=True)

# ---------- Konfiguracja ----------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Brak OPENAI_API_KEY w zmiennych środowiskowych / .env")

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "projects_and_cvs")

# ---------- Inicjalizacja klientów ----------
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.2)
emb_model = OpenAIEmbeddings(model="text-embedding-3-small")

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
        emb_model=emb_model,
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
        answer = call_llm_with_rag(llm, request.question, docs, metas)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    return ChatResponse(
        answer=answer,
        context_documents=docs,
    )

@app.post("/add_rfp")
async def add_rfp(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Plik musi być PDF.")
    
    result: str = "OK"

    try:
        content = await file.read()

        # tu indeksujemy RFP w Chroma
        result = load_rfps_into_collection(
            pdf_bytes=content,
            filename=file.filename,
            collection=collection,
            emb_model=emb_model,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Nie udało się przetworzyć pliku: {e}")

    return {"status": result, "filename": file.filename}
