import os

import chromadb
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from chromadb import PersistentClient
from fastapi import FastAPI, HTTPException, UploadFile, File
from dotenv import load_dotenv
from app.schemas import RagRequest, RagResponse , GraphRagRequest, GraphRagResponse
from app.chroma import build_context_from_chroma, call_llm_with_rag, load_rfps_into_collection
from app.utility import wait_for_neo4j
from app.graph import convert_to_graph, get_llm_transformer, store_single_graph_document, setup_qa_chain, query_graph

load_dotenv(override=True)

# ---------- Konfiguracja ----------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Brak OPENAI_API_KEY w zmiennych środowiskowych / .env")

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "projects_and_cvs")

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER")
PASSWORD = os.getenv("NEO4J_PASSWORD")

# ---------- Inicjalizacja klientów ----------
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.2)
emb_model = OpenAIEmbeddings(model="text-embedding-3-small")
graf = wait_for_neo4j(URI, USER, PASSWORD)
llm_transformer = get_llm_transformer(llm)
graph_cypher_qa_chain = setup_qa_chain(llm, graf)

# Klient Chroma (Persistent – dane na dysku)
chroma_client: PersistentClient = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION_NAME)

# ---------- FastAPI ----------

app = FastAPI(title="API", version="1.0.0")

@app.post("/ask_rag", response_model=RagResponse)
def chat_rag(request: RagRequest):
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

    return RagResponse(
        answer=answer,
        context_documents=docs,
    )


@app.post("/ask_graph", response_model=GraphRagResponse)
def chat_graph(request: GraphRagRequest):

    # tutaj wykonujemy zapytanie do grafu
    graph_response = query_graph(graph_cypher_qa_chain, request.question)

    return GraphRagResponse(
        answer=graph_response.get("answer", "No answer generated"),
    )


@app.post("/add_rfp")
async def add_rfp(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Plik musi być PDF.")

    try:
        content = await file.read()

        # tu indeksujemy RFP w Chroma
        load_rfps_into_collection(
            pdf_bytes=content,
            filename=file.filename,
            collection=collection,
            emb_model=emb_model,
        )
        # oraz konwertujemy do grafu i zapisujemy w Neo4j
        graph_document = convert_to_graph(llm_transformer, content, file.filename)
        if graph_document is not None:
            store_single_graph_document(graph_document, graf)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Nie udało się przetworzyć pliku: {e}")

    return {"status": "OK", "filename": file.filename}
