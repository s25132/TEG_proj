
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from chromadb.api.models.Collection import Collection
from uuid import uuid4
from langsmith.run_helpers import traceable
from app.utilities.utility import extract_text_from_pdf_bytes
from typing import Dict, List


CHAT_MEMORY: Dict[str, List] = {}

def get_history(session_id: str) -> List:
    return CHAT_MEMORY.get(session_id, [])

def save_turn(session_id: str, user_text: str, ai_text: str):
    hist = CHAT_MEMORY.setdefault(session_id, [])
    hist.append(HumanMessage(content=user_text))
    hist.append(AIMessage(content=ai_text))

@traceable
def call_llm_with_rag_and_memory(
    llm: ChatOpenAI,
    question: str,
    context_docs: List[str],
    metadatas: List[dict],
    session_id: str,
    max_history_turns: int = 20,   # ile ostatnich tur trzymać
) -> str:

    # 1) historia (ograniczenie do ostatnich tur)
    history = get_history(session_id)
    if max_history_turns is not None:
        # 1 tura = Human + AI, więc 2 * max_history_turns wiadomości
        history = history[-2 * max_history_turns:]

    # 2) kontekst RAG
    if not context_docs:
        context_text = "Brak dopasowanego kontekstu w bazie (Chroma)."
    else:
        chunks = []
        for doc, meta in zip(context_docs, metadatas):
            kind = meta.get("kind", "unknown")
            name_or_file = meta.get("name") or meta.get("filename") or ""
            label = f"[{kind} {name_or_file}]".strip()
            chunks.append(f"{label}\n{doc}")
        context_text = "\n\n---\n\n".join(chunks)

    system_msg = SystemMessage(content=(
        "You are an assistant that answers questions using ONLY the provided context. "
        "If the answer is not present in the context, say you don't know."
    ))

    # 3) wiadomość usera + kontekst (RAG)
    user_msg = HumanMessage(content=(
        f"User question:\n{question}\n\n"
        f"Context documents:\n{context_text}"
    ))

    # 4) wywołanie z historią
    messages = [system_msg, *history, user_msg]
    response = llm.invoke(messages)

    # 5) zapis do pamięci (zapisuj „czyste” pytanie usera, nie cały RAG)
    save_turn(session_id, question, response.content)

    return response.content


def build_context_from_chroma(emb_model: OpenAIEmbeddings, collection: Collection, question: str, top_k: int):

    emb = emb_model.embed_query(question)

    results = collection.query(
        query_embeddings=[emb],
        n_results=top_k
    )

    docs = results.get("documents", [[]])
    metas = results.get("metadatas", [[]])

    return {
        "documents": docs[0] if docs else [],
        "metadatas": metas[0] if metas else []
    }


def load_rfps_into_collection(
    pdf_bytes: bytes,
    filename: str,
    collection: Collection,
    emb_model: OpenAIEmbeddings
):
    
    existing = collection.get(
        where={"filename": filename},
        limit=1
    )

    if existing and existing.get("ids"):
        # plik już istnieje w kolekcji
        print(f"[RFP] Plik {filename} już istnieje w kolekcji, pomijam.")
        return

    """Ekstrahuje tekst z PDF i zapisuje jako RFP w kolekcji Chroma."""
    full_text = extract_text_from_pdf_bytes(pdf_bytes, doc_type="RFP")

    if not full_text:
        print(f"[RFP] Brak tekstu w pliku: {filename}, pomijam.")
        return

    # unikalne ID dla RFP
    rfp_id = f"rfp_{uuid4()}"

    emb = emb_model.embed_query(full_text)

    # UWAGA: wszystko musi być listą
    collection.add(
        ids=[rfp_id],
        documents=[full_text],
        embeddings=[emb],
        metadatas=[{
            "kind": "rfp",
            "filename": filename,
        }],
    )

    print(f"[RFP] Załadowano 1 RFP do kolekcji z pliku: {filename}")