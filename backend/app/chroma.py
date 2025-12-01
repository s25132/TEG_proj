
from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from chromadb.api.models.Collection import Collection
from uuid import uuid4
from langsmith.run_helpers import traceable
from app.utility import extract_text_from_pdf_bytes


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

@traceable
def call_llm_with_rag(llm: ChatOpenAI, question: str, context_docs: List[str], metadatas: List[dict]) -> str:

    # Budowanie kontekstu – identycznie jak u Ciebie
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

    system_msg = (
        "You are an assistant that answers questions using ONLY the provided context. "
        "The context comes from three types of documents: projects, programmers' CVs, and RFPs. "
        "If you are not sure or the answer is not present in the context, say that "
        "you don't know and do not hallucinate."
    )

    user_msg = (
        f"User question:\n{question}\n\n"
        f"Context documents:\n{context_text}"
    )

    response = llm.invoke([
        SystemMessage(content=system_msg),
        HumanMessage(content=user_msg),
    ])

    return response.content

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
    full_text = extract_text_from_pdf_bytes(pdf_bytes)

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