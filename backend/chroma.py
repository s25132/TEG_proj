from typing import List

from openai import OpenAI
from chromadb.api.models.Collection import Collection
from pypdf import PdfReader
from io import BytesIO
from uuid import uuid4

def build_context_from_chroma( openai_client: OpenAI, collection: Collection, question: str, top_k: int):

    emb = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=question,
    ).data[0].embedding

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


def call_llm_with_rag( openai_client: OpenAI, question: str, context_docs: List[str], metadatas: List[dict]) -> str:
    """Buduje prompt z kontekstu i wywołuje LLM."""
    if not context_docs:
        # fallback gdy nic nie znaleziono w Chroma
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
        "The context comes from two types of documents: projects and programmers' CVs. "
        "If you are not sure or the answer is not present in the context, say that "
        "you don't know and do not hallucinate."
    )

    user_msg = (
        f"User question:\n{question}\n\n"
        f"Context documents:\n{context_text}"
    )

    completion = openai_client.chat.completions.create(
        model="gpt-4.1-mini",  # lub inny model, który masz dostępny
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )

    return completion.choices[0].message.content

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(pdf_bytes))
    texts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        texts.append(page_text)
    return "\n".join(texts).strip()


def load_rfps_into_collection(
    pdf_bytes: bytes,
    filename: str,
    collection: Collection,
    openai_client: OpenAI,
) -> None:
    """Ekstrahuje tekst z PDF i zapisuje jako RFP w kolekcji Chroma."""
    full_text = extract_text_from_pdf_bytes(pdf_bytes)

    if not full_text:
        print(f"[RFP] Brak tekstu w pliku: {filename}, pomijam.")
        return

    # unikalne ID dla RFP
    rfp_id = f"rfp_{uuid4()}"

    embedding = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=full_text,
    ).data[0].embedding

    # UWAGA: wszystko musi być listą
    collection.add(
        ids=[rfp_id],
        documents=[full_text],
        embeddings=[embedding],
        metadatas=[{
            "kind": "rfp",
            "filename": filename,
        }],
    )

    print(f"[RFP] Załadowano 1 RFP do kolekcji z pliku: {filename}")