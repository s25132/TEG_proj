from typing import List

from openai import OpenAI
from chromadb.api.models.Collection import Collection

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