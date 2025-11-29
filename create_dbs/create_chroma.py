import json
import glob
import os
import chromadb
from pypdf import PdfReader

from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv(override=True)


def project_to_text(p: dict) -> str:
    """Zamienia cały obiekt projektu na jeden opisowy tekst."""
    reqs = ", ".join([
        f"{r.get('skill_name')} (proficiency: {r.get('min_proficiency')}, "
        f"mandatory: {r.get('is_mandatory')})"
        for r in p.get("requirements", [])
    ])

    programmers = ", ".join([
        f"{a.get('programmer_name')} (ID {a.get('programmer_id')}, "
        f"from {a.get('assignment_start_date')} to {a.get('assignment_end_date')})"
        for a in p.get("assigned_programmers", [])
    ])

    return (
        f"Project ID: {p.get('id')}. "
        f"Name: {p.get('name')}. "
        f"Client: {p.get('client')}. "
        f"Description: {p.get('description')}. "
        f"Start date: {p.get('start_date')}. "
        f"End date: {p.get('end_date')}. "
        f"Estimated duration (months): {p.get('estimated_duration_months')}. "
        f"Budget: {p.get('budget')}. "
        f"Status: {p.get('status')}. "
        f"Team size: {p.get('team_size')}. "
        f"Requirements: {reqs}. "
        f"Assigned programmers: {programmers}."
    )


def extract_text_from_pdf(path: str) -> str:
    """Czyta cały tekst z PDF-a (wszystkie strony)."""
    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        texts.append(page_text)
    return "\n".join(texts).strip()


def load_projects_into_collection(projects_file: str, collection, emb_model):
    with open(projects_file, "r", encoding="utf-8") as f:
        projects = json.load(f)

    ids = []
    documents = []
    embeddings = []
    metadatas = []

    for p in projects:
        pid = p["id"]
        text = project_to_text(p)

        ids.append(f"project_{pid}")  # żeby nie kolidowało z cv_id
        documents.append(text)
        metadatas.append({
            "kind": "project",
            "project_id": pid,
            "name": p.get("name"),
            "client": p.get("client"),
            "status": p.get("status"),
        })

        emb = emb_model.embed_query(text)
        embeddings.append(emb)

    if ids:
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    print(f"Załadowano {len(ids)} projektów do kolekcji.")


def load_cvs_into_collection(cv_dir: str, collection, emb_model):
    pdf_paths = glob.glob(os.path.join(cv_dir, "*.pdf"))
    if not pdf_paths:
        print(f"Brak plików PDF w katalogu: {cv_dir}")
        return

    ids = []
    documents = []
    embeddings = []
    metadatas = []

    for path in pdf_paths:
        filename = os.path.basename(path)
        base_id = os.path.splitext(filename)[0]   # np. 'jan_kowalski_cv'

        text = extract_text_from_pdf(path)
        if not text:
            print(f"Uwaga: brak tekstu w PDF: {filename}, pomijam.")
            continue

        full_text = f"Curriculum Vitae file: {filename}.\nContent:\n{text}"

        ids.append(f"cv_{base_id}")
        documents.append(full_text)
        metadatas.append({
            "kind": "cv",
            "filename": filename,
        })

        emb = emb_model.embed_query(full_text)

        embeddings.append(emb)

    if ids:
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    print(f"Załadowano {len(ids)} CV do kolekcji z katalogu: {cv_dir}")


def main():
    
    # 1. Klient OpenAI (wymaga OPENAI_API_KEY)
    
    emb_model = OpenAIEmbeddings(model="text-embedding-3-small")

    chroma_dir = os.getenv("CHROMA_DIR")

    chroma_client = chromadb.PersistentClient(path=chroma_dir)

    # Usuń kolekcję jeśli istnieje
    collections = chroma_client.list_collections()

    collection_name = os.getenv("CHROMA_COLLECTION_NAME")

    if any(c.name == collection_name for c in collections):
        chroma_client.delete_collection(collection_name)
        print(f"Usunięto starą kolekcję: {collection_name}")

    # 3. Jedna wspólna kolekcja
    collection = chroma_client.get_or_create_collection(collection_name)

    projects_file = os.getenv("PROJECTS_FILE")
    # 4. Projekty
    if os.path.exists(projects_file):
        load_projects_into_collection(projects_file, collection, emb_model)
    else:
        print(f"Uwaga: nie znaleziono pliku {projects_file}, pomijam ładowanie projektów.")

    cv_dir = os.getenv("PROGRAMMERS_DIR")
    # 5. CV z PDF
    if os.path.isdir(cv_dir):
        load_cvs_into_collection(cv_dir, collection, emb_model)
    else:
        print(f"Uwaga: katalog {cv_dir} nie istnieje, pomijam ładowanie CV.")

    print(f"Baza Chroma zapisana w katalogu: {chroma_dir}")
    print(f"Kolekcja: {collection_name}")

if __name__ == "__main__":
    main()