import time
import math
from pypdf import PdfReader
from langchain_community.graphs import Neo4jGraph
from io import BytesIO

def extract_text_from_pdf_bytes(pdf_bytes: bytes, doc_type: str) -> str:
    reader = PdfReader(BytesIO(pdf_bytes))
    texts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        texts.append(page_text)
        
    texts.append(
        "\n\n[METADATA]\n"
        f"document_type: {doc_type}\n"
    )
    return "\n".join(texts).strip()


def wait_for_neo4j(uri: str, user: str, password: str,
                   timeout: int = 120, interval: int = 1) -> Neo4jGraph:
    """Czeka aż Neo4j będzie gotowy, zwraca obiekt Neo4jGraph."""
    start = time.time()
    print(f"Czekam na Neo4j pod {uri}...")

    while True:
        try:
            graph = Neo4jGraph(
                url=uri,
                username=user,
                password=password
            )

            # prosty test połączenia
            graph.query("RETURN 1 AS ok")

            print("Neo4j jest gotowy, lecimy dalej.")
            return graph

        except Exception as e:
            elapsed = time.time() - start

            if elapsed >= timeout:
                raise RuntimeError(
                    f"Neo4j nie wstał w ciągu {timeout} sekund!"
                ) from e

            print(f"Neo4j jeszcze niegotowy ({e}), czekam {interval}s...")
            time.sleep(interval)