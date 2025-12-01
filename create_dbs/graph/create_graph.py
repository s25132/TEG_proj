from py2neo import Graph, Node, Relationship
from py2neo.errors import ServiceUnavailable, ConnectionUnavailable
from dotenv import load_dotenv
from pypdf import PdfReader
import os
import time

load_dotenv(override=True)

URI =  os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER")
PASSWORD = os.getenv("NEO4J_PASSWORD")
PROJECTS_FILE = os.getenv("PROJECTS_FILE")
PROGRAMMERS_DIR = os.getenv("PROGRAMMERS_DIR")

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

def wait_for_neo4j(uri, user, password, timeout=120, interval=1) -> Graph:
    start = time.time()
    print(f"Czekam na Neo4j pod {uri}...")

    while True:
        try:
            graph = Graph(uri, auth=(user, password))
            graph.run("RETURN 1").evaluate()

            print("Neo4j jest gotowy, lecimy dalej.")
            return graph

        except (ServiceUnavailable, ConnectionUnavailable, Exception) as e:
            elapsed = time.time() - start

            if elapsed >= timeout:
                raise RuntimeError(f"Neo4j nie wstał w ciągu {timeout} sekund!") from e

            print(f"Neo4j jeszcze niegotowy ({e}), czekam {interval}s...")
            time.sleep(interval)


def create_example_graph(graph: Graph):
    # wyczyść bazę
    graph.run("MATCH (n) DETACH DELETE n")

    # utwórz węzły
    alice = Node("Person", name="Alice", age=30)
    bob = Node("Person", name="Bob", age=25)
    charlie = Node("Person", name="Charlie", age=35)

    # dodaj węzły do grafu
    graph.create(alice)
    graph.create(bob)
    graph.create(charlie)

    # utwórz relacje
    graph.create(Relationship(alice, "FRIEND", bob))
    graph.create(Relationship(bob, "FRIEND", charlie))
    graph.create(Relationship(alice, "FRIEND", charlie))

    print("Graph created:")
    print(alice)
    print(bob)
    print(charlie)

    count = graph.run("MATCH (n:Person) RETURN count(n)").evaluate()
    print("Persons:", count)


def main():
    # Neo4j Community Edition nie obsługuje wielu baz danych -> działam na domyślnej bazie neo4j.
    graph = wait_for_neo4j(URI, USER, PASSWORD)
    create_example_graph(graph)


if __name__ == "__main__":
    main()
