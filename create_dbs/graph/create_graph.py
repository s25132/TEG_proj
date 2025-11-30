from py2neo import Graph, Node, Relationship
from py2neo.errors import ServiceUnavailable, ConnectionUnavailable
import time
from dotenv import load_dotenv
import os


load_dotenv(override=True)

URI =  os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER")
PASSWORD = os.getenv("NEO4J_PASSWORD")


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
