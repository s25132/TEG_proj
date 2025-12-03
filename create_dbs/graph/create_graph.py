import json
from typing import List
from glob import glob
from pathlib import Path
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate,SystemMessagePromptTemplate, HumanMessagePromptTemplate
from dotenv import load_dotenv
from pypdf import PdfReader
import os
import time

load_dotenv(override=True)

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER")
PASSWORD = os.getenv("NEO4J_PASSWORD")
PROJECTS_FILE = os.getenv("PROJECTS_FILE")
PROGRAMMERS_DIR = os.getenv("PROGRAMMERS_DIR")

def project_to_text(p: dict, document_type: str) -> str:
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

    text = (
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

    # Bardzo jednoznaczny blok metadanych:
    text += (
        "\n\n[METADATA]\n"
        f"document_type: {document_type}\n"
    )

    return text


def extract_text_from_pdf(path: str, document_type: str) -> str:
    """Czyta cały tekst z PDF-a (wszystkie strony)."""
    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        texts.append(page_text)
    
    # Dodajemy metadane na końcu dokumentu
    texts.append(
        "\n\n[METADATA]\n"
        f"document_type: {document_type}\n"
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

def get_llm_transformer() -> LLMGraphTransformer:
    """Setup LLM and graph transformer with CV-specific schema."""
    # Initialize LLM - using GPT-4o-mini for cost efficiency
    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    talent_allowed_nodes = [
            "Person",       # programista
            "Company",      # klient / pracodawca
            "University",
            "Skill",        # umiejętność (np. Java, React)
            "Technology",   # jeśli chcesz odróżniać od Skill
            "Project",      # zarówno realny projekt, jak i RFP (odróżniane po 'status' / 'source')
            "Certification",
            "Location",
            "JobTitle",
            "Industry",
    ]

        # 2. Dozwolone relacje (z kierunkiem)
    talent_allowed_relationships = [
            # --- CV (programmer_profiles.json) ---
            ("Person", "WORKED_AT", "Company"),
            ("Person", "STUDIED_AT", "University"),
            ("Person", "HAS_SKILL", "Skill"),
            ("Person", "LOCATED_IN", "Location"),
            ("Person", "HOLDS_POSITION", "JobTitle"),
            ("Person", "WORKED_ON", "Project"),
            ("Person", "EARNED", "Certification"),
            ("Person", "ASSIGNED_TO", "Project"),

            ("JobTitle", "AT_COMPANY", "Company"),
            ("University", "LOCATED_IN", "Location"),

            # --- Projekty (projects.json) ---
            ("Project", "USED_TECHNOLOGY", "Technology"),
            ("Project", "FOR_COMPANY", "Company"),
            ("Company", "IN_INDUSTRY", "Industry"),
            ("Skill", "RELATED_TO", "Technology"),
            ("Certification", "ISSUED_BY", "Company"),

            # --- Rozszerzenie pod projects.json + rfps.json ---
            # wymagane skille w projekcie / RFP
            ("Project", "REQUIRES_SKILL", "Skill"),
            # preferowane certyfikaty
            ("Project", "PREFERS_CERTIFICATION", "Certification"),
            # lokalizacja projektu / RFP
            ("Project", "LOCATED_IN", "Location"),
        ]

        #3. Jawnie zdefiniowane właściwości węzłów
    talent_node_properties = [
            # --- ogólne / techniczne ---
            "name",         # krótka nazwa (np. "Core Platform Revamp", "Java")
            "title",        # dłuższy tytuł (np. RFP title)
            "description",  # opis projektu / RFP
            "source",       # z jakiego JSON-a pochodzi: "profiles" / "projects" / "rfps"
            "document_type", # typ dokumentu: "cv" / "project" / "rfp"

            # --- czasowe ---
            "start_date",       # np. okres zatrudnienia / start projektu / start assignmentu
            "end_date",
            "duration_months",  # z rfps.json
            "years_experience", # dla umiejętności / osoby

            # --- poziomy umiejętności / wymagania ---
            "level",            # np. "junior", "senior"
            "proficiency",      # poziom u programisty (np. "Intermediate", "Expert")
            "min_proficiency",  # minimalny poziom w RFP / projekcie
            "is_mandatory",     # czy skill jest wymagany (True/False)

            # --- projekty / RFP ---
            "project_type",     # z rfps.json
            "team_size",        # z rfps.json lub projects.json
            "budget_min",       # np. z przetworzonego budget_range
            "budget_max",
            "status",           # np. "RFP", "ongoing", "completed"
            "remote_allowed",   # bool z rfps.json

            # --- Person ---
            "email",            # z programmer_profiles.json
        ]
    

    SYSTEM_PROMPT = """
        # Knowledge Graph Instructions

        You extract a knowledge graph from the input text.

        - Nodes represent entities like Person, Company, Project, Skill, etc.
        - Edges represent relationships between these entities.
        - You MUST also extract node properties whenever they are explicitly present in the text.

        ## METADATA block

        The input text may contain a block:

        [METADATA]
        document_type: <VALUE>

        If such a block is present, you MUST:
        - Set the node property `document_type` on the main document node (e.g. Project, CV, RFP) to that VALUE.
        - Never ignore this field when it appears.

        Do not hallucinate values that are not explicitly given.
        """

    FINAL_TIP = HumanMessagePromptTemplate(
        prompt=PromptTemplate.from_template(
            "Tip: Make sure to answer in the correct format and do not include any "
            "explanations. Use the given format to extract information from the "
            "following input: {input}"
        )
    )

    chat_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
            FINAL_TIP,
        ]
    )

        # Initialize transformer with strict schema
    llm_transformer = LLMGraphTransformer(
        llm=model,
        allowed_nodes=talent_allowed_nodes,
        allowed_relationships=talent_allowed_relationships,
        node_properties=talent_node_properties,
        strict_mode=True,
        prompt=chat_prompt,
    )

    return llm_transformer

def convert_cv_to_graph(llm_transformer: LLMGraphTransformer, pdf_path: str) -> List:
        """Convert a single CV PDF to graph documents.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List: Graph documents extracted from the CV
        """
        print(f"Processing: {Path(pdf_path).name}")

        # Extract text from PDF
        text_content = extract_text_from_pdf(pdf_path, document_type="cv")

        if not text_content.strip():
            print(f"No text extracted from {pdf_path}")
            return []

        # Create Document object
        document = Document(
            page_content=text_content,
            metadata={"source": pdf_path, "type": "cv"}
        )

        # Convert to graph documents using LLM
        try:
            graph_documents =  llm_transformer.convert_to_graph_documents([document])
            print(f"✓ Extracted graph from {Path(pdf_path).name}")

            # Log extraction statistics
            if graph_documents:
                nodes_count = len(graph_documents[0].nodes)
                relationships_count = len(graph_documents[0].relationships)
                print(f"  - Nodes: {nodes_count}, Relationships: {relationships_count}")

            return graph_documents
        except Exception as e:
            print(f"Error converting {pdf_path} to graph: {e}")
            return []
        
def store_graph_documents(graph_documents: List, graph: Neo4jGraph):
        """Store graph documents in Neo4j.

        Args:
            graph_documents: List of GraphDocument objects
        """
        try:
            # Add graph documents to Neo4j with enhanced options
            graph.add_graph_documents(
                graph_documents,
                baseEntityLabel=True,  # Add base Entity label for indexing
                include_source=True    # Include source documents for RAG
            )

            # Calculate and log statistics
            total_nodes = sum(len(doc.nodes) for doc in graph_documents)
            total_relationships = sum(len(doc.relationships) for doc in graph_documents)

            print(f"✓ Stored {len(graph_documents)} documents in Neo4j")
            print(f"✓ Total nodes: {total_nodes}")
            print(f"✓ Total relationships: {total_relationships}")

        except Exception as e:
            print(f"Failed to store graph documents: {e}")
            raise

def process_all_cvs(llm_transformer: LLMGraphTransformer, graph: Neo4jGraph, cv_directory: str) -> int:
        """Process all PDF CVs in the directory.

        Args:
            cv_directory: Directory containing PDF CVs

        Returns:
            int: Number of successfully processed CVs
        """
        # Find all PDF files
        pdf_pattern = os.path.join(cv_directory, "*.pdf")
        pdf_files = glob(pdf_pattern)

        if not pdf_files:
            print(f"No PDF files found in {Path(cv_directory).resolve()}")
            return 0

        print(f"Found {len(pdf_files)} PDF files to process")

        processed_count = 0
        all_graph_documents = []

        # Process each CV
        for pdf_path in pdf_files:
            graph_documents = convert_cv_to_graph(llm_transformer, pdf_path)

            if graph_documents:
                all_graph_documents.extend(graph_documents)
                processed_count += 1
            else:
                print(f"Failed to process {pdf_path}")

        # Store all graph documents in Neo4j
        if all_graph_documents:
            print("Storing graph documents in Neo4j...")
            store_graph_documents(all_graph_documents, graph)

        return processed_count


def process_all_projects(llm_transformer: LLMGraphTransformer, graph: Neo4jGraph, projects_file: str):
    with open(projects_file, "r", encoding="utf-8") as f:
        projects = json.load(f)

    processed_count = 0
    all_graph_documents = []

    for p in projects:
        text_content = project_to_text(p, document_type="project")

        if not text_content.strip():
            print(f"No text extracted from {projects_file}")
            return []

        # Create Document object
        document = Document(
            page_content=text_content,
            metadata={"source": projects_file, "type": "project"}
        )

        # Convert to graph documents using LLM
        graph_documents =  llm_transformer.convert_to_graph_documents([document])
        print(f"✓ Extracted graph from {Path(projects_file).name}")

        # Log extraction statistics
        if graph_documents:
            nodes_count = len(graph_documents[0].nodes)
            relationships_count = len(graph_documents[0].relationships)
            print(f"  - Nodes: {nodes_count}, Relationships: {relationships_count}")

        if graph_documents:
            all_graph_documents.extend(graph_documents)
            processed_count += 1
        else:
            print(f"Failed to process {projects_file}")

    # Store all graph documents in Neo4j
    if all_graph_documents:
        print("Storing graph documents in Neo4j...")
        store_graph_documents(all_graph_documents, graph)

    return processed_count



def main():
    # Neo4j Community Edition nie obsługuje wielu baz danych -> działam na domyślnej bazie neo4j.
    graph = wait_for_neo4j(URI, USER, PASSWORD)
    # wyczyść bazę
    graph.query("MATCH (n) DETACH DELETE n")
    print("Baza danych wyczyszczona.")

    llm_transformer = get_llm_transformer()

    processed_cvs = process_all_cvs(llm_transformer, graph, PROGRAMMERS_DIR)
    print(f"Przetworzono łącznie {processed_cvs} CV.")

    processed_projects = process_all_projects(llm_transformer, graph, PROJECTS_FILE)
    print(f"Przetworzono łącznie {processed_projects} projekty.")


if __name__ == "__main__":
    main()
