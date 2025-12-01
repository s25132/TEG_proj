
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from pathlib import Path
from typing import Optional
from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument
from langsmith.run_helpers import traceable
from app.utility import extract_text_from_pdf_bytes

def get_llm_transformer(model: ChatOpenAI) -> LLMGraphTransformer:
    """Setup LLM and graph transformer with CV-specific schema."""


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

        # Initialize transformer with strict schema
    llm_transformer = LLMGraphTransformer(
        llm=model,
        allowed_nodes=talent_allowed_nodes,
        allowed_relationships=talent_allowed_relationships,
        node_properties=talent_node_properties,
        strict_mode=True
    )

    return llm_transformer


def store_single_graph_document(graph_document, graph: Neo4jGraph):
    """Store exactly one graph document in Neo4j.

    Args:
        graph_document: a single GraphDocument object
        graph: Neo4jGraph connection
    """
    try:
        # Add only one document
        graph.add_graph_documents(
            [graph_document],        # wrap in list but save only one
            baseEntityLabel=True,
            include_source=True
        )

        # Stats for the single document
        total_nodes = len(graph_document.nodes)
        total_relationships = len(graph_document.relationships)

        print("✓ Stored 1 document in Neo4j")
        print(f"✓ Nodes: {total_nodes}")
        print(f"✓ Relationships: {total_relationships}")

    except Exception as e:
        print(f"Failed to store graph document: {e}")
        raise

@traceable
def convert_to_graph(
    llm_transformer: LLMGraphTransformer,
    pdf_bytes: bytes,
    filename: str,
) -> Optional[GraphDocument]:
    """Convert a single CV PDF to a single GraphDocument.

    Args:
        llm_transformer: initialized LLMGraphTransformer
        filename: name of the PDF file
        pdf_bytes: bytes of the PDF file

    Returns:
        Optional[GraphDocument]: single GraphDocument extracted from the CV,
        or None if extraction failed.
    """
    print(f"Processing: {Path(filename).name}")

    # Extract text from PDF
    text_content = extract_text_from_pdf_bytes(pdf_bytes)

    if not text_content.strip():
        print(f"No text extracted from {filename}")
        return None

    # Create Document object
    document = Document(
        page_content=text_content,
        metadata={"source": filename, "type": "rfp"}
    )

    # Convert to graph documents using LLM
    try:
        graph_documents = llm_transformer.convert_to_graph_documents([document])
        print(f"✓ Extracted graph from {filename}")

        if not graph_documents:
            print("No graph documents returned by LLMGraphTransformer")
            return None

        graph_document = graph_documents[0]

        # Log extraction statistics for the single document
        nodes_count = len(graph_document.nodes)
        relationships_count = len(graph_document.relationships)
        print(f"  - Nodes: {nodes_count}, Relationships: {relationships_count}")

        return graph_document

    except Exception as e:
        print(f"Error converting {filename} to graph: {e}")
        return None
