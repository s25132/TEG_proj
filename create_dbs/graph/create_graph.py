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
SCHEMA_DIR = os.getenv("SCHEMA_DIR")

def project_to_text(p: dict, document_type: str) -> str:
    """Zamienia cały obiekt projektu na jeden opisowy tekst."""

    reqs = ", ".join([
        f"{r.get('skill_name')} (minimum_level: {r.get('min_proficiency')})"
        for r in p.get("requirements", [])
    ])

    programmers = ", ".join([
        f"{a.get('programmer_name')} (ID {a.get('programmer_id')}, "
        f"start_date: {a.get('assignment_start_date')}, "
        f"end_date: {a.get('assignment_end_date')}"
        + (f", allocation_percentage: {a.get('allocation_percentage')}"
        if a.get("allocation_percentage") is not None else "")
        + ")"
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
        api_key=os.getenv("OPENAI_API_KEY")
        )
    
    # 1. Dozwolone węzły
    with open(os.path.join(SCHEMA_DIR, "allowed_nodes.json"), "r", encoding="utf-8") as f:
        talent_allowed_nodes = json.load(f)

    # 2. Dozwolone relacje (z kierunkiem)    
    with open(os.path.join(SCHEMA_DIR, "allowed_relationships.json"), "r", encoding="utf-8") as f:
        talent_allowed_relationships = [
            tuple(r) for r in json.load(f)
        ]

    # 3. Właściwości węzłów
    with open(os.path.join(SCHEMA_DIR, "allowed_node_properties.json"), "r", encoding="utf-8") as f:
        talent_node_properties = json.load(f)
    

    SYSTEM_PROMPT = r"""
# Knowledge Graph Extraction Instructions (STRICT SCHEMA)

You extract a knowledge graph from the input text.
The output MUST strictly follow the schema defined below.
DO NOT create any nodes, relationships, or properties outside this schema.

---

## ALLOWED NODE TYPES AND PROPERTIES (CORE ENTITIES)

### Person
(Person {{
  id,
  name,
  location,
  email,
  phone,
  years_experience
}})

### Skill
(Skill {{
  id,
  category,
  subcategory
}})

### Company
(Company {{
  id,
  name,
  industry,
  size,
  location
}})

### Project
(Project {{
  id,
  title,
  description,
  start_date,
  end_date,
  budget,
  status,
  document_type
}})

### Certification
(Certification {{
  id,
  name,
  provider,
  date_earned,
  expiry_date
}})

### University
(University {{
  id,
  name,
  location,
  ranking
}})

### RFP
(RFP {{
  id,
  title,
  description,
  requirements,
  budget,
  deadline,
  document_type
}})

---

## ALLOWED RELATIONSHIP TYPES AND PROPERTIES

### Person → Skill
(Person)-[HAS_SKILL {{
  proficiency,        // integer 1–5
  years_experience
}}]->(Skill)

### Person → Company
(Person)-[WORKED_AT {{
  role,
  start_date,
  end_date
}}]->(Company)

### Person → Project
(Person)-[WORKED_ON {{
  role,
  contribution,
  start_date,
  end_date
}}]->(Project)

(Person)-[ASSIGNED_TO {{
  allocation_percentage,
  start_date,
  end_date
}}]->(Project)

### Person → Certification
(Person)-[EARNED {{
  date,
  score
}}]->(Certification)

### Person → University
(Person)-[STUDIED_AT {{
  degree,
  graduation_year,
  gpa
}}]->(University)

### Project → Skill
(Project)-[REQUIRES {{
  minimum_level,
  preferred_level
}}]->(Skill)

### RFP → Skill
(RFP)-[NEEDS {{
  required_count,
  experience_level
}}]->(Skill)

---

## RELATIONSHIP PROPERTY EXTRACTION RULES (MANDATORY)

When you create any relationship above:

1) You MUST extract relationship properties ONLY when values are explicitly present in the text.
2) If a relationship exists but no valid property values are explicitly present, create the relationship WITHOUT properties.
3) NEVER infer, guess, or fabricate values.

### HAS_SKILL
Extract:
- proficiency (int 1–5) if explicitly present as:
  - "proficiency: <1-5>"
  - OR "level: <Beginner|Intermediate|Advanced|Expert>" using this mapping ONLY when the level word is present:
    - Beginner -> 2
    - Intermediate -> 3
    - Advanced -> 4
    - Expert -> 5
- years_experience if explicitly present as a number of years (e.g., "5 years", "5 yrs", "5 years experience")

### WORKED_AT / WORKED_ON / ASSIGNED_TO
Extract:
- role only if explicitly present as a title
- start_date / end_date only if explicitly present (year or date)
- allocation_percentage only if explicitly present as a number (e.g., "allocation_percentage: 50" or "50%")

### STUDIED_AT
Extract:
- degree only if explicitly present
- graduation_year only if explicitly present as a year
- gpa only if explicitly present as a numeric GPA

### EARNED
Extract:
- date only if explicitly present
- score only if explicitly present

### REQUIRES / NEEDS
Extract:
- minimum_level / preferred_level / experience_level only if explicitly present with the SAME field name
- required_count only if explicitly present as a number

---

## METADATA BLOCK (MANDATORY HANDLING)

The input text may contain a block:

[METADATA]
document_type: <VALUE>

If present, you MUST:
- Assign `document_type` to the main node:
  - Project -> document_type
  - RFP -> document_type
- NEVER ignore this field
- DO NOT infer or hallucinate document_type if missing

---

## STATUS FIELD (MANDATORY HANDLING)

If the input text contains:

Status: <VALUE>

You MUST:
- Assign `status` to Project OR RFP
- Use the EXACT extracted value
- NEVER omit `status` if present
- DO NOT infer status if it is not explicitly provided

---

## GENERAL EXTRACTION RULES

- Extract ONLY explicitly stated information
- DO NOT guess missing values
- DO NOT normalize or reinterpret values (EXCEPT the explicit HAS_SKILL level mapping above)
- DO NOT create placeholder or default values
- If a property is not present in the text, omit it
- IDs must be stable and unique per entity (use deterministic IDs if possible)
- Use consistent entity resolution (same entity -> same node)

---

## OUTPUT RULES (CRITICAL)

- Output ONLY the extracted graph data
- DO NOT include explanations, comments, or reasoning
- DO NOT include text outside the graph structure
- Format the output as structured graph documents
- Ensure strict adherence to the defined schema
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
        relationship_properties=True, 
        strict_mode=False,
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
                include_source=False    # Include source documents for RAG
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
        for pdf_path in pdf_files[:10]:
            graph_documents = convert_cv_to_graph(llm_transformer, pdf_path)

            if graph_documents:
                
                for rel in graph_documents[0].relationships:
                    if rel.type in ["HAS_SKILL","ASSIGNED_TO","REQUIRES","WORKED_AT","WORKED_ON"]:
                        print("REL:", rel.type, "PROPS:", rel.properties)

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
            for rel in graph_documents[0].relationships:
                if rel.type in ["HAS_SKILL","ASSIGNED_TO","REQUIRES","WORKED_AT","WORKED_ON"]:
                    print("REL:", rel.type, "PROPS:", rel.properties)

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
