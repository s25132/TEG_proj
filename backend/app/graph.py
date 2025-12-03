
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate,SystemMessagePromptTemplate, HumanMessagePromptTemplate
from pathlib import Path
from typing import Any, Optional, Dict
from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument
from langchain_core.prompts.prompt import PromptTemplate
from langsmith.run_helpers import traceable
from app.utility import extract_text_from_pdf_bytes


def setup_qa_chain(model: ChatOpenAI,  graph: Neo4jGraph) -> GraphCypherQAChain:
        """Setup the GraphCypherQA chain."""

        # Custom Cypher generation prompt with case-insensitive matching
        CYPHER_GENERATION_TEMPLATE = """Task: Generate Cypher statement to query a graph database.
        Instructions:
        Use only the provided relationship types and properties in the schema.
        Do not use any other relationship types or properties that are not provided.
        For skill matching, always use case-insensitive comparison using toLower() function.
        For count queries, ensure you return meaningful column names.

        Schema:
        {schema}

        Note: Do not include any explanations or apologies in your responses.
        Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
        Do not include any text except the generated Cypher statement.

        Examples: Here are a few examples of generated Cypher statements for particular questions:

        # How many Python programmers do we have?
        MATCH (p:Person)-[:HAS_SKILL]->(s:Skill)
        WHERE toLower(s.id) = toLower("Python")
        RETURN count(p) AS pythonProgrammers

        # Who has React skills?
        MATCH (p:Person)-[:HAS_SKILL]->(s:Skill)
        WHERE toLower(s.id) = toLower("React")
        RETURN p.id AS name

        # Find people with both Python and Django skills
        MATCH (p:Person)-[:HAS_SKILL]->(s1:Skill), (p)-[:HAS_SKILL]->(s2:Skill)
        WHERE toLower(s1.id) = toLower("Python") AND toLower(s2.id) = toLower("Django")
        RETURN p.id AS name

        # How many developers are AWS certified?
        MATCH (p:Person)-[:EARNED]->(c:Certification)
        WHERE toLower(c.name) CONTAINS toLower("aws")
        RETURN count(DISTINCT p) AS awsCertifiedDevelopers

        # List all senior developers with React and Python skills
        MATCH (p:Person)-[:HAS_SKILL]->(s1:Skill), (p)-[:HAS_SKILL]->(s2:Skill)
        WHERE toLower(s1.name) = toLower("React")
        AND toLower(s2.name) = toLower("Python")
        AND toLower(p.level) = toLower("senior")
        RETURN DISTINCT p

        # Find pairs of people who have worked together on completed projects
        # Find developers who worked together successfully
        MATCH (p1:Person)-[:WORKED_ON]->(pr:Project)<-[:WORKED_ON]-(p2:Person)
        WHERE p1 <> p2
        AND pr.status = "completed"
        RETURN DISTINCT p1, p2, pr

        # Identify skills that are a single point of failure (only one person has them)
        MATCH (p:Person)-[:HAS_SKILL]->(s:Skill)
        WITH s, collect(p) AS people
        WHERE size(people) = 1
        RETURN s.name AS skill, people[0] AS singlePointOfFailure

        # How many Python developers are available (not assigned to any project) in Q2 2025 in the baseline scenario?
        # How many Python developers are available in Q2 2025?
        WITH date("2025-04-01") AS q2Start, date("2025-06-30") AS q2End

        MATCH (p:Person)-[:HAS_SKILL]->(s:Skill)
        WHERE toLower(s.name) = toLower("Python")

        OPTIONAL MATCH (p)-[a:ASSIGNED_TO]->(pr:Project)
        WHERE a.start_date <= q2End
          AND a.end_date   >= q2Start
        WITH p, collect(a) AS assignmentsInQ2
        WHERE size(assignmentsInQ2) = 0

        RETURN count(DISTINCT p) AS availablePythonDevelopersQ2  

        # List pairs of developers who have worked together on the highest number of successful (completed) projects
        # Which pairs of developers most often worked together on successful projects?
        MATCH (p1:Person)-[:WORKED_ON]->(pr:Project)<-[:WORKED_ON]-(p2:Person)
        WHERE id(p1) < id(p2)             
        AND pr.status = "completed"
        WITH p1, p2, count(DISTINCT pr) AS successfulProjects
        RETURN p1.name AS developer1, p2.name AS developer2, successfulProjects
        ORDER BY successfulProjects DESC, developer1, developer2

        # Identify skills that are a single point of failure in mandatory project requirements
        # On which projects is there only one developer with a mandatory skill?
        MATCH (pr:Project)-[req:REQUIRES_SKILL]->(s:Skill)
        WHERE req.is_mandatory = true
        WITH pr, s

        MATCH (p:Person)-[:HAS_SKILL]->(s)
        MATCH (p)-[:WORKED_ON]->(pr)
        WITH pr, s, collect(DISTINCT p) AS people
        WHERE size(people) = 1

        RETURN pr.name  AS projectName,
            s.name   AS skill,
            people[0].name AS singlePointOfFailure
        ORDER BY projectName, skill



        The question is:
        {question}"""

        CYPHER_GENERATION_PROMPT = PromptTemplate(
            input_variables=["schema", "question"],
            template=CYPHER_GENERATION_TEMPLATE
        )

        # Custom QA prompt for better handling of numeric results
        CYPHER_QA_TEMPLATE = """You are an assistant that helps to form nice and human understandable answers.
        The information part contains the result(s) of a Cypher query that was executed against a knowledge graph.
        Information is provided as a list of records from the graph database.

        Guidelines :
        - If the information contains count results or numbers, state the exact count clearly.
        - For count queries that return 0, say "There are 0 [items]" - this is a valid result, not missing information.
        - If the information is empty or null, then say you don't know the answer.
        - Use the provided information to construct a helpful answer.
        - Be specific and mention actual names, numbers, or details from the information.

        Information:
        {context}

        Question: {question}
        Helpful Answer:"""

        CYPHER_QA_PROMPT = PromptTemplate(
            input_variables=["context", "question"],
            template=CYPHER_QA_TEMPLATE
        )

        # Create the GraphCypher QA chain with custom prompts
        qa_chain = GraphCypherQAChain.from_llm(
            llm=model,
            graph=graph,
            verbose=True,  # Show generated Cypher queries
            cypher_prompt=CYPHER_GENERATION_PROMPT,
            qa_prompt=CYPHER_QA_PROMPT,
            return_intermediate_steps=True,
            allow_dangerous_requests=True  # Allow DELETE operations for demo
        )

        return qa_chain

@traceable
def query_graph(chain: GraphCypherQAChain, question: str) -> Dict[str, Any]:
        """Execute a natural language query against the graph.

        Args:
            question: Natural language question

        Returns:
            Dict containing query results and metadata
        """
        try:
            print(f"Executing query: {question}")

            # Execute the query
            result = chain.invoke({"query": question})

            # Extract components
            response = {
                "question": question,
                "answer": result.get("result", "No answer generated"),
                "cypher_query": result.get("intermediate_steps", [{}])[0].get("query", ""),
                "success": True
            }

            print(f"✓ Query executed successfully")
            return response

        except Exception as e:
            print(f"Query failed: {e}")
            return {
                "question": question,
                "answer": f"Error: {str(e)}",
                "cypher_query": "",
                "success": False
            }

def get_llm_transformer(model: ChatOpenAI) -> LLMGraphTransformer:

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
    text_content = extract_text_from_pdf_bytes(pdf_bytes, doc_type="RFP")

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
