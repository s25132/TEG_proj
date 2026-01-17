import json
import os
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
from app.utilities.utility import extract_text_from_pdf_bytes
from langchain.agents import AgentExecutor

SCHEMA_DIR = os.getenv("SCHEMA_DIR")


def rfp_title_exists(graph: Neo4jGraph, title: str) -> bool:
    if not title or not str(title).strip():
        return False  # brak title -> nie blokujemy (albo możesz zwrócić True i zawsze skipować)
    
    print(f"Checking if RFP with title exists: {title}")

    q = """
    MATCH (r:Rfp)
    WHERE toLower(r.title) = toLower($title)
    RETURN count(r) > 0 AS exists
    """
    res = graph.query(q, {"title": title.strip()})
    print(f"RFP title exists result: {res}")
    return bool(res and res[0].get("exists"))


def get_rfp_title_from_graph_document(gd: GraphDocument) -> Optional[str]:
    for n in gd.nodes:
        print(f"Node type: {n.type}, properties: {n.properties}")
        if n.type == "Rfp":
            print(f"RFP Node properties: {n.properties}")
            props = n.properties or {}
            print(f"RFP Node props: {props}")
            title = props.get("title")
            print(f"Extracted title: {title}")
            if title and str(title).strip():
                return str(title).strip()
    return None

def setup_qa_chain(model: ChatOpenAI,  graph: Neo4jGraph) -> GraphCypherQAChain:
        """Setup the GraphCypherQA chain."""

        # Custom Cypher generation prompt with case-insensitive matching
        CYPHER_GENERATION_TEMPLATE = """Task: Translate a natural language question into a Cypher query for a Neo4j graph database.

You MUST strictly follow the provided schema.
You are NOT allowed to invent labels, relationship types, or properties.
Use ONLY what is explicitly defined below.

If a question cannot be answered using this schema, return an empty Cypher query.

---

## GRAPH SCHEMA

### NODE TYPES AND PROPERTIES

(Person {{
  id,
  name,
  location,
  email,
  phone,
  years_experience
}})

(Skill {{
  id,
  category,
  subcategory
}})

(Company {{
  id,
  name,
  industry,
  size,
  location
}})

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

(Certification {{
  id,
  name,
  provider,
  date_earned,
  expiry_date
}})

(University {{
  id,
  name,
  location,
  ranking
}})

(RFP {{
  id,
  title,
  description,
  requirements,
  budget,
  start_date,
  document_type
}})

---

### RELATIONSHIP TYPES AND PROPERTIES

(Person)-[HAS_SKILL {{
  proficiency,
  years_experience
}}]->(Skill)

(Person)-[WORKED_AT {{
  role,
  start_date,
  end_date
}}]->(Company)

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

(Person)-[EARNED {{
  date,
  score
}}]->(Certification)

(Person)-[STUDIED_AT {{
  degree,
  graduation_year,
  gpa
}}]->(University)

(Project)-[REQUIRES {{
  minimum_level,
  preferred_level
}}]->(Skill)

(RFP)-[NEEDS {{
  required_count,
  experience_level
}}]->(Skill)

---

## QUERY GENERATION RULES (MANDATORY)

1. Use ONLY the node labels, relationship types, and properties defined above.
2. NEVER invent properties or relationship types.
3. Use case-insensitive matching for text fields using `toLower()`.
4. When filtering by dates, use Neo4j `date()` where applicable.
5. When counting results, return meaningful column names.
6. If the question asks for something unsupported by the schema, return an empty query.
7. Do NOT include explanations, comments, or any text outside the Cypher query.
8. If you search for RFP use Rfp label. Example "MATCH (r:Rfp) WHERE ..."

---

## OUTPUT FORMAT
Return ONLY a valid Cypher query.

Examples:

# How many rpfs we have ?
MATCH (r:Rfp) return count(DISTINCT r)

# How many Python developers are available next month?
WITH
  date() + duration({{months: 1}}) AS startNextMonth,
  date() + duration({{months: 2}}) AS endNextMonth
MATCH (p:Person)-[:HAS_SKILL]->(s:Skill)
WHERE toLower(s.id) = toLower("python")
OPTIONAL MATCH (p)-[a:ASSIGNED_TO]->(pr:Project)
WHERE a.start_date <= endNextMonth
  AND a.end_date   >= startNextMonth
WITH p, collect(a) AS assignmentsInNextMonth
WHERE size(assignmentsInNextMonth) = 0
RETURN count(DISTINCT p) AS availablePythonDevelopersNextMonth


# How many developers have AWS certifications?
MATCH (p:Person)-[:EARNED]->(c:Certification)
WHERE toLower(c.name) CONTAINS toLower("aws")
   OR toLower(c.provider) CONTAINS toLower("aws")
RETURN count(DISTINCT p) AS awsCertifiedDevelopers


# Find all React and Node.js developers with at least 5 years of experience.
# ind senior developers with React AND Node.js experience
MATCH (p:Person)-[:HAS_SKILL]->(s1:Skill),
      (p:Person)-[:HAS_SKILL]->(s2:Skill)
WHERE toLower(s1.id) = toLower("react")
  AND (
        toLower(s2.id) = toLower("node.js")
     OR toLower(s2.id) = toLower("nodejs")
     OR toLower(s2.id) = toLower("node")
  )
  AND p.years_experience IS NOT NULL
  AND p.years_experience >= 5
RETURN DISTINCT p.id   AS id,
                p.name AS name,
                p.years_experience AS years_experience
ORDER BY years_experience DESC, name



# How many Python developers available Q2?
WITH
  date({{year: date().year, month: 4, day: 1}}) AS q2Start,
  date({{year: date().year, month: 6, day: 30}}) AS q2End

MATCH (p:Person)-[:HAS_SKILL]->(s:Skill)
WHERE toLower(s.id) = toLower("python")

OPTIONAL MATCH (p)-[a:ASSIGNED_TO]->(:Project)
WHERE a.start_date <= q2End
  AND a.end_date   >= q2Start

WITH p, collect(a) AS assignmentsInQ2
WHERE size(assignmentsInQ2) = 0

RETURN count(DISTINCT p) AS availablePythonDevelopersQ2

# What skills are missing candidates for RFP titled "Senior Backend Developer"?
MATCH (r:Rfp)-[:NEEDS]->(s:Skill)
WHERE toLower(r.title) = toLower($rfpTitle)

OPTIONAL MATCH (p:Person)-[:HAS_SKILL]->(s)

WITH s, collect(p) AS peopleWithSkill
WHERE size(peopleWithSkill) = 0

RETURN s.id AS missingSkill
ORDER BY missingSkill


# List developers sorted by their next free date
# Who becomes free when current projects end?
WITH date() AS today

MATCH (p:Person)-[a:ASSIGNED_TO]->(pr:Project)
WHERE a.start_date <= today
  AND a.end_date   >= today

WITH p, max(a.end_date) AS freeDate, collect(DISTINCT pr.title) AS currentProjects
RETURN
  p.id   AS personId,
  p.name AS personName,
  freeDate,
  currentProjects
ORDER BY freeDate ASC, personName ASC


# Find pairs of developers who have worked together on completed projects
# Find developers who worked together successfully
MATCH (p1:Person)-[:WORKED_ON]->(pr:Project)<-[:WORKED_ON]-(p2:Person)
WHERE p1 <> p2
  AND pr.status = "completed"

WITH
  p1,
  p2,
  count(DISTINCT pr) AS completedProjectsTogether,
  collect(DISTINCT pr.title) AS sharedProjects

RETURN
  p1.id   AS developer1Id,
  p1.name AS developer1Name,
  p2.id   AS developer2Id,
  p2.name AS developer2Name,
  completedProjectsTogether,
  sharedProjects
ORDER BY completedProjectsTogether DESC, developer1Name, developer2Name


# List available developers in Pacific timezone
MATCH (p:Person)
WHERE toLower(p.location) CONTAINS toLower("pacific")

OPTIONAL MATCH (p)-[a:ASSIGNED_TO]->(pr:Project)
WHERE a.start_date <= date()
  AND a.end_date   >= date()

WITH p, collect(a) AS activeAssignments
WHERE size(activeAssignments) = 0

RETURN p


# Number of years of experience for developers
MATCH (p:Person)
WITH toFloat(p.years_experience) AS y
WHERE y IS NOT NULL
RETURN avg(y) AS avgYearsExperience


# Total available FTE in Q4 2025
# Total capacity available for Q4 projects
WITH date("2025-10-01") AS q4Start, date("2025-12-31") AS q4End

MATCH (p:Person)

OPTIONAL MATCH (p)-[a:ASSIGNED_TO]->(:Project)
WHERE a.start_date <= q4End
  AND a.end_date   >= q4Start
  AND a.allocation_percentage IS NOT NULL

WITH p, sum(a.allocation_percentage) AS assignedPct

WITH
  CASE
    WHEN (100 - assignedPct) > 0 THEN (100 - assignedPct)
    ELSE 0
  END AS availablePct

RETURN
  sum(availablePct)              AS totalAvailablePct,
  sum(availablePct) / 100.0      AS totalAvailableFTE



# Top 5 universities of developers with most completed projects
# Developers from same university as our top performers
MATCH (tp:Person)-[:WORKED_ON]->(pr:Project)
WHERE toLower(pr.status) = toLower("completed")
WITH tp, count(DISTINCT pr) AS completedProjects
ORDER BY completedProjects DESC
LIMIT 5

// 2) Find their universities
MATCH (tp)-[:STUDIED_AT]->(u:University)

// 3) Find other developers from the same universities
MATCH (p:Person)-[:STUDIED_AT]->(u)
WHERE p <> tp

RETURN DISTINCT
  u.name AS university,
  p.id   AS developerId
ORDER BY university, developerId


# Who becomes available after current project ends?
// Find developers who become available after a given project ends
// A developer is considered available if they have no other assignments
// starting after the end date of the given project

MATCH (pr:Project)
WHERE toLower(pr.title) = toLower($projectTitle)

WITH pr, pr.end_date AS projectEndDate

MATCH (p:Person)-[a:ASSIGNED_TO]->(pr)
WHERE a.end_date = projectEndDate

OPTIONAL MATCH (p)-[a2:ASSIGNED_TO]->(otherPr:Project)
WHERE a2.start_date > projectEndDate

WITH p, collect(a2) AS futureAssignments
WHERE size(futureAssignments) = 0

RETURN DISTINCT
  p.id   AS developerId,
  p.name AS developerName
ORDER BY developerName

# Count of skills by graduation year
MATCH (p:Person)-[st:STUDIED_AT]->(:University)
WHERE st.graduation_year IS NOT NULL

MATCH (p)-[:HAS_SKILL]->(s:Skill)

RETURN
  st.graduation_year AS graduationYear,
  s.id               AS skill,
  count(DISTINCT p)  AS peopleCount
ORDER BY graduationYear ASC, peopleCount DESC, skill ASC


# Skills gaps analysis for upcoming project pipeline (next 3 months)
// Gap = skills required by upcoming projects, but with ZERO available people having them
// Availability = person has the skill AND has no ASSIGNED_TO overlapping the window

WITH
  date() AS winStart,
  date() + duration({{months: 3}}) AS winEnd

// 1) pick upcoming projects intersecting the window
MATCH (pr:Project)-[req:REQUIRES]->(s:Skill)
WHERE toLower(pr.status) IN ["upcoming","planned","pipeline"]
  AND pr.start_date <= winEnd
  AND pr.end_date   >= winStart

WITH winStart, winEnd, pr, s, req

// 2) count available people with that skill in the window
OPTIONAL MATCH (p:Person)-[:HAS_SKILL]->(s)
OPTIONAL MATCH (p)-[a:ASSIGNED_TO]->(:Project)
WHERE a.start_date <= winEnd
  AND a.end_date   >= winStart

WITH
  pr,
  s,
  req,
  p,
  collect(a) AS overlappingAssignments

WITH
  pr,
  s,
  req,
  collect(DISTINCT CASE WHEN p IS NOT NULL AND size(overlappingAssignments) = 0 THEN p END) AS availablePeople

WITH
  pr,
  s,
  req,
  [x IN availablePeople WHERE x IS NOT NULL] AS availablePeopleFiltered

RETURN
  pr.title AS projectTitle,
  s.id     AS requiredSkill,
  req.minimum_level   AS minimumLevel,
  req.preferred_level AS preferredLevel,
  size(availablePeopleFiltered) AS availablePeopleCount,
  CASE WHEN size(availablePeopleFiltered) = 0 THEN true ELSE false END AS isGap
ORDER BY isGap DESC, availablePeopleCount ASC, projectTitle ASC, requiredSkill ASC



# Risk assessment: single points of failure for a given project
// Single point of failure = exactly ONE person in the project has the required skill

MATCH (pr:Project)-[req:REQUIRES]->(s:Skill)
WHERE toLower(pr.title) = toLower($projectTitle)
  AND toLower(pr.status) IN ["ongoing", "active"]

// developers working on this project
MATCH (p:Person)-[:WORKED_ON]->(pr)
MATCH (p)-[:HAS_SKILL]->(s)

WITH pr, s, collect(DISTINCT p) AS skilledPeople
WHERE size(skilledPeople) = 1

RETURN
  pr.title            AS projectTitle,
  s.id                AS criticalSkill,
  skilledPeople[0].id AS singlePointOfFailureDeveloper,
  size(skilledPeople) AS developerCount
ORDER BY criticalSkill

# Risk assessment: single points of failure across all ongoing projects
MATCH (pr:Project)-[req:REQUIRES]->(s:Skill)
WHERE toLower(pr.status) IN ["ongoing", "active"]

MATCH (p:Person)-[:WORKED_ON]->(pr)
MATCH (p)-[:HAS_SKILL]->(s)

WITH pr, s, collect(DISTINCT p) AS skilledPeople
WHERE size(skilledPeople) = 1

RETURN
  pr.title            AS projectTitle,
  s.id                AS criticalSkill,
  skilledPeople[0].id AS singlePointOfFailureDeveloper
ORDER BY projectTitle, criticalSkill
---

Question:
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

import json
from typing import Any, Dict, List

def _to_int_or_none(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        if isinstance(x, int):
            return x
        if isinstance(x, float):
            return int(x)
        s = str(x).strip()
        if not s:
            return None
        return int(s)
    except Exception:
        return None

def _normalize_university_ranking(rank: Any) -> str:
    # u Ciebie 9999 to placeholder
    if rank in (None, "", compiler := "Not specified"):
        return "Not specified"
    if rank == 9999:
        return "Not specified"
    s = str(rank).strip()
    if s.lower() in {"n/a", "na", "not specified", "none", "null"}:
        return "Not specified"
    return s

def record_to_passage(r: Any, tool_name: str = "") -> str:
    """
    Zamienia rekord (dict/JSON/string) na krótki, tekstowy 'passage' dla RAGAS.
    """
    # jeśli to JSON-string z recordem
    if isinstance(r, str):
        s = r.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                r = json.loads(s)
            except Exception:
                return s
        else:
            return s

    # jeśli to dict z danymi o kandydacie
    if isinstance(r, dict):
        name = r.get("name") or r.get("personId") or "Unknown"
        proj = r.get("projectCount")
        uni = r.get("universityName")
        rank = _normalize_university_ranking(r.get("universityRanking"))

        parts = [f"{name}."]
        if proj is not None:
            parts.append(f"Project count: {proj}.")
        parts.append(f"University ranking: {rank}.")
        if uni:
            parts.append(f"University: {uni}.")
        # celowo pomijamy r.get("score")
        return " ".join(parts)

    # fallback: jako string, ale bez mega debugowania
    return str(r)

@traceable(process_inputs=lambda inputs: {"question": inputs["question"]})
def invoke_agent(agent: AgentExecutor, question: str) -> Dict[str, Any]:
    try:
        print(f"Executing query: {question}")

        result = agent.invoke(
            {"input": question},
            config={"configurable": {"session_id": "user_session"}}
        )

        answer = result.get("output", "No answer generated") if isinstance(result, dict) else str(result)
        steps = result.get("intermediate_steps", []) if isinstance(result, dict) else []

        retrieved_contexts: List[str] = []
        cypher_query = ""
        used_tools: List[str] = []

        for step in steps:
            if not (isinstance(step, tuple) and len(step) == 2):
                # nie dodawaj logów do contexts
                continue

            action, observation = step
            tool_name = getattr(action, "tool", None) or "unknown_tool"
            used_tools.append(tool_name)

            if tool_name == "graph_qa":
                if isinstance(observation, dict):
                    cypher_query = observation.get("cypher_query", "") or cypher_query
                    ctx = observation.get("retrieved_contexts", [])
                    if isinstance(ctx, list):
                        retrieved_contexts.extend([record_to_passage(x, tool_name) for x in ctx[:30]])
                    else:
                        retrieved_contexts.append(record_to_passage(ctx, tool_name))
                else:
                    retrieved_contexts.append(record_to_passage(observation, tool_name))

            elif tool_name in (
                "rank_best_devs_university",
                "match_devs_to_rfp_scored",
                "match_devs_to_rfp_scored_whatif",
                "compare_baseline_vs_whatif_for_rfp",
            ):
                if isinstance(observation, list):
                    for r in observation[:30]:
                        retrieved_contexts.append(record_to_passage(r, tool_name))
                else:
                    retrieved_contexts.append(record_to_passage(observation, tool_name))

            else:
                retrieved_contexts.append(record_to_passage(observation, tool_name))

        # ✅ filtr: tylko sensowne teksty do RAGAS
        retrieved_contexts = [
            c.strip() for c in retrieved_contexts
            if isinstance(c, str) and len(c.strip()) > 20
        ]

        # jeśli nadal pusto → dopiero wtedy fallback
        if not retrieved_contexts:
            retrieved_contexts = answer.strip().split("\n")[:5]

        return {
            "question": question,
            "answer": answer,
            "cypher_query": cypher_query,
            "retrieved_contexts": retrieved_contexts,
            "raw_context": steps,   # debug osobno
            "success": True,
        }

    except Exception as e:
        print(f"Query failed: {e}")
        return {
            "question": question,
            "answer": f"Error: {str(e)}",
            "cypher_query": "",
            "retrieved_contexts": [f"Error context: {str(e)}"],
            "raw_context": [],
            "success": False,
        }

def get_llm_transformer(model: ChatOpenAI) -> LLMGraphTransformer:

    SYSTEM_PROMPT  = r"""
# Knowledge Graph Extraction Instructions (STRICT RFP SCHEMA)

You extract a knowledge graph ONLY for RFP documents from the input text.
The output MUST strictly follow the schema defined below.
DO NOT create any nodes, relationships, or properties outside this schema.
If the input is not an RFP, output an empty graph (no nodes, no relationships).

---

## ID RULE (MANDATORY)
For RFP.id you MUST create a deterministic unique id based on the title.
Example:
- title: "Senior Python Developers" -> id: "rfp_senior_python_developers"
Rules:
- lowercase
- replace spaces and non-alphanumerics with underscores
- prefix with "rfp_"
If title is missing, do NOT create any RFP node.

## ALLOWED NODE TYPES AND PROPERTIES (RFP ONLY)

### RFP
(RFP {{
  id,
  title,
  description,
  requirements,
  budget,
  start_date,
  document_type
}})

### Skill
(Skill {{
  id,
  category,
  subcategory
}})

---

## ALLOWED RELATIONSHIP TYPES AND PROPERTIES (RFP ONLY)

### RFP → Skill
(RFP)-[NEEDS {{
  required_count,
  experience_level
}}]->(Skill)

---

## RELATIONSHIP PROPERTY EXTRACTION RULES (MANDATORY)

When you create a NEEDS relationship:

1) You MUST extract relationship properties ONLY when values are explicitly present in the text.
2) If a NEEDS relationship exists but no valid property values are explicitly present, create the relationship WITHOUT properties.
3) NEVER infer, guess, or fabricate values.

### NEEDS
Extract:
- required_count only if explicitly present as a number (e.g., "required_count: 3", "need 3 engineers")
- experience_level only if explicitly present as a clear level label (e.g., "experience_level: senior", "mid", "junior")

---

## METADATA BLOCK (MANDATORY HANDLING)

The input text may contain a block:

[METADATA]
document_type: <VALUE>

If present, you MUST:
- Assign `document_type` to the RFP node exactly as provided.
- NEVER ignore this field.
- DO NOT infer or hallucinate document_type if missing.

---

## GENERAL EXTRACTION RULES

- Extract ONLY explicitly stated information.
- DO NOT guess missing values.
- DO NOT normalize or reinterpret values.
- DO NOT create placeholder or default values.
- If a property is not present in the text, omit it.
- The main node MUST be an RFP if anything is extracted.
- IDs must be stable and unique per entity (use deterministic IDs if possible).
- Use consistent entity resolution (same entity -> same node).

---

## OUTPUT RULES (CRITICAL)

- Output ONLY the extracted graph data.
- DO NOT include explanations, comments, or reasoning.
- DO NOT include text outside the graph structure.
- Ensure strict adherence to the defined schema.
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
        allowed_nodes=["RFP", "Skill"],
        allowed_relationships=[("RFP", "NEEDS", "Skill")],
        node_properties=True, 
        relationship_properties=["required_count", "experience_level"],
        strict_mode=False,
        prompt=chat_prompt,
    )


    return llm_transformer


def store_single_graph_document(graph_document: GraphDocument, graph: Neo4jGraph) -> bool:
    """Zapisuje 1 GraphDocument, ale SKIP jeśli RFP o tym samym title już istnieje.
    Zwraca True gdy zapisano, False gdy pominięto.
    """
    try:
        title = get_rfp_title_from_graph_document(graph_document)

        print(f"Graph document with title: {title}")

        if title and rfp_title_exists(graph, title):
            print(f"SKIP: RFP with title already exists: {title}")
            return False

        graph.add_graph_documents(
            [graph_document],
            baseEntityLabel=True,
            include_source=False
        )

        print("✓ Stored 1 document in Neo4j")
        print(f"✓ Nodes: {len(graph_document.nodes)}")
        print(f"✓ Relationships: {len(graph_document.relationships)}")
        return True

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

        for rel in graph_document.relationships:
            if rel.type in ["NEEDS"]:
                print("REL:", rel.type, "PROPS:", rel.properties)

        return graph_document

    except Exception as e:
        print(f"Error converting {filename} to graph: {e}")
        return None