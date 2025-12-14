from typing import List, Dict, Any, Optional
import math

from langchain_core.tools import tool
from langchain_community.graphs import Neo4jGraph
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# -------------------------
# Scoring helpers
# -------------------------

def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))

def to_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default

def university_score_from_ranking(ranking: Any) -> float:
    r = to_float(ranking, default=9999.0)
    if r <= 50:
        return 1.0
    if r >= 500:
        return 0.0
    return 1.0 - (r - 50.0) / 450.0

def compute_score(row: Dict[str, Any]) -> float:
    years = to_float(row.get("yearsExperience"), 0.0)
    projects = to_float(row.get("projectCount"), 0.0)
    uni_rank = row.get("universityRanking", 9999)

    years_norm = clamp(years / 10.0)
    projects_norm = 1.0 - math.exp(-projects / 5.0)
    uni_norm = university_score_from_ranking(uni_rank)

    score = 0.45 * years_norm + 0.45 * projects_norm + 0.10 * uni_norm
    return round(score, 4)

def _rank_rows(rows: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
    """Internal (non-tool) ranking: adds score + sorts + returns top_n."""
    for r in rows:
        r["score"] = compute_score(r)

    # stabilne sortowanie (przy remisach)
    rows.sort(
        key=lambda x: (
            x.get("score", 0.0),
            to_float(x.get("yearsExperience"), 0.0),
            to_float(x.get("projectCount"), 0.0),
            str(x.get("personId", "")),
        ),
        reverse=True,
    )
    return rows[: max(0, int(top_n))]


# -------------------------
# Tools
# -------------------------

def make_tools(graph: Neo4jGraph):
    @tool
    def fetch_candidates(projectKeyword: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Fetch developer features needed for ranking.
        If projectKeyword is provided, restrict to people who worked on projects matching that keyword.
        """
        if projectKeyword and projectKeyword.strip():
            cypher = """
            MATCH (p:Person)-[:WORKED_ON]->(pr:Project)
            WHERE toLower(pr.title) CONTAINS toLower($kw)

            WITH DISTINCT p
            OPTIONAL MATCH (p)-[:WORKED_ON]->(pr2:Project)
            WITH p, count(DISTINCT pr2) AS projectCount

            OPTIONAL MATCH (p)-[:STUDIED_AT]->(u:University)

            RETURN
              p.id   AS personId,
              p.name AS name,
              coalesce(p.years_experience, 0) AS yearsExperience,
              projectCount,
              coalesce(u.ranking, 9999) AS universityRanking,
              u.name AS universityName
            """
            return graph.query(cypher, {"kw": projectKeyword.strip()})

        cypher = """
        MATCH (p:Person)

        OPTIONAL MATCH (p)-[:WORKED_ON]->(pr:Project)
        WITH p, count(DISTINCT pr) AS projectCount

        OPTIONAL MATCH (p)-[:STUDIED_AT]->(u:University)

        RETURN
          p.id   AS personId,
          p.name AS name,
          coalesce(p.years_experience, 0) AS yearsExperience,
          projectCount,
          coalesce(u.ranking, 9999) AS universityRanking,
          u.name AS universityName
        """
        return graph.query(cypher)

    @tool
    def recommend_candidates(projectKeyword: Optional[str] = None, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Fetch and rank candidates. Use this for 'top/best/ranking' developer requests.
        Only use recommend_candidates for questions like "Give me best developers based on project counts and university rankings"
        This avoids passing `rows` between tools (more reliable).
        """
        rows = fetch_candidates.invoke({"projectKeyword": projectKeyword})
        return _rank_rows(rows, top_n=top_n)

    # ✅ celowo NIE zwracamy rank_candidates jako tool
    return [fetch_candidates, recommend_candidates]


def make_graph_qa_tool(qa_chain):
    @tool
    def graph_qa(question: str) -> Dict[str, Any]:
        """
        Answer a question using GraphCypherQAChain over Neo4j.
        Returns answer + cypher_query + retrieved_contexts.
        """
        res = qa_chain.invoke({"query": question})

        out = {"answer": "", "cypher_query": "", "retrieved_contexts": []}

        if isinstance(res, dict):
            out["answer"] = res.get("result", "")

            steps = res.get("intermediate_steps", [])
            # GraphCypherQAChain często: steps[0]={"query":...}, steps[1]={"context":[...]}
            if len(steps) > 0 and isinstance(steps[0], dict):
                out["cypher_query"] = steps[0].get("query", "") or ""

            if len(steps) > 1 and isinstance(steps[1], dict):
                ctx = steps[1].get("context", [])
                if isinstance(ctx, list):
                    out["retrieved_contexts"] = [
                        ", ".join(f"{k}={v}" for k, v in rec.items()) if isinstance(rec, dict) else str(rec)
                        for rec in ctx
                    ]
        else:
            out["answer"] = str(res)

        return out

    return graph_qa


# -------------------------
# Agent
# -------------------------

def setup_agent(model, graph: Neo4jGraph, qa_chain):
    custom_tools = make_tools(graph)  # [fetch_candidates, recommend_candidates]
    graph_qa_tool = make_graph_qa_tool(qa_chain)

    tools = [graph_qa_tool, *custom_tools]

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant.\n"
         "Use tools when needed.\n"
         "- If the user asks for top/best/ranking, call recommend_candidates.Only use recommend_candidates for that. Example: Give me best developers based on project counts and university rankings\n"
         "- For general graph questions, call graph_qa.\n"
         "Never call tools with missing required arguments."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm=model, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)
