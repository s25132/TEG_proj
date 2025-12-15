from typing import List, Dict, Any, Optional
import math

from langchain_core.tools import tool
from langchain_community.graphs import Neo4jGraph
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# -------------------------
# Scoring helpers (Twoje)
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
    for r in rows:
        r["score"] = compute_score(r)

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
# Tool 1: graph_qa (Twoje, lekko wpięte)
# -------------------------
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
# Tool 2: ranking (score przez _rank_rows)
# -------------------------
def make_rank_tool(graph: Neo4jGraph):
    @tool
    def rank_best_devs_university(top_n: int = 5, projectKeyword: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Rank developers using score = f(yearsExperience, projectCount, universityRanking).
        Use ONLY for questions like:
        # List developers with their project counts and university rankings
        # Give me best developers based on project counts and university rankings
        DO NOT use for other types of questions. DO NOT use for only count queries like "How many developers have AWS certifications?" 
        or "Top 5 universities of developers with most completed projects".
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
            rows = graph.query(cypher, {"kw": projectKeyword.strip()})
            return _rank_rows(rows, top_n=top_n)

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
        rows = graph.query(cypher)
        return _rank_rows(rows, top_n=top_n)

    return rank_best_devs_university


# -------------------------
# Agent: zawsze graph_qa, a ranking tylko dla tego jednego typu pytań
# -------------------------
def setup_agent(model, graph: Neo4jGraph, qa_chain):
    graph_qa_tool = make_graph_qa_tool(qa_chain)
    rank_tool = make_rank_tool(graph)

    tools = [graph_qa_tool, rank_tool]

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant.\n"
         "MANDATORY TOOL POLICY:\n"
         "1) For EVERY user question, you MUST call graph_qa(question) first.\n"
         "2) After graph_qa, you may call rank_best_devs_university ONLY if the user's input is EXACTLY one of:\n"
         "   - 'List developers with their project counts and university rankings'\n"
         "   - 'Give me best developers based on project counts and university rankings'\n"
         "   (case-insensitive, optional trailing punctuation)\n"
         "3) Never call rank_best_devs_university for any other question.\n"
         "4) Respond with a human-friendly final answer.\n"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm=model, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)
