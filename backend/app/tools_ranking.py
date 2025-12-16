from typing import List, Dict, Any, Optional
import math

from langchain_core.tools import tool
from langchain_community.graphs import Neo4jGraph


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