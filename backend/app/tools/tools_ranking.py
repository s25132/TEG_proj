from typing import List, Dict, Any
from langchain_core.tools import tool
from langchain_community.graphs import Neo4jGraph
from app.utilities.rank_utility import _rank_rows


def make_rank_tool(graph: Neo4jGraph):

    @tool
    def rank_best_devs_university(top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Rank developers using score = f(yearsExperience, projectCount, universityRanking).

        Use ONLY for questions like:
        - List developers with their project counts and university rankings
        - Give me best developers based on project counts and university rankings

        DO NOT use for:
        - pure count queries
        - skill-only queries
        """

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

        rows = graph.query(cypher) or []
        return _rank_rows(rows, top_n=top_n)

    return rank_best_devs_university
