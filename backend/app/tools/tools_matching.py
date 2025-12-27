from typing import Any, Dict, List
from langchain_core.tools import tool
from langchain_community.graphs import Neo4jGraph
from app.utilities.rank_utility import _rank_rows
from app.utilities.utility import parse_start_date

def make_simple_match_tool(graph: Neo4jGraph):

    @tool
    def match_devs_to_rfp_scored(
        rfpTitle: str,
        default_required_count: int = 1,
        per_skill_candidate_limit: int = 50,
    ) -> Dict[str, Any]:
        """Match developers to one RFP using scoring and pick best candidates per skill."""

        if not rfpTitle or not rfpTitle.strip():
            return {"assignments": [], "unfilled": [{"reason": "missing rfpTitle"}]}

        title = rfpTitle.strip()

        needs_q = """
            MATCH (r:Rfp)-[n:NEEDS]->(s)
                WHERE toLower(r.title) = toLower($title)
            RETURN 
                s.id AS skillId,
                n.required_count AS required_count,
                r.start_date AS start_date
        ORDER BY skillId
        """
        needs = graph.query(needs_q, {"title": title}) or []
        if not needs:
            return {"assignments": [], "unfilled": [{"reason": f'no NEEDS found for "{title}"'}]}

        used = set()
        assignments: List[Dict[str, Any]] = []
        unfilled: List[Dict[str, Any]] = []

        rfp_start_date = parse_start_date(needs[0].get("start_date"))

        for need in needs:
            skill = need.get("skillId")
            if not isinstance(skill, str) or not skill.strip():
                continue

            skill = skill.strip()
            normSkill = skill[6:] if skill.lower().startswith("skill_") else skill

            req = need.get("required_count")
            try:
                req_n = int(req) if req is not None else int(default_required_count)
            except Exception:
                req_n = int(default_required_count)

            # pobierz kandydatów + atrybuty do scoringu
            cand_q = """
            MATCH (p:Person)-[:HAS_SKILL]->(s)
            WHERE toLower(s.id) = toLower($skill) OR toLower(s.id) = toLower($normSkill)
            AND NOT EXISTS {
                MATCH (p)-[a:ASSIGNED_TO]->(:Project)
                WHERE (a.end_date IS NULL OR date(a.end_date) >= date($rfpStart))
            }
            

            OPTIONAL MATCH (p)-[:WORKED_ON]->(pr:Project)
            WITH p, count(DISTINCT pr) AS projectCount

            OPTIONAL MATCH (p)-[:STUDIED_AT]->(u:University)

            RETURN
                p.id AS personId,
                p.name AS name,
                coalesce(p.years_experience, 0) AS yearsExperience,
                projectCount,
                coalesce(u.ranking, 9999) AS universityRanking
            LIMIT $limit
            """
            cands = graph.query(
                cand_q,
                {"skill": skill, "normSkill": normSkill, "limit": int(per_skill_candidate_limit),  "rfpStart": rfp_start_date.date().isoformat()},
            ) or []

            # filtr "used" zanim zrobisz ranking, żeby nie marnować topów
            cands = [c for c in cands if c.get("personId") and c["personId"] not in used]

            # ranking po score i wybór top req_n
            ranked = _rank_rows(cands, top_n=req_n)

            for c in ranked:
                used.add(c["personId"])
                assignments.append(
                    {
                        "skill": skill,
                        "personId": c["personId"],
                        "name": c.get("name"),
                        "score": c.get("score"),
                        "yearsExperience": c.get("yearsExperience"),
                        "projectCount": c.get("projectCount"),
                        "universityRanking": c.get("universityRanking"),
                    }
                )

            if len(ranked) < req_n:
                unfilled.append({"skill": skill, "required": req_n, "filled": len(ranked)})

        return {"rfpTitle": title, "assignments": assignments, "unfilled": unfilled}

    return match_devs_to_rfp_scored
