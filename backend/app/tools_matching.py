from typing import Any, Dict, List
from langchain_core.tools import tool
from langchain_community.graphs import Neo4jGraph


def make_simple_match_tool(graph: Neo4jGraph):

    @tool
    def match_devs_to_rfp_simple(rfpTitle: str, default_required_count: int = 1) -> Dict[str, Any]:
        """ Match developers to a single RFP in the simplest possible way. Inputs: - rfpTitle: title of the RFP (must exist in the graph) - default_required_count: how many people to assign per skill if NEEDS.required_count is missing Output: - assignments: list of {skill, personId, name} - unfilled: list of skills where not enough people were found """

        if not rfpTitle or not rfpTitle.strip():
            return {"assignments": [], "unfilled": [{"reason": "missing rfpTitle"}]}

        # 1) Read needs: no label on s (works for :Skill, :__Entity__, or both)
        needs_q = """
        MATCH (r:Rfp)-[n:NEEDS]->(s)
        WHERE toLower(r.title) = toLower($title)
        RETURN s.id AS skillId, n.required_count AS required_count
        ORDER BY skillId
        """
        needs = graph.query(needs_q, {"title": rfpTitle.strip()})

        if not needs:
            return {"assignments": [], "unfilled": [{"reason": f'no NEEDS found for "{rfpTitle}"'}]}

        used = set()
        assignments: List[Dict[str, Any]] = []
        unfilled: List[Dict[str, Any]] = []

        for need in needs:
            skill = need.get("skillId")  # e.g. "Skill_Java"
            if not skill:
                continue

            normSkill = skill
            if isinstance(normSkill, str) and normSkill.lower().startswith("skill_"):
                normSkill = normSkill[6:]  # "Java" (case-insensitive compare anyway)

            req = need.get("required_count")
            try:
                req_n = int(req) if req is not None else int(default_required_count)
            except Exception:
                req_n = int(default_required_count)

            # 2) Find candidates: no label on s, match by both "Skill_Java" and "java"
            cand_q = """
            MATCH (p:Person)-[:HAS_SKILL]->(s)
            WHERE toLower(s.id) = toLower($skill)
               OR toLower(s.id) = toLower($normSkill)
            RETURN p.id AS personId, p.name AS name
            LIMIT 50
            """
            cands = graph.query(cand_q, {"skill": skill, "normSkill": normSkill}) or []

            picked = []
            for c in cands:
                pid = c.get("personId")
                if pid and pid not in used:
                    used.add(pid)
                    picked.append(c)
                if len(picked) >= req_n:
                    break

            if len(picked) < req_n:
                unfilled.append({"skill": skill, "required": req_n, "filled": len(picked)})

            for p in picked:
                assignments.append({"skill": skill, "personId": p["personId"], "name": p["name"]})

        return {"rfpTitle": rfpTitle.strip(), "assignments": assignments, "unfilled": unfilled}

    return match_devs_to_rfp_simple
