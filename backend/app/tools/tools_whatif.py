from typing import Any, Dict, List, Tuple
from langchain_core.tools import tool
from langchain_community.graphs import Neo4jGraph
from app.utilities.rank_utility import _rank_rows
from app.utilities.utility import parse_start_date


def make_whatif_match_tool(graph: Neo4jGraph):

    @tool
    def match_devs_to_rfp_scored_whatif(
        rfpTitle: str,
        extra_devs: List[Dict[str, Any]] = None,
        default_required_count: int = 1,
        per_skill_candidate_limit: int = 50,
    ) -> Dict[str, Any]:
        """
        What-if: match developers to an RFP using scoring, but include extra (hypothetical) developers
        provided in `extra_devs`.

        extra_devs item example:
        {
          "personId": "W1",
          "name": "WhatIf Dev",
          "skills": ["Java", "Skill_AWS", "Python"],
          "yearsExperience": 6,
          "projectCount": 8,
          "universityRanking": 120
        }
        """

        if not rfpTitle or not rfpTitle.strip():
            return {"assignments": [], "unfilled": [{"reason": "missing rfpTitle"}]}

        title = rfpTitle.strip()
        extra_devs = extra_devs or []

        # ---- helper: does an extra dev have a given skill? (case-insensitive, supports Skill_ prefix)
        def _dev_has_skill(dev: Dict[str, Any], skill_id: str, norm_skill: str) -> bool:
            skills = dev.get("skills", [])
            if not isinstance(skills, list):
                return False

            want_a = (skill_id or "").strip().lower()
            want_b = (norm_skill or "").strip().lower()

            for s in skills:
                if not isinstance(s, str):
                    continue
                ss = s.strip().lower()
                if ss == want_a or ss == want_b:
                    return True
                # also allow Skill_Java vs Java mismatch
                if ss.startswith("skill_") and ss[6:] == want_b:
                    return True
                if want_a.startswith("skill_") and want_a[6:] == ss:
                    return True
            return False

        # ---- read needs for RFP
        needs_q = """
        MATCH (r:Rfp)-[n:NEEDS]->(s)
        WHERE toLower(r.title) = toLower($title)
        RETURN s.id AS skillId, n.required_count AS required_count, r.start_date AS start_date
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

            # ---- candidates from graph (same as u Ciebie)
            cand_q = """
            MATCH (p:Person)-[:HAS_SKILL]->(s)
            WHERE toLower(s.id) = toLower($skill)
                OR toLower(s.id) = toLower($normSkill)          
            AND NOT EXISTS {
                MATCH (p)-[a:ASSIGNED_TO]->(:Project)
                WHERE date(a.start_date) <= date($rfpStart)
                AND (a.end_date IS NULL OR date(a.end_date) >= date($rfpStart))
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
            graph_cands = graph.query(
                cand_q,
                {"skill": skill, "normSkill": normSkill, "limit": int(per_skill_candidate_limit), "rfpStart": rfp_start_date.date().isoformat()},
            ) or []

            # ---- candidates from extra_devs (what-if)
            extra_cands: List[Dict[str, Any]] = []
            for dev in extra_devs:
                if not isinstance(dev, dict):
                    continue
                if not dev.get("personId"):
                    continue
                if _dev_has_skill(dev, skill, normSkill):
                    extra_cands.append(
                        {
                            "personId": dev.get("personId"),
                            "name": dev.get("name"),
                            "yearsExperience": dev.get("yearsExperience", 0),
                            "projectCount": dev.get("projectCount", 0),
                            "universityRanking": dev.get("universityRanking", 9999),
                            "source": "what_if",
                        }
                    )

            # ---- merge + remove already used
            cands = (graph_cands or []) + (extra_cands or [])
            cands = [c for c in cands if c.get("personId") and c["personId"] not in used]

            # ---- rank using your scoring
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
                        "source": c.get("source", "graph"),  # graph vs what_if
                    }
                )

            if len(ranked) < req_n:
                unfilled.append({"skill": skill, "required": req_n, "filled": len(ranked)})

        return {"rfpTitle": title, "assignments": assignments, "unfilled": unfilled}

    return match_devs_to_rfp_scored_whatif

def make_compare_whatif_tool(match_devs_to_rfp_scored_whatif):
    """
    match_devs_to_rfp_scored_whatif: callable/tool that accepts
      (rfpTitle, extra_devs, default_required_count, per_skill_candidate_limit)
    and returns {rfpTitle, assignments, unfilled}
    """

    def _key(a: Dict[str, Any]) -> Tuple[str, str]:
        # identity of an assignment row
        return (str(a.get("skill", "")), str(a.get("personId", "")))

    def _by_skill(assignments: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        out: Dict[str, List[Dict[str, Any]]] = {}
        for a in assignments or []:
            sk = str(a.get("skill", ""))
            out.setdefault(sk, []).append(a)
        return out

    def _unfilled_map(unfilled: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        m: Dict[str, Dict[str, Any]] = {}
        for u in unfilled or []:
            sk = str(u.get("skill", u.get("reason", "")))
            m[sk] = u
        return m

    @tool
    def compare_baseline_vs_whatif_for_rfp(
        rfpTitle: str,
        extra_devs: List[Dict[str, Any]] = None,
        default_required_count: int = 1,
        per_skill_candidate_limit: int = 50,
    ) -> Dict[str, Any]:
        """
        Compare baseline vs what-if matching for an RFP and return a diff and include extra (hypothetical) developers
        provided in `extra_devs`.

        extra_devs item example:
        {
          "personId": "W1",
          "name": "WhatIf Dev",
          "skills": ["Java", "Skill_AWS", "Python"],
          "yearsExperience": 6,
          "projectCount": 8,
          "universityRanking": 120
        }
        
        Returns:
          - baseline, what_if (raw results)
          - diff: added_assignments, removed_assignments, per_skill_changes
          - unfilled_diff: skills improved/worsened/unchanged
        """
        baseline = match_devs_to_rfp_scored_whatif.invoke({
            "rfpTitle": rfpTitle,
            "extra_devs": [],
            "default_required_count": default_required_count,
            "per_skill_candidate_limit": per_skill_candidate_limit,
        })

        what_if = match_devs_to_rfp_scored_whatif.invoke({
            "rfpTitle": rfpTitle,
            "extra_devs": extra_devs or [],
            "default_required_count": default_required_count,
            "per_skill_candidate_limit": per_skill_candidate_limit,
        })  

        b_asg = baseline.get("assignments", []) or []
        w_asg = what_if.get("assignments", []) or []

        b_map = { _key(a): a for a in b_asg }
        w_map = { _key(a): a for a in w_asg }

        added_keys = [k for k in w_map.keys() if k not in b_map]
        removed_keys = [k for k in b_map.keys() if k not in w_map]

        added_assignments = [w_map[k] for k in added_keys]
        removed_assignments = [b_map[k] for k in removed_keys]

        # per-skill comparison: show baseline vs what-if picks
        b_by = _by_skill(b_asg)
        w_by = _by_skill(w_asg)

        all_skills = sorted(set(list(b_by.keys()) + list(w_by.keys())))

        per_skill_changes: List[Dict[str, Any]] = []
        for sk in all_skills:
            b_list = b_by.get(sk, [])
            w_list = w_by.get(sk, [])

            b_ids = [str(x.get("personId")) for x in b_list]
            w_ids = [str(x.get("personId")) for x in w_list]

            if b_ids != w_ids:
                per_skill_changes.append(
                    {
                        "skill": sk,
                        "baseline": [
                            {"personId": x.get("personId"), "name": x.get("name"), "score": x.get("score"), "source": x.get("source", "graph")}
                            for x in b_list
                        ],
                        "what_if": [
                            {"personId": x.get("personId"), "name": x.get("name"), "score": x.get("score"), "source": x.get("source", "graph")}
                            for x in w_list
                        ],
                    }
                )

        # unfilled diff
        b_unf = _unfilled_map(baseline.get("unfilled", []) or [])
        w_unf = _unfilled_map(what_if.get("unfilled", []) or [])

        all_unf_skills = sorted(set(list(b_unf.keys()) + list(w_unf.keys())))

        improved: List[Dict[str, Any]] = []
        worsened: List[Dict[str, Any]] = []
        unchanged: List[Dict[str, Any]] = []

        for sk in all_unf_skills:
            b = b_unf.get(sk)
            w = w_unf.get(sk)

            # handle "reason" entries
            if b and "reason" in b or w and "reason" in w:
                if b != w:
                    per = {"skill": sk, "baseline": b, "what_if": w}
                    # classify as changed
                    improved.append(per)
                else:
                    unchanged.append({"skill": sk, "baseline": b, "what_if": w})
                continue

            b_req = int(b.get("required", 0)) if b else 0
            b_fill = int(b.get("filled", 0)) if b else b_req  # if not present => treated as filled
            w_req = int(w.get("required", 0)) if w else 0
            w_fill = int(w.get("filled", 0)) if w else w_req

            # interpret "missing count" (how many still missing)
            b_missing = max(0, b_req - b_fill)
            w_missing = max(0, w_req - w_fill)

            rec = {
                "skill": sk,
                "baseline_missing": b_missing,
                "what_if_missing": w_missing,
                "baseline": b,
                "what_if": w,
            }

            if w_missing < b_missing:
                improved.append(rec)
            elif w_missing > b_missing:
                worsened.append(rec)
            else:
                unchanged.append(rec)

        return {
            "rfpTitle": rfpTitle,
            "baseline": baseline,
            "what_if": what_if,
            "diff": {
                "added_assignments": added_assignments,
                "removed_assignments": removed_assignments,
                "per_skill_changes": per_skill_changes,
            },
            "unfilled_diff": {
                "improved": improved,
                "worsened": worsened,
                "unchanged": unchanged,
            },
        }

    return compare_baseline_vs_whatif_for_rfp