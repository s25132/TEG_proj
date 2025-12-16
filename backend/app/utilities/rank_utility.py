from typing import List, Dict, Any
import math

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

