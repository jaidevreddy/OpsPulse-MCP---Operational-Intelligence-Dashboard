from typing import Any, Dict, List
from collections import Counter, defaultdict
from pathlib import Path
import os

from src.loader import load_config, load_incidents, load_tickets, compute_ticket_sla_breaches
from src.cache import memo_get, memo_set, file_sig, ROOT


DATA_DIR = Path(os.getenv("OPSPULSE_DATA_DIR", ROOT / "data"))
CONFIG_PATH = Path(os.getenv("OPSPULSE_CONFIG_PATH", ROOT / "config" / "config.json"))


def _dataset_sig() -> str:
    inc = DATA_DIR / "sample_incidents.json"
    tck = DATA_DIR / "sample_tickets.json"
    return f"{file_sig(inc)}|{file_sig(tck)}"


def _config_sig() -> str:
    return file_sig(CONFIG_PATH)


def _text_for_item(obj: Any) -> str:
    parts = []
    for k in ["title", "root_cause", "service", "type"]:
        v = getattr(obj, k, None)
        if v:
            parts.append(str(v))
    tags = getattr(obj, "tags", None) or []
    parts.extend([str(x) for x in tags])
    return " ".join(parts).lower()


def extract_themes() -> Dict[str, Any]:
    key = ("themes", _dataset_sig() + "|" + _config_sig())
    cached = memo_get(key)
    if cached:
        return cached

    cfg = load_config()
    taxonomy: Dict[str, List[str]] = cfg.get("theme_taxonomy", {})

    incidents = load_incidents()
    tickets = load_tickets()

    theme_hits = Counter()
    per_theme_items = defaultdict(list)

    for item in list(incidents) + list(tickets):
        text = _text_for_item(item)
        for theme, keywords in taxonomy.items():
            if any(kw.lower() in text for kw in keywords):
                theme_hits[theme] += 1
                per_theme_items[theme].append(getattr(item, "id", "unknown"))

    result = {
        "themes_ranked": [{"theme": t, "count": c} for t, c in theme_hits.most_common()],
        "theme_to_item_ids": dict(per_theme_items)
    }
    memo_set(key, result)
    return result


def cluster_issues() -> Dict[str, Any]:
    key = ("clusters", _dataset_sig() + "|" + _config_sig())
    cached = memo_get(key)
    if cached:
        return cached

    incidents = load_incidents()

    clusters = defaultdict(list)
    for inc in incidents:
        rc = (inc.root_cause or "unknown").strip().lower()
        bucket = f"{inc.service}::{rc}"
        clusters[bucket].append(inc.id)

    ranked = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)

    result = {
        "clusters": [
            {"cluster_id": f"C{idx+1:03d}", "key": k, "size": len(v), "incident_ids": v}
            for idx, (k, v) in enumerate(ranked)
        ]
    }
    memo_set(key, result)
    return result


def score_health() -> Dict[str, Any]:
    key = ("health", _dataset_sig() + "|" + _config_sig())
    cached = memo_get(key)
    if cached:
        return cached

    cfg = load_config()
    scoring = cfg.get("health_scoring", {})
    penalties = scoring.get("penalties", {})
    thresholds = scoring.get("thresholds", {})
    base = int(scoring.get("base_score", 100))

    incidents = load_incidents()
    tickets = load_tickets()

    open_inc = [i for i in incidents if i.status == "open"]
    sev1_open = [i for i in open_inc if i.severity == "SEV1"]
    sev2_open = [i for i in open_inc if i.severity == "SEV2"]
    sla_breaches = compute_ticket_sla_breaches(tickets)

    per_service = Counter([i.service for i in incidents])
    high_volume_threshold = int(thresholds.get("high_volume_service_incidents", 999999))
    high_volume_services = [s for s, c in per_service.items() if c >= high_volume_threshold]
    multiplier = float(penalties.get("high_volume_service_multiplier", 1.0)) if high_volume_services else 1.0

    score = base
    score -= int(penalties.get("open_incident", 0)) * len(open_inc)
    score -= int(penalties.get("sev1_open_incident", 0)) * len(sev1_open)
    score -= int(penalties.get("sev2_open_incident", 0)) * len(sev2_open)
    score -= int(penalties.get("sla_breach_open_ticket", 0)) * int(sla_breaches)

    score = int(max(0, min(100, round(score * multiplier))))

    result = {
        "health_score": score,
        "drivers": {
            "open_incidents": len(open_inc),
            "sev1_open_incidents": len(sev1_open),
            "sev2_open_incidents": len(sev2_open),
            "sla_breaches_open_tickets": int(sla_breaches),
            "high_volume_services": high_volume_services
        }
    }
    memo_set(key, result)
    return result


def get_recommendations(top_n: int = 3) -> Dict[str, Any]:
    key = ("reco", f"{_dataset_sig()}|{_config_sig()}|topn={top_n}")
    cached = memo_get(key)
    if cached:
        return cached

    cfg = load_config()
    templates: Dict[str, List[str]] = cfg.get("recommendation_templates", {})

    themes = extract_themes()
    ranked = themes.get("themes_ranked", [])[:max(1, int(top_n))]

    recos = []
    for row in ranked:
        theme = row["theme"]
        tpls = templates.get(theme, [])
        recos.append({
            "theme": theme,
            "signal_count": row["count"],
            "actions": tpls[:3]
        })

    result = {"recommendations": recos}
    memo_set(key, result)
    return result
