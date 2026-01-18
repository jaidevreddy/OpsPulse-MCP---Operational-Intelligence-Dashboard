import os
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime, timezone

from dateutil.parser import isoparse

from src.schemas import Incident, Ticket
from src.cache import ROOT, load_json_cached


# Env-based file paths (Challenge 2 requirement)
DATA_DIR = Path(os.getenv("OPSPULSE_DATA_DIR", ROOT / "data"))
CONFIG_PATH = Path(os.getenv("OPSPULSE_CONFIG_PATH", ROOT / "config" / "config.json"))


def load_config() -> Dict[str, Any]:
    return load_json_cached(CONFIG_PATH)


def load_incidents() -> List[Incident]:
    raw = load_json_cached(DATA_DIR / "sample_incidents.json")
    return [Incident(**row) for row in raw]


def load_tickets() -> List[Ticket]:
    raw = load_json_cached(DATA_DIR / "sample_tickets.json")
    return [Ticket(**row) for row in raw]


def compute_ticket_sla_breaches(tickets: List[Ticket]) -> int:
    now = datetime.now(timezone.utc)
    breaches = 0
    for t in tickets:
        if t.status != "open" or not t.sla_due_at:
            continue
        due = isoparse(t.sla_due_at)
        if due.tzinfo is None:
            due = due.replace(tzinfo=timezone.utc)
        if now > due:
            breaches += 1
    return breaches


def build_basic_summary() -> Dict[str, Any]:
    incidents = load_incidents()
    tickets = load_tickets()

    open_incidents = [i for i in incidents if i.status == "open"]
    sev1_open = [i for i in open_incidents if i.severity == "SEV1"]
    sev2_open = [i for i in open_incidents if i.severity == "SEV2"]

    sla_breaches = compute_ticket_sla_breaches(tickets)

    by_service = {}
    for i in incidents:
        by_service.setdefault(i.service, {"incidents": 0, "open": 0})
        by_service[i.service]["incidents"] += 1
        if i.status == "open":
            by_service[i.service]["open"] += 1

    return {
        "counts": {
            "incidents_total": len(incidents),
            "incidents_open": len(open_incidents),
            "sev1_open": len(sev1_open),
            "sev2_open": len(sev2_open),
            "tickets_total": len(tickets),
            "sla_breaches_open_tickets": sla_breaches
        },
        "by_service": by_service
    }
