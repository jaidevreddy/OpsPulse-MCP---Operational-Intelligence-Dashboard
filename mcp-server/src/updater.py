import json
import os
from pathlib import Path
from typing import Dict, Any

from src.cache import ROOT

DATA_DIR = Path(os.getenv("OPSPULSE_DATA_DIR", ROOT / "data"))

INC_PATH = DATA_DIR / "sample_incidents.json"
TCK_PATH = DATA_DIR / "sample_tickets.json"


def _read(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write(path: Path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def add_ticket(ticket: Dict[str, Any]) -> Dict[str, Any]:
    data = _read(TCK_PATH)

    existing_ids = {t.get("id") for t in data}
    if ticket.get("id") in existing_ids:
        return {"ok": False, "error": "ticket_id_exists", "id": ticket.get("id")}

    data.append(ticket)
    _write(TCK_PATH, data)
    return {"ok": True, "added_ticket_id": ticket.get("id"), "tickets_total": len(data)}


def add_incident(incident: Dict[str, Any]) -> Dict[str, Any]:
    data = _read(INC_PATH)

    existing_ids = {i.get("id") for i in data}
    if incident.get("id") in existing_ids:
        return {"ok": False, "error": "incident_id_exists", "id": incident.get("id")}

    data.append(incident)
    _write(INC_PATH, data)
    return {"ok": True, "added_incident_id": incident.get("id"), "incidents_total": len(data)}
