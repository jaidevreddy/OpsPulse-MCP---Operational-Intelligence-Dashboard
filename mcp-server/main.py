import os
import json
import yaml
import hashlib
import io
import subprocess
import requests
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw, ImageFont
import textwrap


# PDF generator
from fpdf import FPDF

# ==============================
# Paths + Config Loading
# ==============================
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
CONFIG_DIR = BASE_DIR / "config"

DEFAULT_PROFILE = os.getenv("OPSPULSE_PROFILE", "default")  # default | infra

DEFAULT_TAXONOMY = CONFIG_DIR / "taxonomy.yaml"
DEFAULT_SCORING = CONFIG_DIR / "scoring.yaml"

TICKETS_PATH = Path(os.getenv("OPSPULSE_TICKETS_PATH", str(DATA_DIR / "sample_tickets.json")))
INCIDENTS_PATH = Path(os.getenv("OPSPULSE_INCIDENTS_PATH", str(DATA_DIR / "sample_incidents.json")))

CACHE_ENABLED = os.getenv("OPSPULSE_CACHE_ENABLED", "true").lower() == "true"

# ==============================
# Gemini Config (optional)
# ==============================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_taxonomy_for_profile(profile: str) -> dict:
    profile_dir = CONFIG_DIR / "profiles" / profile
    taxonomy_path = profile_dir / "taxonomy.yaml"
    if taxonomy_path.exists():
        return _load_yaml(taxonomy_path)
    return _load_yaml(DEFAULT_TAXONOMY)


def load_scoring_for_profile(profile: str) -> dict:
    profile_dir = CONFIG_DIR / "profiles" / profile
    scoring_path = profile_dir / "scoring.yaml"
    if scoring_path.exists():
        return _load_yaml(scoring_path)
    return _load_yaml(DEFAULT_SCORING)


# ==============================
# Global Runtime State
# ==============================
DATASET: Dict[str, Any] = {"tickets": [], "incidents": []}
DATASET_HASH: Optional[str] = None

CACHE: Dict[str, Any] = {}


def compute_hash(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def cache_get(key: str):
    if not CACHE_ENABLED:
        return None
    return CACHE.get(key)


def cache_set(key: str, value: Any):
    if not CACHE_ENABLED:
        return
    CACHE[key] = value


def load_ops_data_from_disk() -> Dict[str, Any]:
    tickets = []
    incidents = []

    if TICKETS_PATH.exists():
        with open(TICKETS_PATH, "r", encoding="utf-8") as f:
            tickets = json.load(f)

    if INCIDENTS_PATH.exists():
        with open(INCIDENTS_PATH, "r", encoding="utf-8") as f:
            incidents = json.load(f)

    return {"tickets": tickets, "incidents": incidents}


def refresh_runtime():
    global DATASET, DATASET_HASH
    DATASET = load_ops_data_from_disk()
    DATASET_HASH = compute_hash(DATASET)
    CACHE.clear()


refresh_runtime()

# ==============================
# Payment Failure Keyword Matcher
# ==============================
PAYMENT_FAILURE_KEYWORDS = [
    "payment failure",
    "payment failed",
    "payment declined",
    "upi failed",
    "card declined",
    "checkout failed",
]


def _text_blob(obj: dict) -> str:
    title = str(obj.get("title", "") or "")
    desc = str(obj.get("description", "") or "")
    tags = obj.get("tags", [])
    if not isinstance(tags, list):
        tags = []
    tags_joined = " ".join([str(t) for t in tags])
    return f"{title} {desc} {tags_joined}".lower()


def count_payment_failure_signals() -> int:
    count = 0

    tickets = DATASET.get("tickets", [])
    incidents = DATASET.get("incidents", [])

    for t in tickets:
        tx = _text_blob(t)
        if any(k in tx for k in PAYMENT_FAILURE_KEYWORDS):
            count += 1

    for i in incidents:
        tx = _text_blob(i)
        if any(k in tx for k in PAYMENT_FAILURE_KEYWORDS):
            count += 1

    return count


def payment_failure_breakdown_value() -> int:
    c = count_payment_failure_signals()

    if c <= 0:
        return 0
    if c == 1:
        return 25
    if c == 2:
        return 45
    if c == 3:
        return 60
    if c == 4:
        return 75
    if c >= 5:
        return min(100, 75 + (c - 4) * 5)

    return 0


# ==============================
# Gemini Helper (used only for Summary card recos)
# ==============================
def gemini_generate_json(prompt: str) -> dict:
    if not GEMINI_API_KEY:
        raise ValueError("Missing GEMINI_API_KEY in environment")

    url = f"https://generativelanguage.googleapis.com/v1beta/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2},
    }

    r = requests.post(url, json=body, timeout=60)
    r.raise_for_status()
    data = r.json()

    text = data["candidates"][0]["content"]["parts"][0]["text"].strip()

    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    return json.loads(text)


# ==============================
# ✅ PDF helper: wrap long words
# ==============================
def wrap_long_words(text: str, max_len: int = 80) -> str:
    """
    FPDF crashes if a SINGLE WORD is longer than the page width.
    This function breaks long unbroken strings safely.
    """
    out_lines = []

    for line in text.split("\n"):
        words = line.split(" ")
        new_words = []

        for w in words:
            if len(w) > max_len:
                chunks = [w[i : i + max_len] for i in range(0, len(w), max_len)]
                new_words.extend(chunks)
            else:
                new_words.append(w)

        out_lines.append(" ".join(new_words))

    return "\n".join(out_lines)


# ==============================
# FastAPI App
# ==============================
app = FastAPI(title="OpsPulse MCP Server", version="0.7.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/about")
def about():
    return {
        "name": "OpsPulse MCP",
        "version": "0.7.1",
        "default_profile": DEFAULT_PROFILE,
        "challenge": "MCP + Docker + WebSocket + Report download ready",
        "mcp_endpoint": "/mcp",
        "tickets_path": str(TICKETS_PATH),
        "incidents_path": str(INCIDENTS_PATH),
        "cache_enabled": CACHE_ENABLED,
    }


@app.get("/health/live")
def health_live():
    return {"status": "live", "ts": datetime.utcnow().isoformat()}


@app.get("/health/ready")
def health_ready():
    ok = isinstance(DATASET, dict) and "tickets" in DATASET and "incidents" in DATASET
    return {"status": "ready" if ok else "not_ready", "default_profile": DEFAULT_PROFILE}


@app.get("/cache/stats")
def cache_stats():
    if not CACHE_ENABLED:
        return {"cache_enabled": False, "default_profile": DEFAULT_PROFILE, "keys": 0, "dataset_hash": DATASET_HASH}

    keys = sorted(list(CACHE.keys()))
    return {
        "cache_enabled": True,
        "keys": len(keys),
        "cache_keys": keys,
        "default_profile": DEFAULT_PROFILE,
        "dataset_hash": DATASET_HASH,
    }


# ==============================
# Core Tool Logic
# ==============================
def tool_ping() -> Dict[str, Any]:
    return {"message": "pong", "ts": datetime.utcnow().isoformat()}


def tool_load_ops_data() -> Dict[str, Any]:
    return {
        "dataset_hash": DATASET_HASH,
        "tickets": len(DATASET.get("tickets", [])),
        "incidents": len(DATASET.get("incidents", [])),
        "default_profile": DEFAULT_PROFILE,
    }


def tool_get_basic_summary() -> Dict[str, Any]:
    cache_key = f"summary::{DATASET_HASH}"
    cached = cache_get(cache_key)
    if cached:
        return cached

    tickets = DATASET.get("tickets", [])
    incidents = DATASET.get("incidents", [])

    open_tickets = [t for t in tickets if t.get("status", "").lower() != "closed"]
    open_incidents = [i for i in incidents if i.get("status", "").lower() != "closed"]

    out = {
        "tickets_total": len(tickets),
        "tickets_open": len(open_tickets),
        "incidents_total": len(incidents),
        "incidents_open": len(open_incidents),
    }

    cache_set(cache_key, out)
    return out


def tool_extract_themes(profile: str) -> Dict[str, Any]:
    cache_key = f"themes::{profile}::{DATASET_HASH}"
    cached = cache_get(cache_key)
    if cached:
        return cached

    taxonomy_cfg = load_taxonomy_for_profile(profile)
    themes_cfg = taxonomy_cfg.get("themes", [])

    tickets = DATASET.get("tickets", [])
    incidents = DATASET.get("incidents", [])

    texts = []
    for t in tickets:
        texts.append((t.get("title", "") + " " + t.get("description", "")).lower())
    for i in incidents:
        texts.append((i.get("title", "") + " " + i.get("description", "")).lower())

    theme_counts = {}
    for theme in themes_cfg:
        name = theme.get("name")
        kws = [k.lower() for k in theme.get("keywords", [])]
        if not name or not kws:
            continue

        count = 0
        for tx in texts:
            if any(k in tx for k in kws):
                count += 1

        if count > 0:
            theme_counts[name] = count

    out = {
        "profile": profile,
        "themes": sorted(
            [{"theme": k, "signals": v} for k, v in theme_counts.items()],
            key=lambda x: x["signals"],
            reverse=True,
        ),
    }

    cache_set(cache_key, out)
    return out


def tool_score_health(profile: str) -> Dict[str, Any]:
    cache_key = f"health::{profile}::{DATASET_HASH}"
    cached = cache_get(cache_key)
    if cached:
        return cached

    scoring_cfg = load_scoring_for_profile(profile)
    rules = scoring_cfg.get("health_score", {})
    base = int(rules.get("base", 100))
    deductions = rules.get("deductions", {})

    tickets = DATASET.get("tickets", [])
    incidents = DATASET.get("incidents", [])

    open_incidents = [i for i in incidents if i.get("status", "").lower() != "closed"]
    open_tickets = [t for t in tickets if t.get("status", "").lower() != "closed"]

    sev1 = [i for i in open_incidents if str(i.get("severity", "")).lower() in ["sev1", "1"]]
    sla_breaches = [t for t in open_tickets if t.get("sla_breached") is True]

    score = base
    score -= int(deductions.get("open_incident", 10)) * len(open_incidents)
    score -= int(deductions.get("sev1_incident", 25)) * len(sev1)
    score -= int(deductions.get("sla_breach", 5)) * len(sla_breaches)

    score = max(0, min(100, score))

    out = {
        "health_score": score,
        "drivers": {
            "open_incidents": len(open_incidents),
            "sev1_open_incidents": len(sev1),
            "sla_breaches_open_tickets": len(sla_breaches),
        },
    }

    cache_set(cache_key, out)
    return out


def tool_get_recommendations(profile: str, top_n: int = 3) -> Dict[str, Any]:
    cache_key = f"llm_recos::{profile}::{top_n}::{DATASET_HASH}"
    cached = cache_get(cache_key)
    if cached:
        return cached

    health = tool_score_health(profile=profile)
    themes = tool_extract_themes(profile=profile)
    theme_list = themes.get("themes", [])
    top_themes = theme_list[: max(1, int(top_n))]

    tickets = DATASET.get("tickets", [])
    incidents = DATASET.get("incidents", [])

    prompt = f"""
You are a senior SRE.
Generate {top_n} SHORT technical summary actions for a dashboard.

Rules:
- Return ONLY valid JSON (no markdown)
- Each recommendation MUST be <= 12 words
- Must be technical and specific
- No generic phrases
- priority: P0/P1/P2

Profile: {profile}
Health score: {health.get("health_score")}

Themes:
{json.dumps(top_themes, indent=2)}

Incidents:
{json.dumps(incidents[:10], indent=2)}

Tickets:
{json.dumps(tickets[:10], indent=2)}

Return JSON EXACTLY:
{{
  "recommendations": [
    {{
      "priority": "P0",
      "theme": "<theme>",
      "recommendation": "<short technical action>"
    }}
  ]
}}
""".strip()

    try:
        llm_out = gemini_generate_json(prompt)
        recos = llm_out.get("recommendations", [])

        cleaned = []
        for r in recos:
            pr = str(r.get("priority", "P2")).upper()
            if pr not in ["P0", "P1", "P2"]:
                pr = "P2"

            cleaned.append(
                {
                    "priority": pr,
                    "theme": r.get("theme") or "Ops",
                    "recommendation": (r.get("recommendation") or "").strip()[:120],
                }
            )

        out = {"health_score": health.get("health_score"), "recommendations": cleaned[: int(top_n)], "mode": "llm"}
        cache_set(cache_key, out)
        return out

    except Exception as e:
        fallback = []
        for idx, th in enumerate(top_themes):
            fallback.append(
                {
                    "priority": ["P0", "P1", "P2"][min(idx, 2)],
                    "theme": th.get("theme", "Ops"),
                    "recommendation": f"Check {th.get('theme','Ops')} stability and apply fixes.",
                }
            )

        out = {
            "health_score": health.get("health_score"),
            "recommendations": fallback[: int(top_n)],
            "mode": "fallback",
            "llm_error": str(e),
        }
        cache_set(cache_key, out)
        return out


def tool_refresh_data() -> Dict[str, Any]:
    refresh_runtime()
    return {"ok": True, "message": "Dataset refreshed", "dataset_hash": DATASET_HASH}


def tool_add_ticket(ticket: dict) -> Dict[str, Any]:
    DATASET["tickets"].append(ticket)

    try:
        current = []
        if TICKETS_PATH.exists():
            with open(TICKETS_PATH, "r", encoding="utf-8") as f:
                current = json.load(f)
        if not isinstance(current, list):
            current = []

        current.append(ticket)
        TICKETS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(TICKETS_PATH, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2)

    except Exception as e:
        return {"ok": False, "error": f"Ticket added but file save failed: {str(e)}"}

    global DATASET_HASH
    DATASET_HASH = compute_hash(DATASET)
    CACHE.clear()
    return {"ok": True, "tickets_total": len(DATASET['tickets']), "dataset_hash": DATASET_HASH}


def tool_add_incident(incident: dict) -> Dict[str, Any]:
    DATASET["incidents"].append(incident)

    try:
        current = []
        if INCIDENTS_PATH.exists():
            with open(INCIDENTS_PATH, "r", encoding="utf-8") as f:
                current = json.load(f)
        if not isinstance(current, list):
            current = []

        current.append(incident)
        INCIDENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(INCIDENTS_PATH, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2)

    except Exception as e:
        return {"ok": False, "error": f"Incident added but file save failed: {str(e)}"}

    global DATASET_HASH
    DATASET_HASH = compute_hash(DATASET)
    CACHE.clear()
    return {"ok": True, "incidents_total": len(DATASET['incidents']), "dataset_hash": DATASET_HASH}


# ==============================
# Snapshot Endpoint
# ==============================
def build_snapshot(profile: str) -> Dict[str, Any]:
    basic = tool_get_basic_summary()
    health = tool_score_health(profile=profile)
    recos = tool_get_recommendations(profile=profile, top_n=4)

    score = int(health.get("health_score", 0))

    tickets_total = int(basic.get("tickets_total", 0))
    incidents_total = int(basic.get("incidents_total", 0))

    areas = []
    for r in recos.get("recommendations", []):
        areas.append({"tag": r.get("theme", "Ops"), "label": r.get("recommendation", ""), "value": r.get("priority", "P2")})

    if profile == "infra":
        breakdown = [
            {"label": "DB Capacity", "value": min(100, incidents_total * 10)},
            {"label": "Kubernetes", "value": min(100, tickets_total * 10)},
            {"label": "Autoscaling", "value": min(100, (tickets_total + incidents_total) * 7)},
            {"label": "Queue", "value": min(100, incidents_total * 8)},
            {"label": "Deploy", "value": min(100, tickets_total * 6)},
            {"label": "Networking", "value": min(100, incidents_total * 6)},
            {"label": "Cloud Limits", "value": min(100, tickets_total * 5)},
        ]
    else:
        payments_value = payment_failure_breakdown_value()
        payments_signals = count_payment_failure_signals()

        breakdown = [
            {"label": "App/API", "value": min(100, incidents_total * 10)},
            {"label": "Payments", "value": payments_value, "meta": {"mode": "keyword_dynamic", "keyword": "payment failure", "signals": payments_signals}},
            {"label": "Customer", "value": min(100, incidents_total * 7)},
            {"label": "Support", "value": min(100, tickets_total * 10)},
            {"label": "SLA", "value": min(100, int(health.get("drivers", {}).get("sla_breaches_open_tickets", 0)) * 20)},
        ]

    return {
        "health": {"scorePercent": score, "scoreNumerator": score, "scoreDenominator": 100},
        "stats": {
            "resources": {"value": str(tickets_total + incidents_total), "subtext": "Tickets + Incidents tracked"},
            "probes": {"value": str(incidents_total), "subtext": "Open + closed incidents"},
            "cache": {"value": f"{len(CACHE)} keys" if CACHE_ENABLED else "off", "subtext": "Cached computations"},
        },
        "trend": {
            "summaryValue": str(score),
            "deltaText": "↑ 0.0%",
            "points": [{"label": "W1", "value": max(0, score - 12)}, {"label": "W2", "value": max(0, score - 7)}, {"label": "W3", "value": max(0, score - 4)}, {"label": "W4", "value": score}],
        },
        "areasToAddress": areas,
        "breakdown": breakdown,
        "meta": {"profile": profile, "dataset_hash": DATASET_HASH, "cache_enabled": CACHE_ENABLED, "llm_mode": recos.get("mode", "unknown")},
    }


@app.get("/snapshot")
def snapshot(profile: str = DEFAULT_PROFILE):
    return build_snapshot(profile)


# ==============================
# ✅ Report Download Endpoint (PDF)
# ==============================
@app.get("/report/download")
def download_report(profile: str = DEFAULT_PROFILE):
    """
    ✅ Uses llm_client_gemini.py output DIRECTLY.
    ✅ Generates PDF by rendering text to images (bulletproof, no FPDF unicode crash)
    """

    script_path = BASE_DIR / "llm_client_gemini.py"

    if not script_path.exists():
        return JSONResponse({"ok": False, "error": f"llm_client_gemini.py not found at {script_path}"}, status_code=500)

    try:
        result = subprocess.run(
            ["python3", str(script_path)],
            capture_output=True,
            text=True,
            cwd=str(BASE_DIR),
            timeout=120,
        )
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    if result.returncode != 0:
        return JSONResponse(
            {"ok": False, "error": "llm_client_gemini.py failed", "stderr": result.stderr[-2000:]},
            status_code=500,
        )

    stdout = result.stdout.strip()

    if "OPSPULSE EXECUTIVE REPORT" in stdout:
        report_text = stdout.split("OPSPULSE EXECUTIVE REPORT", 1)[-1].strip()
    else:
        report_text = stdout

    # -------------------------
    # Render report into images
    # -------------------------
    page_width, page_height = 1240, 1754  # A4-ish at 150 dpi
    margin = 80
    line_height = 28

    # Use default font (works everywhere)
    try:
        font = ImageFont.truetype("Arial.ttf", 18)
        font_bold = ImageFont.truetype("Arial Bold.ttf", 26)
    except:
        font = ImageFont.load_default()
        font_bold = ImageFont.load_default()

    # Header
    header_lines = [
        "OpsPulse Executive Brief",
        f"Profile: {profile}",
        f"Generated: {datetime.utcnow().isoformat()} UTC",
        "",
    ]
    header_text = "\n".join(header_lines) + "\n" + report_text

    # Wrap lines safely
    wrapped_lines = []
    for raw_line in header_text.split("\n"):
        if raw_line.strip() == "":
            wrapped_lines.append("")
            continue
        # wrap big lines to avoid overflow
        wrapped_lines.extend(textwrap.wrap(raw_line, width=95, break_long_words=True, break_on_hyphens=False))

    pages = []
    y = margin

    img = Image.new("RGB", (page_width, page_height), "white")
    draw = ImageDraw.Draw(img)

    # Draw title bigger
    if wrapped_lines:
        draw.text((margin, y), wrapped_lines[0], fill="black", font=font_bold)
        y += line_height * 2
        wrapped_lines = wrapped_lines[1:]

    for line in wrapped_lines:
        if y > page_height - margin:
            pages.append(img)
            img = Image.new("RGB", (page_width, page_height), "white")
            draw = ImageDraw.Draw(img)
            y = margin

        draw.text((margin, y), line, fill="black", font=font)
        y += line_height

    pages.append(img)

    # Convert image pages to PDF bytes
    pdf_bytes_io = io.BytesIO()
    pages[0].save(pdf_bytes_io, format="PDF", save_all=True, append_images=pages[1:])
    pdf_bytes_io.seek(0)

    filename = f"opspulse-executive-brief-{profile}.pdf"

    return StreamingResponse(
        pdf_bytes_io,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ==============================
# MCP WebSocket tools map
# ==============================
TOOLS = {
    "ping": lambda args: tool_ping(),
    "load_ops_data": lambda args: tool_load_ops_data(),
    "get_basic_summary": lambda args: tool_get_basic_summary(),
    "extract_themes": lambda args: tool_extract_themes(profile=args.get("profile", DEFAULT_PROFILE)),
    "score_health": lambda args: tool_score_health(profile=args.get("profile", DEFAULT_PROFILE)),
    "get_recommendations": lambda args: tool_get_recommendations(profile=args.get("profile", DEFAULT_PROFILE), top_n=int(args.get("top_n", 3))),
    "refresh_data": lambda args: tool_refresh_data(),
    "add_ticket": lambda args: tool_add_ticket(args.get("ticket", {})),
    "add_incident": lambda args: tool_add_incident(args.get("incident", {})),
}


@app.websocket("/mcp")
async def mcp_ws(websocket: WebSocket):
    await websocket.accept()

    await websocket.send_text(
        json.dumps(
            {
                "ok": True,
                "message": "OpsPulse MCP WebSocket endpoint ready",
                "endpoint": "/mcp",
            }
        )
    )

    try:
        while True:
            try:
                msg = await websocket.receive_json()
            except WebSocketDisconnect:
                break

            tool = msg.get("tool")
            args = msg.get("args", {})

            if tool not in TOOLS:
                await websocket.send_text(
                    json.dumps(
                        {
                            "ok": False,
                            "error": f"Unknown tool '{tool}'",
                            "available_tools": list(TOOLS.keys()),
                        }
                    )
                )
                continue

            try:
                result = TOOLS[tool](args)
                await websocket.send_text(json.dumps(result))
            except Exception as e:
                await websocket.send_text(json.dumps({"ok": False, "error": str(e), "tool": tool}))

    except WebSocketDisconnect:
        return
    except Exception:
        await websocket.close(code=1011)


@app.get("/")
def root():
    return JSONResponse({"ok": True, "message": "OpsPulse MCP running", "endpoint": "/mcp"})
