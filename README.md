# OpsPulse MCP - Operational Intelligence Dashboard

OpsPulse MCP is a lightweight MVP Operational Intelligence Dashboard that helps teams track and analyze production issues in a structured way. It includes a FastAPI-based MCP server with a WebSocket endpoint where tools can be triggered to create and store incidents/tickets, load ops dataset stats, generate summaries, and extract profile-based themes (default vs infra). The system supports YAML-driven scoring profiles to dynamically compute health metrics and category breakdowns (like payments failures) based on keyword patterns in the data. To keep it fast and scalable, it includes caching with cache statistics visibility, along with health probes to verify service readiness. On top of that, it can generate an executive summary report using an LLM client and export it as a downloadable PDF using a stable image-to-PDF pipeline.
> This repo contains both backend and frontend implementations.

---

## What We Built

### Backend (MCP Server - FastAPI)
- FastAPI backend with MCP WebSocket endpoint: `WS /mcp`
- Profile-aware scoring system using YAML configs (`default`, `infra`)
- Persistent storage for:
  - tickets
  - incidents  
  stored in JSON (simple + portable)
- Snapshot breakdown endpoint to generate dashboard metrics
- Dynamic breakdown bar (example: **Payments failures**) based on keyword counts in tickets/incidents
- Cache stats endpoint (hit/miss + cache keys visibility)
- Health probes (basic service readiness information)
- LLM executive summary generation using **Gemini client**
- Report download endpoint with safe PDF generation:
  - renders text → images using Pillow
  - exports images → PDF (avoids unicode/font crashes)

---

### Frontend (Dashboard UI)
- Modern operational dashboard UI
- Displays:
  - Health Score gauge
  - Utilization / breakdown bar chart
  - Ticket + incident metrics
  - Areas to address
  - Executive report generator + download
- Uses API calls to backend to fetch snapshot + report data

---

## Tech Stack

**Backend**
- Python
- FastAPI
- WebSocket (MCP tool-style interaction)
- YAML configs
- JSON persistence
- Gemini LLM client
- Pillow (image-to-PDF export)

**Frontend**
- Next.js / React
- TailwindCSS
- ShadCN UI components

---

## Project Structure

```
opspulse-mcp/
  mcp-server/
    main.py
    llm_client_gemini.py
    cache.py
    config/
      default.yaml
      infra.yaml
    data/
      tickets.json
      incidents.json

  opspulse-frontend/
    src/
    package.json
    next.config.*
```

---

# Setup Guide

## 1) Backend Setup (mcp-server)

### Step 1: Go into backend folder
```bash
cd mcp-server
```

### Step 2: Create and activate Python venv
```bash
python -m venv .venv
source .venv/bin/activate
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

If `requirements.txt` isn't available:
```bash
pip install fastapi uvicorn pyyaml pillow python-dotenv
```

### Step 4: Run backend
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Backend runs at:
```
http://localhost:8000
```

---

# Manual MCP WebSocket Testing (CLI)

Once the backend is running, you can directly test the MCP WebSocket endpoint from terminal.

### WebSocket Endpoint
```
ws://127.0.0.1:8000/mcp
```

### Step 1: Install `wscat`
If you don’t already have it:
```bash
npm install -g wscat
```

### Step 2: Connect to the MCP socket
Open **Terminal 2** and run:
```bash
npx wscat -c ws://127.0.0.1:8000/mcp
```

On successful connection, the server returns a ready / connected message confirming the MCP endpoint is live.

### Step 3: MCP Tool Commands (Copy-Paste Ready)

**Tool 1: Ping**
```json
{"tool":"ping","args":{}}
```

**Tool 2: Load dataset counts**
```json
{"tool":"load_ops_data","args":{}}
```

**Tool 3: Basic summary**
```json
{"tool":"get_basic_summary","args":{}}
```

**Tool 4: Extract themes (profile-aware)**

Default profile:
```json
{"tool":"extract_themes","args":{"profile":"default"}}
```

Infra profile:
```json
{"tool":"extract_themes","args":{"profile":"infra"}}
```
---

## 2) Frontend Setup (opspulse-frontend)

### Step 1: Go into frontend folder
```bash
cd opspulse-frontend
```

### Step 2: Install dependencies
```bash
npm install
```

### Step 3: Configure env variables
Create:
```bash
opspulse-frontend/.env.local
```

Example:
```env
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```

### Step 4: Run frontend
```bash
npm run dev
```

Frontend runs at:
```
http://localhost:3000
```

---

# How to Use

## Create tickets/incidents
You can create new items from the UI.  
They persist into JSON files under:

- `mcp-server/data/tickets.json`
- `mcp-server/data/incidents.json`

These keyword-based entries dynamically affect chart categories like **Payments failures**.

---

## Generate report
From the dashboard:
- click **Generate Report**
- backend calls the LLM client
- report is displayed + can be downloaded as a PDF

---

## Notes
- New profiles can be added by creating a new YAML file inside `mcp-server/config/`
- The system is designed to stay simple and demo-friendly while still being scalable

---

## License
MIT 


