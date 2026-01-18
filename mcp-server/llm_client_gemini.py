import os
import json
import asyncio
import requests
from dotenv import load_dotenv
import websockets

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
MCP_WS_URL = os.getenv("MCP_WS_URL", "ws://127.0.0.1:8000/mcp")

if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY in .env")


async def mcp_call(ws, tool: str, args: dict = {}):
    await ws.send(json.dumps({"tool": tool, "args": args}))
    resp = await ws.recv()
    return json.loads(resp)


def gemini_generate_report(payload: dict) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

    prompt = f"""
You are an operational intelligence analyst.
You are given outputs from an OpsPulse MCP server. Generate a concise executive report.

Rules:
- Output must be structured with headings
- Highlight top risks and why
- Highlight top themes
- Give 3 prioritized recommendations (P0/P1/P2)
- Mention health score and key drivers

MCP OUTPUT JSON:
{json.dumps(payload, indent=2)}
"""

    body = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }

    r = requests.post(url, json=body, timeout=60)
    r.raise_for_status()
    data = r.json()

    return data["candidates"][0]["content"]["parts"][0]["text"]


async def main():
    async with websockets.connect(MCP_WS_URL) as ws:
        # receive welcome msg
        welcome = await ws.recv()
        print("MCP CONNECTED:", welcome)

        # MCP workflow
        health = await mcp_call(ws, "score_health")
        themes = await mcp_call(ws, "extract_themes")
        clusters = await mcp_call(ws, "cluster_issues")
        recos = await mcp_call(ws, "get_recommendations", {"top_n": 3})

        payload = {
            "health": health,
            "themes": themes,
            "clusters": clusters,
            "recommendations": recos
        }

        print("\n🚀 Calling Gemini...\n")
        report = gemini_generate_report(payload)

        print("OPSPULSE EXECUTIVE REPORT")
        print(report)


if __name__ == "__main__":
    asyncio.run(main())
