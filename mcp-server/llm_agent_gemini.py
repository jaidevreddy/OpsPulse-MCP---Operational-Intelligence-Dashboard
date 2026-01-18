import os
import json
import asyncio
import requests
from dotenv import load_dotenv
import websockets

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")
MCP_WS_URL = os.getenv("MCP_WS_URL", "ws://127.0.0.1:8000/mcp")

if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY in .env")


# -----------------------------
# MCP Client Helpers
# -----------------------------
async def mcp_call(ws, tool: str, args: dict = None):
    if args is None:
        args = {}
    await ws.send(json.dumps({"tool": tool, "args": args}))
    resp = await ws.recv()
    return json.loads(resp)


# -----------------------------
# Gemini Call Helper
# -----------------------------
def gemini_chat(messages: list[dict]) -> dict:
    """
    messages format:
    [
      {"role":"user","content":"..."},
      {"role":"assistant","content":"..."}
    ]
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

    # Convert our messages into Gemini format
    contents = []
    for m in messages:
        role = "user" if m["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})

    body = {
        "contents": contents,
        "generationConfig": {
            "temperature": 0.2
        }
    }

    r = requests.post(url, json=body, timeout=60)
    r.raise_for_status()
    return r.json()


def extract_text(gemini_response: dict) -> str:
    return gemini_response["candidates"][0]["content"]["parts"][0]["text"]


# -----------------------------
# Tool Selection Parser
# -----------------------------
def parse_tool_request(text: str):
    """
    Gemini will output something like:
    TOOL_CALL: {"tool":"score_health","args":{}}

    Or if finished:
    FINAL_REPORT: ...
    """
    text = text.strip()

    if text.startswith("TOOL_CALL:"):
        payload = text.replace("TOOL_CALL:", "").strip()
        return ("tool", json.loads(payload))

    if text.startswith("FINAL_REPORT:"):
        payload = text.replace("FINAL_REPORT:", "").strip()
        return ("final", payload)

    return ("unknown", text)


# -----------------------------
# Main Agent Loop
# -----------------------------
async def main():
    async with websockets.connect(MCP_WS_URL) as ws:
        welcome = await ws.recv()
        print("MCP CONNECTED:", welcome)

        # Define tools visible to Gemini (keep short but accurate)
        tool_catalog = {
            "score_health": "Compute overall operational health score and drivers",
            "extract_themes": "Extract major operational themes from tickets/incidents",
            "cluster_issues": "Cluster issues into grouped categories with labels",
            "get_recommendations": "Return prioritized recommendations based on current state",
            "get_basic_summary": "Basic stats summary of current dataset",
        }

        system_prompt = f"""
You are OpsPulse, an operational intelligence agent.

You can call tools from an MCP server.

Available tools:
{json.dumps(tool_catalog, indent=2)}

RULES:
- If you need more information, call a tool.
- Always output tool calls as:
TOOL_CALL: {{"tool":"<tool_name>","args":{{...}}}}

- When you have enough information, output:
FINAL_REPORT: <your report>

Report requirements:
- Health score + drivers
- Top risks (P0)
- Themes
- Clusters summary
- 3 recommendations (P0/P1/P2)
"""

        messages = [{"role": "user", "content": system_prompt}]

        # Agent scratchpad
        observations = {}

        # Loop tool calling
        for step in range(1, 8):  # max 7 tool calls
            user_prompt = f"""
Current observations JSON:
{json.dumps(observations, indent=2)}

Decide next best step.
"""
            messages.append({"role": "user", "content": user_prompt})

            gemini_resp = gemini_chat(messages)
            out_text = extract_text(gemini_resp)

            print(f"\n--- GEMINI STEP {step} ---")
            print(out_text)

            kind, payload = parse_tool_request(out_text)

            if kind == "tool":
                tool = payload["tool"]
                args = payload.get("args", {})

                print(f"\n Calling MCP tool: {tool} args={args}")
                result = await mcp_call(ws, tool, args)

                observations[tool] = result

                # send tool result back to Gemini
                messages.append({
                    "role": "assistant",
                    "content": f"Tool result for {tool}:\n{json.dumps(result, indent=2)}"
                })

            elif kind == "final":

                print("OPSPULSE FINAL REPORT")
                print(payload)
                return

            else:
                # If Gemini didn't follow format, nudge it
                messages.append({
                    "role": "assistant",
                    "content": "You must respond using TOOL_CALL or FINAL_REPORT format only."
                })

        print("\nAgent reached max steps without FINAL_REPORT")


if __name__ == "__main__":
    asyncio.run(main())
