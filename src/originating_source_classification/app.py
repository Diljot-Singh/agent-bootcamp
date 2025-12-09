"""
Simplified Originating Source Classification Agent (Excel-driven)
Workbook: src/originating_source_classification/ApplicationCatalog.xlsx
"""

import asyncio
import contextlib
import json
import signal
import sys
from typing import Dict, List, Optional

import gradio as gr
import pandas as pd
from dotenv import load_dotenv
from gradio.components.chatbot import ChatMessage
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionToolParam

load_dotenv()

EXCEL_PATH = "src/originating_source_classification/ApplicationCatalog.xlsx"
MAX_TURNS = 5
AGENT_LLM_NAME = "gemini-2.5-pro"

# ---------- Load workbook ----------
try:
    apps_df = pd.read_excel(EXCEL_PATH, sheet_name="Application Descriptions")
    deps_df = pd.read_excel(EXCEL_PATH, sheet_name="Application Dependencies")
except Exception as e:
    raise RuntimeError(f"Failed to open expected sheets in {EXCEL_PATH}: {e}")

# Normalize column names
apps_df.columns = [c.strip() for c in apps_df.columns]
deps_df.columns = [c.strip() for c in deps_df.columns]

# Validate columns
required_apps = ["App ID", "Application Name", "Description", "Business Line(s)", "Type (Contain External feed or not)"]
required_deps = ["From_AppID", "To_AppID", "Reason"]
missing_apps = [c for c in required_apps if c not in apps_df.columns]
missing_deps = [c for c in required_deps if c not in deps_df.columns]
if missing_apps:
    raise RuntimeError(f"Missing columns in 'Application Descriptions' sheet: {missing_apps}")
if missing_deps:
    raise RuntimeError(f"Missing columns in 'Application Dependencies' sheet: {missing_deps}")

# Index apps by App ID
apps_index: Dict[str, Dict] = {}
name_to_id: Dict[str, str] = {}
for _, r in apps_df.iterrows():
    aid = str(r["App ID"]).strip()
    name = str(r.get("Application Name", "")).strip()
    apps_index[aid] = {
        "app_id": aid,
        "application_name": name,
        "description": str(r.get("Description", "")),
        "business_line": str(r.get("Business Line(s)", "")),
        "type": str(r.get("Type (Contain External feed or not)", "")),
    }
    name_to_id[name.lower()] = aid

# Build dependency maps
incoming_map: Dict[str, List[Dict]] = {}
outgoing_map: Dict[str, List[Dict]] = {}
for _, r in deps_df.iterrows():
    frm = str(r["From_AppID"]).strip()
    to = str(r["To_AppID"]).strip()
    reason = str(r.get("Reason", ""))
    incoming_map.setdefault(to, []).append({"from": frm, "to": to, "reason": reason})
    outgoing_map.setdefault(frm, []).append({"from": frm, "to": to, "reason": reason})

# ---------- System prompt ----------
system_message: ChatCompletionSystemMessageParam = {
    "role": "system",
    "content": (
        """
        You are an expert enterprise architecture reasoning agent.

        Primary evidence: Read and reason from the application's 'Description' field — this is the main signal for whether the app creates or originates new business data for the business line specified by the user.

        Dependencies: Use the dependencies sheet only as supporting context to identify what systems send data to or receive data from the target application. Treat them as helpful context, not strict rules.

        Careful reasoning: 
        - If the target application receives data from another system that is an originating source, this does not automatically make the target application non-originating. 
        - The target application may add additional new data elements, transformations, or generate new outputs that constitute originating new business data. 
        - Analyze whether the description of the target application explicitly mentions creation, lifecycle management, or generation of primary business data.
        - When connections or the role of the application are unclear, respond conservatively with "Maybe".

        Focus: User queries will specify a business line. Use this as the main focus and reason whether the application originates primary business data for that line.

        Final output: Yes, No, Maybe, or Application Not Found, followed by reasoning. 
        The reasoning must clearly outline the key elements in the description used to reach this conclusion and also mention the dependencies originating and leading into this application and their impact on the output.
    
        """
    )
}

# ---------- Tools ----------
tools: List[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "lookup_application",
            "description": "Return the application row given App ID or Name.",
            "parameters": {"type": "object", "properties": {"application_id_or_name": {"type": "string"}}, "required": ["application_id_or_name"]},
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_dependencies",
            "description": "Return incoming and outgoing dependencies for an App ID.",
            "parameters": {"type": "object", "properties": {"application_id": {"type": "string"}}, "required": ["application_id"]},
            "strict": True,
        },
    },
]

# ---------- Local tool implementations ----------
def _find_application(key: str) -> Optional[Dict]:
    key = str(key).strip()
    # Exact App ID match
    if key in apps_index:
        return apps_index[key]
    # Name match (case-insensitive)
    kl = key.lower()
    if kl in name_to_id:
        return apps_index[name_to_id[kl]]
    # Partial match
    for a in apps_index.values():
        if kl in (a.get("application_name") or "").lower():
            return a
    return None

def _get_dependencies(app_id: str) -> Dict:
    inc = incoming_map.get(app_id, [])
    out = outgoing_map.get(app_id, [])
    
    # Convert AppIDs to names and add their descriptions for reasoning clarity
    for dep_list in [inc, out]:
        for d in dep_list:
            d_from = d["from"]
            d_to = d["to"]
            from_app = apps_index.get(d_from, {})
            to_app = apps_index.get(d_to, {})
            d["from_name"] = from_app.get("application_name", d_from)
            d["from_description"] = from_app.get("description", "")
            d["to_name"] = to_app.get("application_name", d_to)
            d["to_description"] = to_app.get("description", "")
    return {"incoming": inc, "outgoing": out}

# ---------- OpenAI client ----------
async_openai_client = AsyncOpenAI()

async def _cleanup_clients() -> None:
    await async_openai_client.close()

def _handle_sigint(signum: int, frame: object) -> None:
    with contextlib.suppress(Exception):
        asyncio.get_event_loop().run_until_complete(_cleanup_clients())
    sys.exit(0)

# ---------- Main ReAct loop ----------
async def react_rag(query: str, history: List[ChatMessage]):
    oai_messages = [system_message, {"role": "user", "content": query}]
    for _ in range(MAX_TURNS):
        completion = await async_openai_client.chat.completions.create(
            model=AGENT_LLM_NAME,
            messages=oai_messages,
            tools=tools,
            reasoning_effort=None,
        )
        message = completion.choices[0].message
        oai_messages.append(message)
        history.append(ChatMessage(role="assistant", content=message.content or ""))
        tool_calls = message.tool_calls
        if tool_calls is None:
            yield history
            break
        for tool_call in tool_calls:
            fname = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            if fname == "lookup_application":
                q = args.get("application_id_or_name")
                res = _find_application(q)
                oai_messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(res)})
                history.append(ChatMessage(role="assistant", content=json.dumps(res), metadata={"title": "lookup_application", "q": q}))
            elif fname == "lookup_dependencies":
                aid = args.get("application_id")
                res = _get_dependencies(aid)
                oai_messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(res)})
                history.append(ChatMessage(role="assistant", content=json.dumps(res), metadata={"title": "lookup_dependencies", "app_id": aid}))
            else:
                oai_messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": "null"})
                history.append(ChatMessage(role="assistant", content="null"))
            yield history

# ---------- Gradio UI ----------
demo = gr.ChatInterface(
    react_rag,
    title="Originating Source Agent (simplified, Excel)",
    type="messages",
    examples=["Is APP123 an originating source for Retail Banking?", "Does PaymentsHub originate data for 'Wealth'?"]
)

# ---------- Run ----------
if __name__ == "__main__":
    signal.signal(signal.SIGINT, _handle_sigint)
    try:
        demo.launch(share=True)
    finally:
        asyncio.run(_cleanup_clients())
