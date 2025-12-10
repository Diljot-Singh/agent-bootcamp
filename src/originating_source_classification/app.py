"""
Originating Source Classification Agent (Excel-driven)
Workbook: src/originating_source_classification/ApplicationCatalog.xlsx
Model: gemini-2.5-flash
Structured JSON output via Pydantic (local validation)
"""

import asyncio
import contextlib
import json
import signal
import sys
from enum import Enum
from typing import Dict, List, Optional

import gradio as gr
import pandas as pd
from dotenv import load_dotenv
from gradio.components.chatbot import ChatMessage
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionToolParam

from pydantic import BaseModel, Field

load_dotenv()

EXCEL_PATH = "src/originating_source_classification/ApplicationCatalog.xlsx"
MAX_TURNS = 5
AGENT_LLM_NAME = "gemini-2.5-flash"

# ---------- Pydantic schema ----------
class OriginatingSourceAnswer(str, Enum):
    YES = "Yes"
    NO = "No"
    MAYBE = "Maybe"
    NOT_FOUND = "Application Not Found"


class OriginatingSourceResponse(BaseModel):
    answer: OriginatingSourceAnswer = Field(..., description="Classification result.")
    reasoning: str = Field(..., description="Explanation referencing description and dependencies.")
    confidence_score: int = Field(..., ge=0, le=100, description="0–100 model confidence for a 'Yes' classification.")


# Pydantic v2: rebuild model to allow proper schema generation
OriginatingSourceResponse.model_rebuild()

# ---------- Load workbook ----------
try:
    apps_df = pd.read_excel(EXCEL_PATH, sheet_name="Application Descriptions")
    deps_df = pd.read_excel(EXCEL_PATH, sheet_name="Application Dependencies")
except Exception as e:
    raise RuntimeError(f"Failed to open expected sheets in {EXCEL_PATH}: {e}")

apps_df.columns = [c.strip() for c in apps_df.columns]
deps_df.columns = [c.strip() for c in deps_df.columns]

required_apps = ["App ID", "Application Name", "Description", "Business Line(s)", "Type (Contain External feed or not)"]
required_deps = ["From_AppID", "To_AppID", "Reason"]
missing_apps = [c for c in required_apps if c not in apps_df.columns]
missing_deps = [c for c in required_deps if c not in deps_df.columns]
if missing_apps:
    raise RuntimeError(f"Missing columns in 'Application Descriptions' sheet: {missing_apps}")
if missing_deps:
    raise RuntimeError(f"Missing columns in 'Application Dependencies' sheet: {missing_deps}")

apps_index: Dict[str, Dict] = {}
name_to_id: Dict[str, str] = {}
for _, r in apps_df.iterrows():
    aid = str(r["App ID"]).strip()
    name = str(r.get("Application Name", "")).strip()
    apps_index[aid] = {
        "app_id": aid,
        "application_name": name,
        "description": str(r.get("Description", "")),
        "business_lines": str(r.get("Business Line(s)", "")),
        "type_(contain_external_feed_or_not)": str(r.get("Type (Contain External feed or not)", "")),
    }
    name_to_id[name.lower()] = aid

incoming_map: Dict[str, List[Dict]] = {}
outgoing_map: Dict[str, List[Dict]] = {}
for _, r in deps_df.iterrows():
    frm = str(r["From_AppID"]).strip()
    to = str(r["To_AppID"]).strip()
    reason = str(r.get("Reason", ""))
    incoming_map.setdefault(to, []).append({"from_appid": frm, "to_appid": to, "reason": reason})
    outgoing_map.setdefault(frm, []).append({"from_appid": frm, "to_appid": to, "reason": reason})

# ---------- System prompt ----------
system_message: ChatCompletionSystemMessageParam = {
    "role": "system",
    "content": (
        """
        You are an agent that classifies whether a given application is an originating source for a specific business line. 
        You must always return a structured JSON object (validated locally via Pydantic) with fields:
          - answer: "Yes", "No", "Maybe", or "Application Not Found"
          - reasoning: explanation referencing Description and Dependencies
          - confidence_score: integer 0–100

        Follow this structured flow using tool calls:

        1. Extract application details:
           - Use the 'lookup_application' tool to retrieve the application's Description, Business Line(s), and Type (Contain External feed or not).
           - If no matching application exists, return "Application Not Found".

        2. Extract dependencies:
           - Use the 'lookup_dependencies' tool to identify all incoming and outgoing connections for the target application.
           - Incoming dependencies: applications that send data into the target application.
           - Outgoing dependencies: applications that receive data from the target application.

        3. Reasoning:
           - Analyze the application's description and dependencies to determine if it originates primary business data.
           - Classification rules:
             - YES (Application is clearly an originating source). If any of these conditions are satisfied, it should be classified as a “Yes”:
                User Entry Test: Humans or external systems input new primary business data into this system. Example: onboarding forms, application submission, manual data entry.
                New Entity Test: The system creates new business entities that did not exist before, such as: customers, accounts, transactions, policies, products.
                First Entry Test: The system is the first location where this data exists. It does not merely receive or ingest data from another system. Look for keywords: “originates”, “creates”, “new entry”, “initial capture”.
                Primary Data Test: The data generated is primary business data (e.g., customer records, trades, deposits, applications).
             - NO (Application is clearly not an originating source). Any of these conditions is sufficient to classify as “No”:
                No User Data Entry: The system does not accept new data entry from humans or external sources.
                No New Entity Creation: The system only processes or transforms existing data; it does not create new entities.
                Derived Data Only: The system only produces derived data such as aggregations, calculations, accounting entries, summaries, or reports.
                Consolidation / Master Data Management: The system consolidates or aggregates data from multiple sources. It manages master data rather than originating it.
             - MAYBE (Uncertain / Ambiguous):
                If none of the strict Yes conditions are satisfied, and none of the strict No conditions are triggered → classify as Maybe.
                This includes vague descriptions, ambiguous keywords, or mixed evidence from description and dependencies.
             - "APPLICATION NOT FOUND": No matching application exists.
           
            Notes for the model:
                Always use description first, then dependency context to confirm but not override the classification. Incorporate dependency context to support reasoning. For example:
                    - Receiving data from an originating source does not automatically make the target non-originating.
                    - Outgoing connections indicate where data flows but do not define origination by themselves.
                Reference the key evidence for each test in the reasoning field of the structured JSON.
                Keywords are guides; the model must reason using context rather than literal matches.

        4. Output:
           - Return ONLY the structured JSON.
           - Do not output intermediate reasoning or explanations outside the JSON.
           - Make the reasoning concise but reference the key elements in Description and Dependency details.
           - Always output strict single-line JSON without backticks, code fences, or unescaped quotes.
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
    if key in apps_index:
        return apps_index[key]
    kl = key.lower()
    if kl in name_to_id:
        return apps_index[name_to_id[kl]]
    for a in apps_index.values():
        if kl in (a.get("application_name") or "").lower():
            return a
    return None

def _get_dependencies(app_id: str) -> Dict:
    inc = incoming_map.get(app_id, [])
    out = outgoing_map.get(app_id, [])
    for dep_list in [inc, out]:
        for d in dep_list:
            d_from = d["from_appid"]
            d_to = d["to_appid"]
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
            reasoning_effort=None
        )

        message = completion.choices[0].message
        oai_messages.append(message)
        history.append(ChatMessage(role="assistant", content=message.content or ""))

        tool_calls = message.tool_calls
        if not tool_calls:
            # Validate structured JSON locally
            parsed_obj = None
            raw_text = message.content or ""
            try:
                safe_text = raw_text.replace('```', '').replace('\n', ' ').replace('\r', '')
                parsed = json.loads(safe_text)
                parsed_obj = OriginatingSourceResponse.model_validate(parsed)
            except Exception:
                parsed_obj = OriginatingSourceResponse.model_validate({
                    "answer": OriginatingSourceAnswer.MAYBE.value,
                    "reasoning": f"Model returned unparsable content. Raw: {raw_text.replace(chr(10), ' ').replace(chr(13),' ')}",
                    "confidence_score": 0
                })
            history.append(ChatMessage(role="assistant", content=json.dumps(parsed_obj.model_dump()), metadata={"title": "structured_response"}))
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
    title="Originating Source Agent (structured JSON)",
    type="messages",
    examples=[
        "Is Credit Card Processing an originating source for Retail Banking?",
        "Does PaymentsHub originate data for 'Wealth'?"
    ]
)

# ---------- Run ----------
if __name__ == "__main__":
    signal.signal(signal.SIGINT, _handle_sigint)
    try:
        demo.launch(share=True)
    finally:
        asyncio.run(_cleanup_clients())
