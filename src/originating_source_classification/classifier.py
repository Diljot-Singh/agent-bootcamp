"""
Batch Originating Source Classification Agent using Agents + Langfuse
Workbook: src/originating_source_classification/ApplicationCatalog.xlsx
Outputs structured JSON for all applications
"""

import re
import asyncio
import contextlib
import signal
import sys
import json
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
import agents

from pydantic import BaseModel, Field
from src.utils.langfuse.shared_client import langfuse_client
from src.utils import set_up_logging, setup_langfuse_tracer

load_dotenv(verbose=True)
set_up_logging()

EXCEL_PATH = "src/originating_source_classification/ApplicationCatalog.xlsx"
OUTPUT_EXCEL_PATH = "src/originating_source_classification/results.xlsx"
AGENT_LLM_NAME = "gemini-2.5-flash"

# ---------- Pydantic schema ----------
class OriginatingSourceAnswer(str, Enum):
    YES = "Yes"
    NO = "No"
    MAYBE = "Maybe"

class OriginatingSourceResponse(BaseModel):
    answer: OriginatingSourceAnswer = Field(..., description="Classification result.")
    reasoning: str = Field(..., description="Explanation referencing Description and Dependencies.")
    confidence_score: int = Field(..., ge=0, le=100, description="0–100 model confidence for a 'Yes' classification.")

OriginatingSourceResponse.model_rebuild()

# ---------- Load workbook ----------
apps_df = pd.read_excel(EXCEL_PATH, sheet_name="Application Descriptions")
deps_df = pd.read_excel(EXCEL_PATH, sheet_name="Application Dependencies")

apps_df.columns = [c.strip() for c in apps_df.columns]
deps_df.columns = [c.strip() for c in deps_df.columns]

# Index apps
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
        "type": str(r.get("Type (Contain External feed or not)", ""))
    }
    name_to_id[name.lower()] = aid

incoming_map: Dict[str, List[Dict]] = {}
outgoing_map: Dict[str, List[Dict]] = {}
for _, r in deps_df.iterrows():
    frm = str(r["From_AppID"]).strip()
    to = str(r["To_AppID"]).strip()
    reason = str(r.get("Reason", ""))
    incoming_map.setdefault(to, []).append({"from": frm, "to": to, "reason": reason})
    outgoing_map.setdefault(frm, []).append({"from": frm, "to": to, "reason": reason})

# ---------- Local tool implementations ----------
def _find_application_sync(key: str) -> Optional[Dict]:
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

def _get_dependencies_sync(app_id: str) -> Dict:
    inc = incoming_map.get(app_id, [])
    out = outgoing_map.get(app_id, [])
    # augment from/to with names & descriptions
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

# Async wrappers for agents.function_tool
async def lookup_application_tool(application_id_or_name: str) -> Dict:
    app = _find_application_sync(application_id_or_name)
    if not app:
        return {}
    return {
        "App ID": app["app_id"],
        "Application Name": app["application_name"],
        "Description": app["description"],
        "Business Line(s)": app["business_line"],
        "Type (Contain External feed or not)": app["type"]
    }

async def get_dependencies_tool(application_id: str) -> Dict:
    return _get_dependencies_sync(application_id)

# ---------- OpenAI client ----------
async_openai_client = AsyncOpenAI()

async def _cleanup_clients() -> None:
    with contextlib.suppress(Exception):
        await async_openai_client.close()

def _handle_sigint(signum: int, frame: object) -> None:
    with contextlib.suppress(Exception):
        asyncio.get_event_loop().run_until_complete(_cleanup_clients())
    sys.exit(0)

# ---------- Create Agent ----------
lookup_tool = agents.function_tool(lookup_application_tool)
lookup_tool.name = "lookup_application"
lookup_tool.description = "Return full application row (no gold label)"

deps_tool = agents.function_tool(get_dependencies_tool)
deps_tool.name = "get_dependencies"
deps_tool.description = "Return dependencies for an application, including from/to descriptions"

classification_agent = agents.Agent(
    name="ClassificationAgent",
    instructions="""
        You are an agent that classifies whether a given application is an originating source for a specific business line. 
        You must always return a structured JSON object (validated locally via Pydantic) with fields:
          - answer: "Yes", "No", "Maybe"
          - reasoning: explanation referencing Description and Dependencies
          - confidence_score: integer 0–100

        Follow this structured flow using tool calls:

        1. Extract application details:
           - Use the 'lookup_application' tool to retrieve the application's Description, Business Line(s), and Type (Contain External feed or not).

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
                Incorrect Business Line: The application’s business line does not match the target business line in the query.
             - MAYBE (Uncertain / Ambiguous):
                If none of the strict Yes conditions are satisfied, and none of the strict No conditions are triggered → classify as Maybe.
                This includes vague descriptions, ambiguous keywords, or mixed evidence from description and dependencies.
           
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
        """,
    tools=[lookup_tool, deps_tool],
    model=agents.OpenAIChatCompletionsModel(model=AGENT_LLM_NAME, openai_client=async_openai_client)
)


def clean_model_json(text: str) -> str:
    # Remove Python-style escaping of single quotes: \' -> '
    text = text.replace("\\'", "'")

    # Remove Tab escapes like \t inside reasoning (if any)
    # You can keep \n because valid JSON strings allow them only if escaped \\n
    # but Gemini sometimes outputs raw control chars.
    text = text.replace("\t", " ")

    # Fix stray backslashes before non-JSON escape characters
    text = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)

    return text

# ---------- Batch classification with gold label check ----------
async def classify_all_applications(business_line: str):
    results = []

    setup_langfuse_tracer()
    with langfuse_client.start_as_current_span(name="Batch-Originating-Source") as span:
        for app in apps_index.values():
            if not (business_line in app['business_line']):
                continue
            query = f"Is {app['app_id']} an originating source for '{business_line}'"
            span.update(input=query)

            result_stream = agents.Runner.run_streamed(classification_agent, input=query)
            async for _ in result_stream.stream_events():
                pass  # drain intermediate fragments

            full_content = result_stream.final_output.strip()

            # Extract JSON block
            start_idx = full_content.find("```json")
            end_idx = full_content.find("```", start_idx + 7)
            if start_idx != -1 and end_idx != -1:
                json_text = full_content[start_idx + 7:end_idx].strip()
            else:
                json_text = full_content
            json_text = clean_model_json(json_text)
            # Safe parsing
            try:
                parsed = json.loads(json_text)
                structured_obj = OriginatingSourceResponse.model_validate(parsed)
            except Exception as e:
                structured_obj = OriginatingSourceResponse.model_validate({
                    "answer": OriginatingSourceAnswer.MAYBE.value,
                    "reasoning": f"Parsing failed: {str(e)} | Raw content: {full_content}",
                    "confidence_score": 0
                })

            # Compare against gold label
            gold_label_raw = apps_df.loc[apps_df["App ID"] == app["app_id"], "Possible Originating Data Source"].values[0]
            gold_label = str(gold_label_raw).strip().capitalize()  # "Yes", "No", or "Maybe"

            # Check if agent's answer matches gold label
            correct = structured_obj.answer.value == gold_label

            results.append({
                "App ID": app["app_id"],
                "Application Name": app["application_name"],
                "Business Line": app["business_line"],
                "Answer": structured_obj.answer.value,
                "Reasoning": structured_obj.reasoning,
                "Confidence Score": structured_obj.confidence_score,
                "Gold Label": gold_label,
                "Correct": correct
            })
        span.update(output=f"{len(results)} applications classified")

    df = pd.DataFrame(results)

    # ---------- Classification Report ----------
    # Convert enums to raw string values
    df["Answer"] = df["Answer"].astype(str)
    df["Gold Label"] = df["Gold Label"].astype(str)

    # Compute confusion matrix values
    labels = ["Yes", "No", "Maybe"]
    report_rows = []
    for pred in labels:
        row = {"Predicted": pred}
        for gold in labels:
            count = df[(df["Answer"] == pred) & (df["Gold Label"] == gold)].shape[0]
            row[gold] = count
        report_rows.append(row)

    report_df = pd.DataFrame(report_rows)

    # Add totals
    report_df["Total Predicted"] = report_df[labels].sum(axis=1)
    gold_totals = df["Gold Label"].value_counts().reindex(labels, fill_value=0)
    report_df.loc[len(report_df)] = (
        ["Total Actual"] + gold_totals.tolist() + [gold_totals.sum()]
    )

    print("\n=== Classification Report ===")
    print(report_df.to_string(index=False))
    
    df.to_excel(OUTPUT_EXCEL_PATH, index=False)

    # Optionally print summary accuracy
    total = len(results)
    correct_count = sum(r["Correct"] for r in results)
    accuracy = 100 * correct_count / total if total > 0 else 0
    print(f"Classification completed. Results saved to {OUTPUT_EXCEL_PATH}")
    print(f"Accuracy vs Gold Label: {correct_count}/{total} = {accuracy:.2f}%")

    return df


# ---------- Run ----------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch classify applications")
    parser.add_argument("--business-line", type=str, required=True, help="Business line to classify")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        asyncio.run(classify_all_applications(args.business_line))
    finally:
        asyncio.run(_cleanup_clients())
