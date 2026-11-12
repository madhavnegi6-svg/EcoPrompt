"""EcoPrompt MCP server: right-size your AI model and see the energy and water you save.

Run locally for Claude Code (stdio):   python server.py --stdio
Run as a shareable connector (HTTP):   python server.py         -> http://HOST:PORT/mcp
"""

from __future__ import annotations

import hmac
import json
import os
import sys
from mcp.server.transport_security import TransportSecuritySettings
from pathlib import Path
from typing import Any, Literal

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response

from ecoprompt import advisor, stats
from ecoprompt.footprint import resolve_model

INSTRUCTIONS = """EcoPrompt is ON: the user installed it so that EVERY new request is checked before you answer.
It helps people use the smallest AI model that is good enough for each task.

RULE 1. When the user sends a new request or task (writing an essay, email, cover letter or post;
summarising, translating, explaining, analysing, coding, planning, researching...), your FIRST action
must be to call `recommend_model`, before any other tool (including document, search or file tools)
and before writing any answer. Pass a short, anonymised task description (include any length or format
the user asked for), your judgement of its complexity, and the model you are running as `current_model`.

RULE 2. Your first reply is ONLY the EcoPrompt check, kept short: the `nudge` line, then the `question`
(add `how_to_switch` if a model change is suggested). Nothing else: no extra explanation, no tips, no
clarifying questions. Then STOP and wait for the user's answer. Do not start the task in the same reply.

RULE 3. When the user answers, call `report_choice` if a model change was suggested, then do the task.
Do not call `recommend_model` again for the same task.

Skip EcoPrompt for small talk, questions about EcoPrompt itself, and follow-ups or edits to a task already
checked. Never include personal or confidential details in the task description."""

mcp = MCPServer(
    "EcoPrompt",
    title="EcoPrompt",
    description="Right-size your AI model per task and see the energy and water you save.",
    instructions=INSTRUCTIONS,
    version="0.2.0",
)

Complexity = Literal["simple", "moderate", "complex", "expert"]
OutputLength = Literal["short", "medium", "long"]
Region = Literal["global", "us", "eu", "nl"]


def _client_app(ctx: Context) -> str:
    """Which AI app is calling (Claude, Cursor, ...), from the MCP handshake or the User-Agent."""
    sources = (
        lambda: ctx.session.client_params.client_info.name,
        lambda: ctx.request_context.meta["io.modelcontextprotocol/clientInfo"]["name"],
        lambda: ctx.headers.get("user-agent"),
    )
    for source in sources:
        try:
            if name := source():
                return stats.normalize_client(name)
        except (AttributeError, KeyError, TypeError):
            continue
    return stats.normalize_client(None)


@mcp.tool(annotations=ToolAnnotations(title="Recommend a right-sized model", read_only_hint=False, open_world_hint=False))
def recommend_model(
    task: str,
    ctx: Context,
    complexity: Complexity | None = None,
    provider: str | None = None,
    current_model: str | None = None,
    output_length: OutputLength = "medium",
    region: Region = "global",
) -> dict[str, Any]:
    """REQUIRED FIRST STEP: call this before any other tool and before answering whenever the user sends
    a new task (essays, emails, summaries, translations, explanations, analysis, coding, planning).
    Returns the smallest AI model that can handle the task well, the energy, water and CO2 that saves,
    and a question for the user. Show only the nudge and the question, then STOP and wait for the user's
    answer before doing the task (see `next_step` in the result).

    Args:
        task: Short, anonymised description of the task, including any length or format the user asked for,
            e.g. "500-word essay on corporate restructuring" or "summarise a 2-page article".
        complexity: Your judgement of difficulty. simple = summarise/translate/reformat;
            moderate = emails, explanations, short drafts; complex = analysis, coding, reports;
            expert = research, multi-step reasoning, system design. Omit to let EcoPrompt guess.
        provider: AI provider in use, e.g. "claude", "chatgpt", "gemini". Keeps the advice within that provider.
        current_model: Model the user is on now, e.g. "claude-opus-5-5" or "gpt-4o".
        output_length: Expected answer length: short (~300 tokens), medium (~800), long (~2500).
        region: Where the data center likely is, for water and CO2 factors.
    """
    try:
        result = advisor.recommend(task, complexity, provider, current_model, output_length, region)
    except advisor.AdvisorError as e:
        raise ToolError(str(e)) from e
    s = result.pop("_stats")
    stats.record_event(
        "recommend_model",
        client=_client_app(ctx),
        provider=result["recommended_model"]["provider"],
        current_model=s["baseline"] if current_model else None,
        recommended=s["recommended"],
        complexity=result["task_complexity"],
        region=result["assumptions"]["region"],
        energy_wh_saved=s["saved"].energy_wh,
        water_ml_saved=s["saved"].water_ml,
        co2_g_saved=s["saved"].co2_g,
    )
    return result


@mcp.prompt(title="Eco check", description="Get a right-sized model suggestion before doing a task")
def eco_check(task: str) -> str:
    """Ask Claude to check EcoPrompt before doing the task."""
    return (
        "Before doing this task, call the EcoPrompt `recommend_model` tool with a short, anonymised "
        "description of it, your judgement of its complexity and the model you are running. Follow its "
        f"`next_step`: show the check, then wait for my answer.\n\nTask: {task}"
    )


@mcp.tool(annotations=ToolAnnotations(title="Report the user's model choice", read_only_hint=False, open_world_hint=False))
def report_choice(recommended_model: str, chosen_model: str, ctx: Context) -> dict[str, Any]:
    """Record, anonymously, whether the user switched to the recommended model after `recommend_model`.
    Call this as soon as the user answers the EcoPrompt question, then go on to do their task.

    Args:
        recommended_model: The model EcoPrompt recommended, e.g. "claude-haiku".
        chosen_model: The model the user decided to use.
    """
    rec_id, chosen_id = resolve_model(recommended_model), resolve_model(chosen_model)
    if rec_id is None or chosen_id is None:
        raise ToolError("Unknown model name. Use names like 'claude-haiku', 'gpt-4o-mini' or 'gemini-flash'.")
    outcome = "switched" if rec_id == chosen_id else "kept"
    stats.record_event("report_choice", client=_client_app(ctx), current_model=chosen_id, recommended=rec_id, outcome=outcome)
    message = "Thanks for choosing the lighter model!" if outcome == "switched" else "Noted. You kept your current model."
    return {"recorded": outcome, "message": message}


@mcp.tool(annotations=ToolAnnotations(title="Estimate a query's footprint", read_only_hint=True, open_world_hint=False))
def estimate_footprint(
    model: str, ctx: Context, output_tokens: int = 800, input_tokens: int = 500, region: Region = "global"
) -> dict[str, Any]:
    """Estimate the energy (Wh), water (mL) and CO2 (g) of one query to a given model.

    Args:
        model: Model name, e.g. "claude-haiku", "gpt-4o-mini", "gemini-2.5-pro".
        output_tokens: Tokens in the answer (about 0.75 words each).
        input_tokens: Tokens in the prompt and attached context.
        region: Where the data center likely is, for water and CO2 factors.
    """
    try:
        result = advisor.footprint_of(model, output_tokens, input_tokens, region)
    except advisor.AdvisorError as e:
        raise ToolError(str(e)) from e
    stats.record_event("estimate_footprint", client=_client_app(ctx), current_model=result["model"]["id"], region=result["assumptions"]["region"])
    return result


@mcp.tool(annotations=ToolAnnotations(title="Compare models for a task", read_only_hint=True, open_world_hint=False))
def compare_models(
    ctx: Context,
    task: str = "",
    complexity: Complexity | None = None,
    output_length: OutputLength = "medium",
    region: Region = "global",
) -> dict[str, Any]:
    """List all known models from lowest to highest footprint, marking which are suitable for the task.

    Args:
        task: Optional short description of the task.
        complexity: Optional difficulty (simple, moderate, complex, expert).
        output_length: Expected answer length: short, medium or long.
        region: Where the data center likely is, for water and CO2 factors.
    """
    try:
        result = advisor.compare(task, complexity, output_length, region)
    except advisor.AdvisorError as e:
        raise ToolError(str(e)) from e
    stats.record_event("compare_models", client=_client_app(ctx), complexity=result["task_complexity"], region=result["assumptions"]["region"])
    return result


@mcp.tool(annotations=ToolAnnotations(title="Community impact", read_only_hint=True, open_world_hint=False))
def community_impact(ctx: Context) -> dict[str, Any]:
    """Total potential energy, water and CO2 savings from all EcoPrompt recommendations so far (anonymous)."""
    stats.record_event("community_impact", client=_client_app(ctx))
    return stats.totals() | {"note": "Potential savings, assuming people follow the recommendations."}


_STATIC = Path(__file__).parent / "static"
_LANDING = (_STATIC / "index.html").read_text(encoding="utf-8")
_DASHBOARD = (_STATIC / "dashboard.html").read_text(encoding="utf-8")


def _is_admin(request: Request) -> bool:
    expected = os.environ.get("ECOPROMPT_ADMIN_KEY", "")
    given = request.headers.get("x-admin-key") or request.query_params.get("key") or ""
    return bool(expected) and hmac.compare_digest(given, expected)


@mcp.custom_route("/", methods=["GET"])
async def landing(request: Request) -> Response:
    return HTMLResponse(_LANDING)


@mcp.custom_route("/track", methods=["POST"])
async def track(request: Request) -> Response:
    """Landing-page events only (page_view, copy_url). Nothing about the visitor is stored."""
    body = await request.body()
    try:
        event = json.loads(body[:200]).get("event") if body else None
    except (ValueError, AttributeError):
        event = None
    if event not in stats.PAGE_EVENTS:
        return JSONResponse({"error": "unknown event"}, status_code=400)
    stats.record_event(event)
    return Response(status_code=204)


@mcp.custom_route("/stats", methods=["GET"])
async def stats_route(request: Request) -> Response:
    return JSONResponse(stats.totals(), headers={"Access-Control-Allow-Origin": "*"})


@mcp.custom_route("/dashboard", methods=["GET"])
async def dashboard(request: Request) -> Response:
    if not os.environ.get("ECOPROMPT_ADMIN_KEY"):
        return HTMLResponse("Dashboard disabled: set the ECOPROMPT_ADMIN_KEY environment variable.", status_code=404)
    if not _is_admin(request):
        return HTMLResponse("Forbidden: open /dashboard?key=YOUR_ADMIN_KEY", status_code=403)
    return HTMLResponse(_DASHBOARD, headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"})


@mcp.custom_route("/analytics", methods=["GET"])
async def analytics_route(request: Request) -> Response:
    if not _is_admin(request):
        return JSONResponse({"error": "forbidden"}, status_code=403)
    try:
        days = min(max(int(request.query_params.get("days", "30")), 1), 365)
    except ValueError:
        days = 30
    return JSONResponse(stats.analytics(days), headers={"Cache-Control": "no-store"})


@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> Response:
    return JSONResponse({"status": "ok"})


if __name__ == "__main__":
    if "--stdio" in sys.argv:
        mcp.run()
    else:
        mcp.run(
            transport="streamable-http",
            host=os.environ.get("HOST", "127.0.0.1"),
            port=int(os.environ.get("PORT", "8000")),
            stateless_http=True,
            json_response=True,
                        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
        )
