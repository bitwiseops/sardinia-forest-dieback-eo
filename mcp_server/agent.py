"""A LangGraph agent that answers plain-English questions by calling the MCP pipeline tools.

    python -m mcp_server.agent "After the 2024 drought, where should foresters inspect first in bbox [...]?"
    python -m mcp_server.agent --model qwen2.5:7b-instruct "..."      # any Ollama model with tool calling

Default model: EVE-Instruct (eve-esa/EVE-Instruct-GGUF-Q4_K_M), the Earth-observation LLM from ESA Phi-lab /
Pi School, served locally by Ollama. The MCP server is launched as a subprocess over stdio, its tools are
loaded with langchain-mcp-adapters, and a ReAct graph (LangGraph) loops model -> tool -> model until the answer.
Every run writes a transcript (JSON + Markdown) with the model digest, the exact tool calls and raw results,
the answer, and an automatic check that every number in the answer comes from a tool result.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "mcp"
EVE = "hf.co/eve-esa/EVE-Instruct-GGUF-Q4_K_M"
OLLAMA = "http://localhost:11434"

SYSTEM = """You are an Earth-observation analyst. You have tools that run a Sentinel-2 analysis pipeline.
- For any number about vegetation, forests, drought or satellite scenes, call a tool. Never estimate numbers yourself.
- Copy coordinates, polygons and dates from the question exactly into the tool arguments.
- Answer in at most five sentences, quoting the numbers returned by the tools and the file paths they wrote.
- Do not compute new numbers yourself (distances, sums, averages): quote only numbers the tools returned.
- If the question is not about observing an area of the Earth, answer briefly without calling any tool."""

SERVER = {"pipeline": {"transport": "stdio", "command": sys.executable,
                       "args": ["-m", "mcp_server.server"], "cwd": str(ROOT)}}


def model_info(model: str) -> dict:
    """Name and digest of the Ollama model, so a transcript says exactly which weights produced it."""
    try:
        import httpx
        tags = httpx.get(f"{OLLAMA}/api/tags", timeout=5).json()["models"]
        full = model if ":" in model.split("/")[-1] else f"{model}:latest"   # Ollama lists 'name:tag'
        m = next(t for t in tags if t["name"] in (model, full) or t["model"] in (model, full))
        return {"name": model, "digest": m["digest"], "size_gb": round(m["size"] / 1e9, 2),
                "quantization": m.get("details", {}).get("quantization_level")}
    except Exception:
        return {"name": model, "digest": None}


# 2,211 or 2211 or 2211.5 or -0.84 (English formatting: comma = thousands separator)
NUM = re.compile(r"(?<![\w.])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def _numbers(text: str) -> list[str]:
    text = re.sub(r"(?<=\d)[   ](?=\d{3}\b)", "", text)   # 2 211 -> 2211
    return [n.replace(",", "") for n in NUM.findall(text)]


def _text(content) -> str:
    """Tool results arrive as a string or as a list of content blocks ({'type': 'text', 'text': ...})."""
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)


def grounding_check(answer: str, tool_outputs: list[str], question: str) -> dict:
    """Every number in the answer must appear in a tool result (allowing rounding) or in the question."""
    pool = [float(x) for t in tool_outputs for x in _numbers(t)]
    asked = {float(x) for x in _numbers(question)}
    unsupported = []
    for s in _numbers(answer):
        v = float(s)
        dec = len(s.split(".")[1]) if "." in s else 0
        tol = 0.5 * 10 ** -dec + 1e-9
        # a 'decrease of 0.231' quotes a tool value of -0.231: compare magnitudes
        if v in asked or any(abs(abs(v) - abs(p)) <= tol for p in pool):
            continue
        unsupported.append(s)
    return {"numbers_in_answer": len(_numbers(answer)), "unsupported": unsupported, "grounded": not unsupported}


def transcript_from(question: str, messages, model: dict, seconds: float) -> dict:
    steps, tool_texts = [], []
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                steps.append({"type": "tool_call", "tool": tc["name"], "args": tc["args"]})
        elif isinstance(m, ToolMessage):
            txt = _text(m.content)
            tool_texts.append(txt)
            try:
                payload = json.loads(txt)
            except Exception:
                payload = txt
            steps.append({"type": "tool_result", "tool": m.name, "result": payload})
    answer = messages[-1].content if isinstance(messages[-1], AIMessage) else ""
    return {"question": question, "model": model, "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seconds": round(seconds, 1), "steps": steps, "answer": answer,
            "tools_called": [s["tool"] for s in steps if s["type"] == "tool_call"],
            "grounding": grounding_check(answer, tool_texts, question)}


def to_markdown(t: dict) -> str:
    lines = [f"**Question.** {t['question']}", "",
             f"*Model:* `{t['model']['name']}` (digest `{(t['model'].get('digest') or 'n/a')[:12]}`), "
             f"{t['utc']}, {t['seconds']} s", ""]
    for s in t["steps"]:
        if s["type"] == "tool_call":
            lines += [f"→ **calls `{s['tool']}`** with", "```json", json.dumps(s["args"], indent=1), "```"]
        else:
            r = json.dumps(s["result"], indent=1)
            lines += [f"← `{s['tool']}` returned", "```json", r if len(r) < 2500 else r[:2500] + "\n...", "```"]
    if not t["steps"]:
        lines += ["*(no tool called)*"]
    g = t["grounding"]
    lines += ["", f"**Answer.** {t['answer']}", "",
              f"*Grounding check:* {g['numbers_in_answer']} numbers in the answer, "
              + ("all found in tool results." if g["grounded"] else f"not found in tool results: {g['unsupported']}")]
    return "\n".join(lines)


def _print_live(m, elapsed: float):
    if isinstance(m, AIMessage) and m.tool_calls:
        for tc in m.tool_calls:
            print(f"[{elapsed:5.1f}s] model calls {tc['name']}({json.dumps(tc['args'])})", flush=True)
            print("        MCP server runs the Sentinel-2 pipeline...", flush=True)
    elif isinstance(m, ToolMessage):
        txt = _text(m.content)
        try:
            r = json.loads(txt)
            short = {k: v for k, v in r.items() if not isinstance(v, (list, dict))}
        except Exception:
            short = txt[:300]
        print(f"[{elapsed:5.1f}s] {m.name} returned {json.dumps(short)}", flush=True)
    elif isinstance(m, AIMessage):
        print(f"[{elapsed:5.1f}s] answer:\n{m.content}", flush=True)


async def ask(question: str, model: str = EVE, llm=None, name: str | None = None, recursion_limit: int = 8,
              live: bool = False) -> dict:
    """Run one question end to end and save the transcript. `llm` overrides the Ollama model (tests)."""
    client = MultiServerMCPClient(SERVER)
    tools = await client.get_tools()
    if llm is None:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model=model, base_url=OLLAMA, temperature=0, num_ctx=8192)
        info = model_info(model)
    else:
        info = {"name": type(llm).__name__, "digest": None}
    agent = create_react_agent(llm, tools, prompt=SYSTEM)
    t0 = time.time()
    inp, cfg = {"messages": [HumanMessage(question)]}, {"recursion_limit": recursion_limit}
    if not live:
        messages = (await agent.ainvoke(inp, config=cfg))["messages"]
    else:  # print each step as it happens (model decision, tool result, answer)
        messages = [inp["messages"][0]]
        print(f"[user] {question}\n[{info['name']}] thinking...", flush=True)
        async for update in agent.astream(inp, config=cfg, stream_mode="updates"):
            for node in update.values():
                for m in node.get("messages", []):
                    messages.append(m)
                    _print_live(m, time.time() - t0)
    t = transcript_from(question, messages, info, time.time() - t0)
    if live:
        g = t["grounding"]
        print(f"[check] {g['numbers_in_answer']} numbers in the answer, "
              + ("all found in tool results" if g["grounded"] else f"NOT in tool results: {g['unsupported']}"), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f"transcript_{name or datetime.now().strftime('%Y%m%d_%H%M%S')}"
    stem.with_suffix(".json").write_text(json.dumps(t, indent=1, default=str))
    stem.with_suffix(".md").write_text(to_markdown(t))
    return t


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("question")
    p.add_argument("--model", default=EVE)
    p.add_argument("--name", default=None, help="transcript file name suffix")
    p.add_argument("--live", action="store_true", help="print each step as it happens")
    a = p.parse_args()
    t = asyncio.run(ask(a.question, a.model, name=a.name, live=a.live))
    if not a.live:
        print(to_markdown(t))


if __name__ == "__main__":
    main()
