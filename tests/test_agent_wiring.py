"""The LangGraph agent is wired to the MCP server correctly, tested with a scripted fake LLM (no EVE needed).

The fake model emits a predefined tool call, then a final answer quoting the tool's numbers. The test
checks that the call reaches the real MCP server, the real result comes back into the graph, and the
transcript (JSON + Markdown) and grounding check are written.
"""
import json

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from mcp_server import agent

TINY = [9.425, 39.232, 9.440, 39.244]


class ScriptedLLM(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):   # the fake model ignores tool schemas
        return self


@pytest.mark.asyncio
async def test_agent_calls_mcp_tool_and_writes_grounded_transcript(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "OUT", tmp_path)
    call = AIMessage(content="", tool_calls=[{"name": "list_scenes", "id": "c1",
                                               "args": {"bbox": TINY, "start": "2024-07-01", "end": "2024-09-30"}}])
    # the final answer can only be grounded if the number really came back from the server: we read it first
    from mcp_server.server import list_scenes
    n = list_scenes(TINY, "2024-07-01", "2024-09-30")["n_acquisitions"]
    final = AIMessage(content=f"There are {n} Sentinel-2 acquisitions in summer 2024.")
    llm = ScriptedLLM(messages=iter([call, final]))

    t = await agent.ask("How many clear Sentinel-2 images in summer 2024 here?", llm=llm, name="test")

    assert t["tools_called"] == ["list_scenes"]
    result = next(s["result"] for s in t["steps"] if s["type"] == "tool_result")
    assert result["n_acquisitions"] == n
    assert t["grounding"]["grounded"] is True
    saved = json.loads((tmp_path / "transcript_test.json").read_text())
    assert saved["answer"] == final.content and (tmp_path / "transcript_test.md").exists()


def test_grounding_check_flags_invented_numbers():
    g = agent.grounding_check("Forest lost 2,211 ha and 37 cells", ['{"flagged_ha": 2211, "n": 40}'], "q")
    assert g["unsupported"] == ["37"]


def test_grounding_check_accepts_magnitude_of_negative_change():
    g = agent.grounding_check("NDVI fell by 0.231 (27.6%)", ['{"change_abs": -0.231, "change_pct": -27.6}'], "q")
    assert g["grounded"] is True
