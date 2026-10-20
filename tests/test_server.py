"""The MCP server starts over stdio, exposes three tools with schemas, and runs real Sentinel-2 queries.

No LLM is involved: this checks the server any MCP client (EVE, Claude, ...) would talk to.
Network access to Planetary Computer is required; areas are tiny so the test takes about a minute.
"""
import json
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]
SERVER = StdioServerParameters(command=sys.executable, args=["-m", "mcp_server.server"], cwd=str(ROOT))
TINY = [9.425, 39.232, 9.440, 39.244]   # ~1.3 x 1.3 km around priority cell #1, Sette Fratelli


async def _call(name, args=None):
    async with stdio_client(SERVER) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            if name is None:
                return await s.list_tools()
            out = await s.call_tool(name, args)
            return json.loads(out.content[0].text)


@pytest.mark.asyncio
async def test_three_tools_with_schemas():
    tools = (await _call(None)).tools
    names = {t.name for t in tools}
    assert names == {"list_scenes", "compare_vegetation", "dieback_alerts"}
    for t in tools:
        assert t.description and t.inputSchema["type"] == "object"


@pytest.mark.asyncio
async def test_list_scenes_real_data():
    res = await _call("list_scenes", {"bbox": TINY, "start": "2024-07-01", "end": "2024-09-30"})
    assert res["n_acquisitions"] >= 6


@pytest.mark.asyncio
async def test_compare_vegetation_real_data():
    res = await _call("compare_vegetation", {"bbox": TINY, "period_a": "2022-07-01/2022-09-30",
                                             "period_b": "2024-07-01/2024-09-30", "index": "ndmi"})
    assert "error" not in res, res
    assert res["forest_ha"] > 0 and res["median_b"] < res["median_a"]   # 2024 drought: NDMI fell


@pytest.mark.asyncio
async def test_invalid_input_returns_readable_error():
    res = await _call("compare_vegetation", {"bbox": [0, 0, 10, 10], "period_a": "2022-07-01/2022-09-30",
                                             "period_b": "2024-07-01/2024-09-30"})
    assert "limit" in res["error"]


@pytest.mark.asyncio
async def test_z_threshold_sign_does_not_matter():
    """The docstring says |z_threshold|: a positive value (which some LLMs pass) must not flag the whole forest."""
    neg = await _call("dieback_alerts", {"bbox": TINY, "z_threshold": -2.0})
    pos = await _call("dieback_alerts", {"bbox": TINY, "z_threshold": 2.0})
    assert "error" not in neg and "error" not in pos, (neg, pos)
    assert pos["flagged_ha"] == neg["flagged_ha"]
    assert pos["priority_cells_25ha"] == neg["priority_cells_25ha"]
