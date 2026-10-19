"""The three demonstration questions (tool grounding, our use case, out of scope) run through the agent.

    python -m mcp_server.demo_questions [--model <ollama model>]
"""
import argparse
import asyncio

from mcp_server.agent import EVE, ask, to_markdown

POLYGON = ('{"type": "Polygon", "coordinates": [[[9.41, 39.225], [9.445, 39.225], [9.445, 39.25], '
           '[9.41, 39.25], [9.41, 39.225]]]}')
QUESTIONS = {
    "q1_compare_polygon": f"Compare vegetation in this polygon between summer 2022 and summer 2024 and tell me "
                          f"where it dropped: {POLYGON}",
    "q2_dieback_priorities": "After the 2024 drought, which forest areas in the Sette Fratelli massif, bbox "
                             "[9.33, 39.19, 9.52, 39.33], should the forest service inspect first? "
                             "Use summer 2025 as the recovery year.",
    "q3_out_of_scope": "Who painted the ceiling of the Sistine Chapel?",
}


async def run_all(model: str):
    out = {}
    for name, q in QUESTIONS.items():
        out[name] = await ask(q, model, name=name)
        print(to_markdown(out[name]), "\n\n" + "=" * 80 + "\n")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=EVE)
    asyncio.run(run_all(p.parse_args().model))
