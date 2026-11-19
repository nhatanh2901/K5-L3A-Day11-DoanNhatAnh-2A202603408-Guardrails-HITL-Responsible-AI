"""Retry provider failures using the original targets and prompts, preserving evidence."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents.agent import create_red_agent_default
from agents.guards_agent import create_red_agent_advance
from attacks.attacks import run_attacks, save_attack_results, write_run_attack_json
from core.config import get_llm_provider, get_model_name


async def retry(target: str, timeout: float):
    path = ROOT / "outputs" / "attack_results.json"
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if (evidence.get("llm_provider"), evidence.get("llm_model")) != (get_llm_provider(), get_model_name()):
        raise ValueError("Provider/model changed. Run src/main.py --part 4 for fresh evidence.")
    groups = [
        ("red", "unsafe_attacks", "red_default", create_red_agent_default),
        ("advance", "guards_attacks", "red_advance", create_red_agent_advance),
    ]
    for label, key, name, factory in groups:
        rows = evidence[key]
        failed = [row for row in rows if row.get("layer") in ("error", "nemo_error")]
        if not failed or target not in ("all", label):
            continue
        agent, runner = factory()
        replacements = await run_attacks(agent, runner, prompts=failed, target_name=name,
                                         save_json=False, request_timeout_seconds=timeout)
        by_id = {row["id"]: row for row in replacements}
        evidence[key] = [by_id.get(row["id"], row) for row in rows]
        write_run_attack_json(evidence[key], target_name=name)
        save_attack_results(unsafe_results=evidence["unsafe_attacks"],
                            guards_results=evidence["guards_attacks"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["red", "advance", "all"], default="all")
    parser.add_argument("--timeout", type=float, default=90)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    asyncio.run(retry(args.target, args.timeout))
