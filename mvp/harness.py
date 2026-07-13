"""Harness: run rules through the verifier across tasks and seeds, log everything.

Every rollout is a JSONL record (spec section 10 deliverable 4: full logs).
"""

import dataclasses
import json
import os
import statistics
from collections import defaultdict

from .rules import DeltaNoHidden
from .verifier import Task, TrialResult, make_tasks, run_trial

DEFAULT_BUDGET = 400_000
DEFAULT_SEEDS = 10


def run_matrix(
    rules,
    tasks: list[Task] | None = None,
    seeds: int = DEFAULT_SEEDS,
    budget: int = DEFAULT_BUDGET,
    log_path: str | None = None,
    quiet: bool = False,
) -> list[TrialResult]:
    tasks = tasks or make_tasks()
    results: list[TrialResult] = []
    log_f = None
    if log_path:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        log_f = open(log_path, "a")
    try:
        for rule in rules:
            arch_override_needed = isinstance(rule, DeltaNoHidden)
            for task in tasks:
                override = task.no_hidden_arch if arch_override_needed else None
                for seed in range(seeds):
                    r = run_trial(rule, task, seed, budget=budget, arch_override=override)
                    results.append(r)
                    if log_f:
                        log_f.write(json.dumps(dataclasses.asdict(r)) + "\n")
                if not quiet:
                    sub = [r for r in results if r.rule == rule.name and r.task == task.name]
                    rate = sum(r.solved for r in sub) / len(sub)
                    print(f"  {rule.name:24s} {task.name:12s} solve-rate {rate:.0%}")
    finally:
        if log_f:
            log_f.close()
    return results


def summarize(results: list[TrialResult]) -> dict:
    """Per (rule, task): solve rate + median units-to-criterion among solves."""
    grouped = defaultdict(list)
    for r in results:
        grouped[(r.rule, r.task)].append(r)
    summary = {}
    for (rule, task), rs in grouped.items():
        solved = [r for r in rs if r.solved]
        summary[(rule, task)] = {
            "n": len(rs),
            "solve_rate": len(solved) / len(rs),
            "median_units": statistics.median(r.units_to_criterion for r in solved) if solved else None,
            "median_final_mse": statistics.median(r.final_mse for r in rs),
            "errors": sum(1 for r in rs if r.error),
        }
    return summary


def format_summary_table(summary: dict) -> str:
    rules = sorted({k[0] for k in summary})
    tasks = sorted({k[1] for k in summary})
    lines = [
        "| rule | " + " | ".join(f"{t} (rate / median units)" for t in tasks) + " |",
        "|---" * (len(tasks) + 1) + "|",
    ]
    for rule in rules:
        cells = []
        for task in tasks:
            s = summary.get((rule, task))
            if not s:
                cells.append("--")
            elif s["median_units"] is not None:
                cells.append(f"{s['solve_rate']:.0%} / {int(s['median_units']):,}")
            else:
                cells.append(f"{s['solve_rate']:.0%} / --")
        lines.append(f"| {rule} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
