"""Modern-model proposer: the 'twin' positive control for the elicitation harness.

Spec section 4.3: before any period model exists, the entire pipeline
(prompt -> proposed rule -> sandbox -> verifier -> pass@k per hint level) must
be able to extract a working learning rule from a model that certainly knows
backprop. If it can't, the harness -- not the period model -- is broken.

Usage:
    python -m mvp.propose_llm --hint-level 0 --k 5
    python -m mvp.propose_llm --sweep --k 5          # full dose-response curve

Requires Anthropic API credentials (ANTHROPIC_API_KEY or an `ant auth login`
profile). Without credentials the script explains itself and exits cleanly.

NOTE: the sandbox below restricts the builtins available to generated code but
is NOT a security boundary; run in an isolated environment.
"""

import argparse
import dataclasses
import json
import os
import re
import sys

import numpy as np

from .harness import summarize, format_summary_table
from .hints import build_prompt
from .verifier import make_tasks, run_trial

DEFAULT_MODEL = "claude-opus-4-8"

def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    """numpy internals lazily import submodules at call time (e.g. ndarray.mean),
    so a working sandbox must allow numpy/math imports; everything else stays out."""
    if name.split(".")[0] in ("numpy", "math"):
        return __import__(name, globals, locals, fromlist, level)
    raise ImportError(f"import of {name!r} is not allowed in the proposal sandbox")


_SANDBOX_BUILTINS = {
    k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
    for k in (
        "abs", "min", "max", "sum", "len", "range", "enumerate", "zip", "map",
        "filter", "list", "tuple", "dict", "set", "float", "int", "bool", "str",
        "print", "isinstance", "reversed", "sorted", "any", "all", "round", "pow",
        "getattr", "hasattr", "divmod", "iter", "next", "slice",
        "ValueError", "TypeError", "ZeroDivisionError", "Exception", "StopIteration",
        "IndexError", "KeyError", "RuntimeError", "NotImplementedError", "ImportError",
    )
}
_SANDBOX_BUILTINS["__import__"] = _restricted_import


class ProposedRule:
    """Wraps sandbox-executed proposer code in the Rule interface."""

    trusted = False  # gets RestrictedNet only: no backprop_grads helper

    def __init__(self, code: str, name: str):
        self.name = name
        self.code = code
        env = {"np": np, "__builtins__": _SANDBOX_BUILTINS}
        exec(code, env)  # noqa: S102 -- research sandbox, not a security boundary
        if "step" not in env or "init_state" not in env:
            raise ValueError("proposed code must define init_state() and step()")
        self._init = env["init_state"]
        self._step = env["step"]

    def init_state(self, net, task, rng):
        return self._init(net, task, rng) or {}

    def step(self, net, X, Y, rng, state):
        self._step(net, X, Y, rng, state)


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL)
    if blocks:
        return max(blocks, key=len)
    return text  # maybe the model returned bare code


def sample_proposal(client, prompt: str, model: str, sample_idx: int) -> str:
    with client.messages.stream(
        model=model,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        response = stream.get_final_message()
    if response.stop_reason == "refusal":
        raise RuntimeError("model refused the request")
    return "".join(b.text for b in response.content if b.type == "text")


def evaluate_proposal(code: str, name: str, budget: int, seeds: int) -> dict:
    tasks = make_tasks()
    try:
        rule = ProposedRule(code, name)
    except Exception as e:
        return {"name": name, "valid": False, "error": f"{type(e).__name__}: {e}",
                "functional_success": False, "results": []}
    results = []
    for task in tasks:
        for seed in range(seeds):
            results.append(run_trial(rule, task, seed, budget=budget))
    per_task_pass = {
        t.name: sum(r.solved for r in results if r.task == t.name) / seeds >= 0.5
        for t in tasks
    }
    return {
        "name": name,
        "valid": True,
        "error": None,
        # Spec 8.1 functional success: trains past criterion on ALL three tasks
        "functional_success": all(per_task_pass.values()),
        "per_task_pass": per_task_pass,
        "results": [dataclasses.asdict(r) for r in results],
    }


def run_level(client, model: str, hint_level: int, k: int, budget: int, seeds: int,
              out_dir: str) -> dict:
    prompt = build_prompt(hint_level)
    os.makedirs(out_dir, exist_ok=True)
    successes = 0
    records = []
    for i in range(k):
        name = f"L{hint_level}_sample{i}"
        print(f"[{name}] sampling from {model} ...")
        try:
            text = sample_proposal(client, prompt, model, i)
            code = extract_code(text)
        except Exception as e:
            records.append({"name": name, "valid": False, "sample_error": str(e),
                            "functional_success": False})
            continue
        record = evaluate_proposal(code, name, budget, seeds)
        record["code"] = code
        records.append(record)
        successes += record["functional_success"]
        print(f"[{name}] functional_success={record['functional_success']} "
              f"per_task={record.get('per_task_pass')}")
    level_summary = {
        "hint_level": hint_level, "model": model, "k": k,
        "pass_at_k": successes > 0, "success_rate": successes / k,
        "samples": records,
    }
    with open(os.path.join(out_dir, f"llm_hint_L{hint_level}.json"), "w") as f:
        json.dump(level_summary, f, indent=2)
    return level_summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--hint-level", type=int, default=0)
    ap.add_argument("--sweep", action="store_true", help="run all hint levels 0-5")
    ap.add_argument("--k", type=int, default=5, help="samples per hint level")
    ap.add_argument("--budget", type=int, default=400_000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    try:
        import anthropic
    except ImportError:
        print("The 'anthropic' package is required: pip install anthropic")
        return 1
    try:
        client = anthropic.Anthropic()
        client.models.retrieve(args.model)  # cheap credential + model check
    except Exception as e:
        print("Could not reach the Anthropic API (set ANTHROPIC_API_KEY or run "
              f"`ant auth login`): {type(e).__name__}: {e}")
        return 1

    levels = range(6) if args.sweep else [args.hint_level]
    curve = {}
    for lvl in levels:
        s = run_level(client, args.model, lvl, args.k, args.budget, args.seeds, args.out_dir)
        curve[lvl] = s["success_rate"]

    print("\nDose-response (hint level -> functional-success rate):")
    for lvl, rate in curve.items():
        print(f"  L{lvl}: {rate:.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
