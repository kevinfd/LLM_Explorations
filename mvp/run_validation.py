"""MVP instrument validation: prove the verifier separates real discovery from noise.

This is the P1 exit-criterion experiment from the spec (section 9), runnable on
a laptop. It asserts:

  1. POSITIVE CONTROL: backprop reliably solves all three period tasks in budget.
  2. DECOYS FAIL: sign-flipped and shuffled-credit variants (which retain all
     machinery except the actual discovery) do not pass.
  3. TASK VALIDITY: a no-hidden-layer delta rule never solves XOR/parity
     (they genuinely require the hidden layer).
  4. DISCRIMINATION: backprop beats random search and the period floor.

If these hold, the reward channel the whole experiment depends on is sound:
it can be pointed at any proposer (scripted, human, modern LLM twin, or the
eventual pre-1986 model) and its verdicts mean something.

Usage: python -m mvp.run_validation [--seeds 10] [--budget 400000]
"""

import argparse
import json
import sys

from .harness import format_summary_table, run_matrix, summarize
from .rules import REFERENCE_RULES


def check(summary, rule, task, op, threshold) -> tuple[bool, str]:
    rate = summary[(rule, task)]["solve_rate"]
    ok = rate >= threshold if op == ">=" else rate <= threshold
    return ok, f"{'PASS' if ok else 'FAIL'}  {rule} on {task}: solve-rate {rate:.0%} (want {op} {threshold:.0%})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--budget", type=int, default=400_000)
    ap.add_argument("--log", default="results/validation.jsonl")
    ap.add_argument("--summary-out", default="results/validation_summary.md")
    args = ap.parse_args()

    print(f"Running {len(REFERENCE_RULES)} rules x 3 tasks x {args.seeds} seeds "
          f"(budget {args.budget:,} units each)\n")
    results = run_matrix(REFERENCE_RULES, seeds=args.seeds, budget=args.budget,
                         log_path=args.log)
    summary = summarize(results)

    def joint_success(rule: str) -> bool:
        """Spec 8.1 functional-success criterion: >=50% solve rate on ALL tasks."""
        return all(summary[(rule, t)]["solve_rate"] >= 0.5
                   for t in ("xor", "parity3", "encoder424"))

    checks = []
    for task in ("xor", "parity3", "encoder424"):
        checks.append(check(summary, "backprop", task, ">=", 0.7))
        checks.append(check(summary, "decoy_sign_flipped", task, "<=", 0.1))
    # Shuffled credit retains a correct output-layer delta rule, so it can pass
    # the looser tasks via trained-readout-over-drifting-random-features; the
    # tight-bottleneck XOR (2-3-1) is the discriminating cell, and the JOINT
    # all-tasks criterion is what a candidate must meet (spec 8.1).
    checks.append(check(summary, "decoy_shuffled_credit", "xor", "<=", 0.1))
    for task in ("xor", "parity3"):
        checks.append(check(summary, "delta_no_hidden", task, "<=", 0.0))

    for rule, want in (("backprop", True), ("backprop_plain", True),
                       ("decoy_sign_flipped", False), ("decoy_shuffled_credit", False),
                       ("output_only_delta", False), ("delta_no_hidden", False),
                       ("random_search", False)):
        ok = joint_success(rule) == want
        checks.append((ok, f"{'PASS' if ok else 'FAIL'}  joint functional-success({rule}) == {want}"))

    # Discrimination: backprop must beat random search on every task
    # (higher solve rate, or same rate at lower median compute).
    for task in ("xor", "parity3", "encoder424"):
        bp = summary[("backprop", task)]
        rs = summary[("random_search", task)]
        ok = bp["solve_rate"] > rs["solve_rate"] or (
            bp["solve_rate"] == rs["solve_rate"]
            and (rs["median_units"] is None or bp["median_units"] < rs["median_units"])
        )
        checks.append((ok, f"{'PASS' if ok else 'FAIL'}  backprop beats random_search on {task} "
                           f"({bp['solve_rate']:.0%} vs {rs['solve_rate']:.0%})"))

    table = format_summary_table(summary)
    verdict_lines = [line for _, line in checks]
    all_ok = all(ok for ok, _ in checks)

    print("\n" + table + "\n")
    print("\n".join(verdict_lines))
    print(f"\nINSTRUMENT VALIDATION: {'PASS' if all_ok else 'FAIL'}")

    with open(args.summary_out, "w") as f:
        f.write("# Verifier instrument validation\n\n")
        f.write(f"Seeds per cell: {args.seeds} - Budget: {args.budget:,} units - ")
        f.write(f"Verdict: **{'PASS' if all_ok else 'FAIL'}**\n\n")
        f.write(table + "\n\n## Checks\n\n")
        f.write("\n".join(f"- {line}" for line in verdict_lines) + "\n")
    with open("results/validation_checks.json", "w") as f:
        json.dump({"pass": all_ok, "checks": verdict_lines}, f, indent=2)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
