"""End-to-end test of the proposer path with no API key: a mock 'proposal'
(chain-rule learning written against the RestrictedNet interface only) must
achieve functional success, and a mock bad proposal must not.

Usage: python -m mvp.test_mvp
"""

import sys

from .propose_llm import evaluate_proposal

# What a successful proposer would return: backprop implemented from scratch
# using ONLY the restricted interface (net.weights, net.forward_full, numpy).
# Note this code path never touches MLP.backprop_grads.
GOOD_PROPOSAL = """
def init_state(net, task, rng):
    return {"vel": [[np.zeros_like(W), np.zeros_like(b)] for W, b in net.weights]}

def step(net, X, Y, rng, state):
    acts = net.forward_full(X)
    out = acts[-1]
    delta = (out - Y) * out * (1 - out)
    grads = []
    for i in reversed(range(len(net.weights))):
        gW = acts[i].T @ delta / len(Y)
        gb = delta.mean(axis=0)
        grads.append((gW, gb))
        if i > 0:
            delta = (delta @ net.weights[i][0].T) * acts[i] * (1 - acts[i])
    grads.reverse()
    for pair, (gW, gb), v in zip(net.weights, grads, state["vel"]):
        v[0] = 0.9 * v[0] - 2.0 * gW
        v[1] = 0.9 * v[1] - 2.0 * gb
        pair[0] += v[0]
        pair[1] += v[1]
"""

# A plausible-sounding non-solution: adjust only the output layer.
BAD_PROPOSAL = """
def init_state(net, task, rng):
    return {}

def step(net, X, Y, rng, state):
    acts = net.forward_full(X)
    out = acts[-1]
    delta = (out - Y) * out * (1 - out)
    net.weights[-1][0] -= 2.0 * acts[-2].T @ delta / len(Y)
    net.weights[-1][1] -= 2.0 * delta.mean(axis=0)
"""

# Code that doesn't even run.
BROKEN_PROPOSAL = "def step(net):\n    return undefined_name\n"


def main() -> int:
    good = evaluate_proposal(GOOD_PROPOSAL, "mock_good", budget=400_000, seeds=5)
    bad = evaluate_proposal(BAD_PROPOSAL, "mock_bad", budget=400_000, seeds=5)
    broken = evaluate_proposal(BROKEN_PROPOSAL, "mock_broken", budget=400_000, seeds=5)

    checks = [
        (good["functional_success"] is True,
         f"good proposal functional_success == True (got {good['functional_success']}, "
         f"per_task={good.get('per_task_pass')})"),
        (bad["functional_success"] is False,
         f"bad proposal functional_success == False (got {bad['functional_success']})"),
        (broken["functional_success"] is False and not broken["valid"],
         "broken proposal is rejected as invalid"),
    ]
    ok = all(c for c, _ in checks)
    for c, msg in checks:
        print(("PASS  " if c else "FAIL  ") + msg)
    print(f"\nPROPOSER-PATH TEST: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
