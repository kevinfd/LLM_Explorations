"""Reference learning rules for validating the verifier (spec section 4.3 controls).

Three classes:
- POSITIVE CONTROLS: the historical answer (backprop). Must pass all tasks.
- DECOYS: counterfactually-altered variants (NewtonBench-style). A verifier
  that cannot separate these from the real rule is not measuring anything.
- PERIOD-PLAUSIBLE ALTERNATIVES & BASELINES: rules a pre-1986 model might
  legitimately invent instead (perturbation search, finite differences), plus
  the floors every candidate must beat (random search, linear-only learning).

Reference rules are `trusted` and may use MLP.backprop_grads; anything coming
from a proposer-under-test goes through RestrictedNet and cannot.
"""

import numpy as np


class BackpropSGD:
    """Rumelhart-Hinton-Williams 1986: chain-rule gradients + momentum SGD."""

    name = "backprop"
    trusted = True

    def __init__(self, lr: float = 2.0, momentum: float = 0.9, name: str | None = None):
        self.lr = lr
        self.momentum = momentum
        if name:
            self.name = name

    def init_state(self, net, task, rng):
        return {"vel": [[np.zeros_like(W), np.zeros_like(b)] for W, b in net.weights]}

    def step(self, net, X, Y, rng, state):
        acts = net.forward_full(X)
        grads = net.backprop_grads(acts, Y)
        for pair, g, v in zip(net.weights, grads, state["vel"]):
            for j in range(2):
                v[j] = self.momentum * v[j] - self.lr * g[j]
                pair[j] += v[j]


class SignFlippedBackprop(BackpropSGD):
    """DECOY: identical machinery, gradient *ascent*. Must fail."""

    name = "decoy_sign_flipped"

    def step(self, net, X, Y, rng, state):
        acts = net.forward_full(X)
        grads = net.backprop_grads(acts, Y)
        for pair, g in zip(net.weights, grads):
            for j in range(2):
                pair[j] += self.lr * g[j]


class ShuffledCreditBackprop(BackpropSGD):
    """DECOY: correct output-layer updates, but hidden-layer credit assignment
    is randomly permuted across hidden units each step -- i.e. the chain-rule
    routing (the actual discovery) is destroyed while everything else is kept."""

    name = "decoy_shuffled_credit"

    def step(self, net, X, Y, rng, state):
        acts = net.forward_full(X)
        grads = net.backprop_grads(acts, Y)
        for li, (pair, g) in enumerate(zip(net.weights, grads)):
            gW, gb = g
            if li < len(net.weights) - 1:  # hidden layers: scramble credit
                perm = rng.permutation(gW.shape[1])
                gW = gW[:, perm]
                gb = gb[perm]
            pair[0] -= self.lr * gW
            pair[1] -= self.lr * gb


class OutputOnlyDelta:
    """PERIOD FLOOR: Widrow-Hoff delta rule on the output layer only; hidden
    weights frozen at their random initialization. This is exactly where the
    field was stuck pre-backprop (the documented Madaline failure)."""

    name = "output_only_delta"
    trusted = True

    def __init__(self, lr: float = 2.0):
        self.lr = lr

    def init_state(self, net, task, rng):
        return {}

    def step(self, net, X, Y, rng, state):
        acts = net.forward_full(X)
        out = acts[-1]
        delta = (out - Y) * out * (1 - out)
        W, b = net.weights[-1]
        W -= self.lr * acts[-2].T @ delta / len(Y)
        b -= self.lr * delta.mean(axis=0)


class DeltaNoHidden(OutputOnlyDelta):
    """TASK-VALIDITY BASELINE: delta rule with *no hidden layer at all*.
    Provably cannot solve XOR/parity -- if it does, the task is broken.
    (On the 4-4 direct encoder it should trivially succeed; that is expected
    and reported, not asserted.) Run with arch_override=task.no_hidden_arch."""

    name = "delta_no_hidden"


class WeightPerturbation:
    """PERIOD-PLAUSIBLE ALTERNATIVE: perturb all weights with Gaussian noise,
    keep the perturbation iff error decreases (a hill-climber; adaptive step)."""

    name = "weight_perturbation"
    trusted = False  # needs nothing privileged

    def __init__(self, sigma: float = 0.3):
        self.sigma = sigma

    def init_state(self, net, task, rng):
        return {"best": net.mse(task.X, task.Y), "sigma": self.sigma}

    def step(self, net, X, Y, rng, state):
        flat = net.flat_weights()
        trial = flat + rng.normal(0.0, state["sigma"], size=flat.shape)
        net.set_flat_weights(trial)
        err = net.mse(X, Y)
        if err < state["best"]:
            state["best"] = err
            state["sigma"] = min(state["sigma"] * 1.05, 1.0)
        else:
            net.set_flat_weights(flat)
            state["sigma"] = max(state["sigma"] * 0.98, 0.01)


class FiniteDiffSGD:
    """PERIOD-PLAUSIBLE ALTERNATIVE: numerically estimate the gradient by
    central differences per weight, then descend. Mathematically the same
    direction as backprop but pays ~2*P forward passes per step instead of ~2 --
    the efficiency gap that reverse-mode accumulation (the discovery) closes."""

    name = "finite_diff"
    trusted = False

    def __init__(self, lr: float = 2.0, eps: float = 1e-3):
        self.lr = lr
        self.eps = eps

    def init_state(self, net, task, rng):
        return {}

    def step(self, net, X, Y, rng, state):
        flat = net.flat_weights()
        grad = np.zeros_like(flat)
        for i in range(len(flat)):
            flat[i] += self.eps
            net.set_flat_weights(flat)
            hi = net.mse(X, Y)
            flat[i] -= 2 * self.eps
            net.set_flat_weights(flat)
            lo = net.mse(X, Y)
            flat[i] += self.eps
            grad[i] = (hi - lo) / (2 * self.eps)
        net.set_flat_weights(flat - self.lr * grad)


class RandomSearch:
    """COMPUTE-MATCHED BASELINE: sample fresh random weight vectors, keep the
    best. Any claimed discovery must beat this at the same budget."""

    name = "random_search"
    trusted = False

    def __init__(self, scale: float = 5.0):
        self.scale = scale

    def init_state(self, net, task, rng):
        return {"best_err": net.mse(task.X, task.Y), "best_w": net.flat_weights()}

    def step(self, net, X, Y, rng, state):
        trial = rng.uniform(-self.scale, self.scale, size=net.n_params)
        net.set_flat_weights(trial)
        err = net.mse(X, Y)
        if err < state["best_err"]:
            state["best_err"] = err
            state["best_w"] = trial
        else:
            net.set_flat_weights(state["best_w"])


REFERENCE_RULES = [
    BackpropSGD(),                                   # positive control
    BackpropSGD(lr=0.5, momentum=0.0, name="backprop_plain"),
    SignFlippedBackprop(),                           # decoy
    ShuffledCreditBackprop(),                        # decoy
    OutputOnlyDelta(),                               # period floor
    DeltaNoHidden(),                                 # task-validity baseline
    WeightPerturbation(),                            # period-plausible alternative
    FiniteDiffSGD(),                                 # period-plausible alternative
    RandomSearch(),                                  # baseline
]
