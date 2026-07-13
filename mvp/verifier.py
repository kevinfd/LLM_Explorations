"""Period-clean verifier for candidate learning rules (spec section 7.1 E3 / 8.1).

A candidate rule is judged by one thing only: does it train a multilayer network
to criterion on tasks that were canonically posed *before* the 1984 cutoff
(XOR and parity from Minsky & Papert 1969; the 4-2-4 encoder from the PDP-era
demonstrations)? The reward channel therefore contains no post-cutoff knowledge
beyond the task choice itself, which the spec accounts for on the hint ladder
(~L1: it names the problem, not the solution).

Compute is metered in "units" = one sample propagated through the network once.
Every rule draws from the same budget, so comparisons are compute-matched.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


class MLP:
    """Sigmoid multilayer perceptron with metered forward passes.

    weights: list of [W, b] pairs (mutable in place by learning rules).
    Charges `len(X)` units per forward call unless charge=False (harness-side
    criterion checks are free so that early stopping is uniform across rules).
    """

    def __init__(self, sizes: list[int], rng: np.random.Generator, init_scale: float = 0.5):
        self.sizes = list(sizes)
        self.weights = [
            [rng.normal(0.0, init_scale, size=(n_in, n_out)), np.zeros(n_out)]
            for n_in, n_out in zip(sizes[:-1], sizes[1:])
        ]
        self.units_used = 0

    def forward(self, X: np.ndarray, charge: bool = True) -> np.ndarray:
        return self.forward_full(X, charge=charge)[-1]

    def forward_full(self, X: np.ndarray, charge: bool = True) -> list[np.ndarray]:
        """All layer activations, input first, output last."""
        if charge:
            self.units_used += len(X)
        acts = [X]
        for W, b in self.weights:
            acts.append(sigmoid(acts[-1] @ W + b))
        return acts

    def mse(self, X: np.ndarray, Y: np.ndarray, charge: bool = True) -> float:
        out = self.forward(X, charge=charge)
        return float(np.mean((out - Y) ** 2))

    # NOTE: reference-rule helper only. Never expose this to a proposer
    # (a model or human being tested) -- it IS the answer under test.
    def backprop_grads(self, acts: list[np.ndarray], Y: np.ndarray, charge: bool = True):
        if charge:
            self.units_used += len(acts[0])  # backward pass costs ~one forward
        grads = []
        delta = (acts[-1] - Y) * acts[-1] * (1 - acts[-1])
        for i in reversed(range(len(self.weights))):
            gW = acts[i].T @ delta / len(Y)
            gb = delta.mean(axis=0)
            grads.append([gW, gb])
            if i > 0:
                delta = (delta @ self.weights[i][0].T) * acts[i] * (1 - acts[i])
        return grads[::-1]

    def flat_weights(self) -> np.ndarray:
        return np.concatenate([np.concatenate([W.ravel(), b.ravel()]) for W, b in self.weights])

    def set_flat_weights(self, flat: np.ndarray) -> None:
        i = 0
        for pair in self.weights:
            W, b = pair
            pair[0] = flat[i : i + W.size].reshape(W.shape)
            i += W.size
            pair[1] = flat[i : i + b.size].copy()
            i += b.size

    @property
    def n_params(self) -> int:
        return sum(W.size + b.size for W, b in self.weights)


class RestrictedNet:
    """What a proposer-under-test may touch: weights, forward passes, sizes.

    Deliberately excludes backprop_grads -- the proposer must supply the
    credit-assignment mechanism itself.
    """

    def __init__(self, net: MLP):
        self._net = net
        self.weights = net.weights
        self.sizes = net.sizes

    def forward(self, X):
        return self._net.forward(X)

    def forward_full(self, X):
        return self._net.forward_full(X)

    def mse(self, X, Y):
        return self._net.mse(X, Y)

    def flat_weights(self):
        return self._net.flat_weights()

    def set_flat_weights(self, flat):
        self._net.set_flat_weights(flat)

    @property
    def n_params(self):
        return self._net.n_params


@dataclass
class Task:
    name: str
    X: np.ndarray
    Y: np.ndarray
    arch: list[int]           # includes input/output layers
    no_hidden_arch: list[int]  # for the linear-baseline task-validity check
    criterion: Callable[[np.ndarray, np.ndarray], bool]
    description: str = ""


def _threshold_criterion(out: np.ndarray, Y: np.ndarray) -> bool:
    """Every output on the correct side of 0.5 (Rumelhart et al.'s success test)."""
    return bool(np.all((out > 0.5) == (Y > 0.5)))


def _argmax_criterion(out: np.ndarray, Y: np.ndarray) -> bool:
    return bool(np.all(np.argmax(out, axis=1) == np.argmax(Y, axis=1)))


def _bits(n: int) -> np.ndarray:
    return np.array([[(i >> b) & 1 for b in range(n)] for i in range(2 ** n)], dtype=float)


def make_tasks() -> list[Task]:
    xor_X = _bits(2)
    xor_Y = (xor_X.sum(axis=1) % 2).reshape(-1, 1)

    par_X = _bits(3)
    par_Y = (par_X.sum(axis=1) % 2).reshape(-1, 1)

    enc_X = np.eye(4)

    return [
        Task(
            name="xor",
            X=xor_X, Y=xor_Y,
            arch=[2, 3, 1], no_hidden_arch=[2, 1],
            criterion=_threshold_criterion,
            description="Exclusive-or: the canonical non-linearly-separable predicate (Minsky & Papert 1969).",
        ),
        Task(
            name="parity3",
            X=par_X, Y=par_Y,
            arch=[3, 8, 1], no_hidden_arch=[3, 1],
            criterion=_threshold_criterion,
            description="3-bit parity: harder generalization of XOR.",
        ),
        Task(
            name="encoder424",
            X=enc_X, Y=enc_X.copy(),
            arch=[4, 2, 4], no_hidden_arch=[4, 4],
            criterion=_argmax_criterion,
            description="4-2-4 encoder: reproduce one-of-four input through a 2-unit bottleneck.",
        ),
    ]


@dataclass
class TrialResult:
    rule: str
    task: str
    seed: int
    solved: bool
    units_to_criterion: int | None
    final_mse: float
    budget: int
    steps: int
    error: str | None = None


def run_trial(
    rule,
    task: Task,
    seed: int,
    budget: int = 400_000,
    check_interval: int = 20,
    arch_override: list[int] | None = None,
) -> TrialResult:
    """Run one rule on one task with one seed under a fixed compute budget.

    The rule sees only a RestrictedNet unless it is a trusted reference rule
    (rule.trusted == True), in which case it gets the raw MLP (reference rules
    are allowed to use backprop_grads -- that is the point of them).
    """
    rng = np.random.default_rng(seed)
    net = MLP(arch_override or task.arch, rng)
    view = net if getattr(rule, "trusted", False) else RestrictedNet(net)

    steps = 0
    solved = False
    units_at_solve = None
    error = None
    try:
        state = rule.init_state(view, task, rng)
        while net.units_used < budget:
            rule.step(view, task.X, task.Y, rng, state)
            steps += 1
            if steps % check_interval == 0 or net.units_used >= budget:
                out = net.forward(task.X, charge=False)
                if task.criterion(out, task.Y):
                    solved = True
                    units_at_solve = net.units_used
                    break
    except Exception as e:  # a broken proposed rule is a failed trial, not a crash
        error = f"{type(e).__name__}: {e}"

    return TrialResult(
        rule=rule.name,
        task=task.name,
        seed=seed,
        solved=solved,
        units_to_criterion=units_at_solve,
        final_mse=net.mse(task.X, task.Y, charge=False),
        budget=budget,
        steps=steps,
        error=error,
    )
