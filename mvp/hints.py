"""The preregistered graded hint ladder (spec section 7.3) and proposer prompts.

Every hint below is expressible from pre-1985 sources (Minsky & Papert 1969;
Widrow & Hoff 1960 and the documented Madaline failure; Rosenblatt 1962;
Kelley/Bryson/Dreyfus 1960-62). Discovery rate is always reported as a function
of hint level -- that dose-response curve is the answer to "how much steering
was needed", and it turns experimenter leakage into a measured quantity.

The INTERFACE_SPEC is modern engineering scaffolding: it tells the proposer the
*format* of an answer, not its content. It is counted as part of L0.
"""

PROBLEM_STATEMENT = """\
The year is 1984. The following is the state of the art, as found in the
published literature.

A perceptron adjusts the connection weights of a single layer of threshold
units by an error-correction procedure, and there is a convergence proof for
it (Rosenblatt, 1958, 1962). A closely related device, the Adaline, adjusts a
single linear unit by exact gradient descent on the mean squared error -- the
least-mean-squares or "delta" rule (Widrow & Hoff, 1960). These procedures are
limited to what a single layer can represent: it has been proved that no
single-layer machine can compute the exclusive-or or parity predicates
(Minsky & Papert, 1969).

Machines with an intermediate ("hidden") layer of units could in principle
represent such predicates, but no satisfactory procedure is known for deciding
how to change the weights of the intermediate layer: when the output is wrong,
it is not known which of the earlier connections deserves the blame. Attempts
to extend the delta rule to multiple layers of threshold units (the Madaline)
did not yield a general procedure. Rosenblatt (1962) experimented with
"back-propagating error correction", passing corrections heuristically to
earlier layers, without a convergence guarantee or general success.

THE PROBLEM: propose a procedure for training ALL the weights of a layered
network -- including the intermediate layer -- so that the machine can learn
predicates such as exclusive-or from examples. The procedure will be judged
empirically: it will be implemented and run.
"""

INTERFACE_SPEC = """\
Return your procedure as Python code (this format requirement is testing
apparatus, not part of the problem). Define exactly two functions:

    def init_state(net, task, rng):
        # return a dict of any working memory your procedure needs
        return {}

    def step(net, X, Y, rng, state):
        # perform ONE iteration of your procedure, modifying net.weights in place
        ...

You may use numpy as `np`. The network object `net` offers:
  - net.weights: list of [W, b] pairs, layer by layer (numpy arrays; mutate in place)
  - net.forward(X): outputs for a batch of inputs (each unit's response varies
    smoothly with its input: output = 1/(1+exp(-z)))
  - net.forward_full(X): the activations of every layer, input first, output last
  - net.mse(X, Y): mean squared error on the batch
  - net.flat_weights() / net.set_flat_weights(v): all weights as one vector
  - net.n_params: total number of weights
`X` and `Y` are the training examples. `rng` is a numpy random generator.
Every call that propagates the batch through the network is metered against a
fixed compute budget, so efficiency matters. `step` will be called repeatedly
until the budget is exhausted or the task is solved.

Reply with a single Python code block and nothing else outside it.
"""

# L0 is the problem statement + interface only.
HINT_LADDER = {
    0: "",
    1: (
        "Hint: the obstacle of training the intermediate layers may deserve "
        "attack by the methods of numerical optimization -- treat learning as "
        "minimizing a measure of error."
    ),
    2: (
        "Hint: consider units whose response varies smoothly with their input "
        "(as the network above already provides), so that derivatives of the "
        "error exist with respect to every weight."
    ),
    3: (
        "Hint: in trajectory optimization, gradient methods for staged systems "
        "compute sensitivities stage by stage, working backward from the final "
        "stage (Kelley 1960; Bryson & Denham 1962; Dreyfus 1962)."
    ),
    4: (
        "Hint: apply the chain rule to obtain the derivative of the error with "
        "respect to EVERY connection weight, including those of the "
        "intermediate units, and adjust each weight against its derivative."
    ),
    5: (
        "Hint (near-spoiler): compute an error signal for each output unit as "
        "(output - target) times the derivative of the unit's response; "
        "propagate an error signal to each intermediate unit by summing the "
        "error signals of the units it feeds, weighted by the connecting "
        "weights, times the derivative of its own response; change every weight "
        "in proportion to the presynaptic activity times the postsynaptic "
        "error signal, in the direction that reduces the error."
    ),
}


def build_prompt(hint_level: int) -> str:
    if hint_level not in HINT_LADDER:
        raise ValueError(f"hint level must be one of {sorted(HINT_LADDER)}")
    parts = [PROBLEM_STATEMENT]
    if HINT_LADDER[hint_level]:
        parts.append(HINT_LADDER[hint_level])
    parts.append(INTERFACE_SPEC)
    return "\n\n".join(parts)
