"""End-to-end time-locked pretraining pilot (spec section 9, phase P1).

corpus -> hard cutoff filter + canary audit -> from-scratch tokenizer ->
tiny-transformer pretraining with dense checkpoints -> cutoff-integrity
acceptance probes. Every stage writes an artifact under results/pretrain/.

This is the runnable analog of the spec's P1 exit criteria: a from-scratch
model on a provenance-filtered, cutoff-respecting corpus, with the acceptance
tests that must pass before any discovery claim is meaningful. Toy scale by
design -- it validates the pipeline, not scientific conclusions.

Usage:
    python -m pretrain.run_pipeline                 # default 1900 cutoff
    python -m pretrain.run_pipeline --cutoff 1850 --steps 400
"""

import argparse
import json
import os
import sys

import numpy as np

from .corpus import load_documents
from .cutoff import apply_cutoff, build_training_text
from .model import ModelConfig
from .probe import run_probes
from .tokenizer import EOT, BPETokenizer
from .train import TrainConfig, train

OUT = "results/pretrain"


def stratified_sample(admitted, per_doc_chars: int) -> str:
    """Tokenizer-training text: a slice from each admitted doc so the merges
    aren't dominated by the longest work (spec: tokenizer trained on the
    period corpus -- keep it representative)."""
    return "\n".join(d.text[:per_doc_chars] for d in admitted)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cutoff", type=int, default=1900)
    ap.add_argument("--vocab-size", type=int, default=2500)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-head", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--tok-sample-per-doc", type=int, default=200_000)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    # ---- Stage 1: corpus + cutoff audit ---------------------------------
    print("== Stage 1: corpus + cutoff audit ==")
    docs = load_documents()
    report = apply_cutoff(docs, cutoff_year=args.cutoff)
    with open(f"{OUT}/cutoff_audit.md", "w") as f:
        f.write(report.render())
    print(report.render())
    if not report.clean:
        print("\nABORT: canary audit found a leak; corpus is not period-pure.")
        return 1
    if len(report.admitted) < 2 or not report.excluded:
        print("\nABORT: need >=2 admitted docs and >=1 excluded doc for probes.")
        return 1

    # ---- Stage 2: from-scratch tokenizer --------------------------------
    print("\n== Stage 2: from-scratch BPE tokenizer ==")
    tok_text = stratified_sample(report.admitted, args.tok_sample_per_doc)
    tok = BPETokenizer().train(tok_text, vocab_size=args.vocab_size)
    tok.save(f"{OUT}/tokenizer.json")
    print(f"  trained tokenizer: vocab {tok.size} on {len(tok_text):,} chars")

    # ---- Stage 3: tokenize + stratified split ---------------------------
    # Genre-matched val: hold out the tail fraction of EACH admitted document,
    # so the held-out set has the same genre mix as training (avoids the
    # confound of val = one idiosyncratic work). Spec section 8.4.
    print("\n== Stage 3: tokenize corpus (stratified split) ==")
    eot_id = tok.vocab[EOT]
    train_chunks, val_chunks = [], []
    for d in report.admitted:
        doc_ids = tok.encode(d.text)
        n_val = max(1, int(len(doc_ids) * args.val_frac))
        train_chunks.append(doc_ids[:-n_val] + [eot_id])
        val_chunks.append(doc_ids[-n_val:])
    train_ids = np.array([i for c in train_chunks for i in c], dtype=np.int32)
    pre_val_ids = np.array([i for c in val_chunks for i in c], dtype=np.int32)
    admitted_text = build_training_text(report.admitted)
    excluded_text = "\n\n".join(d.text for d in report.excluded)
    post_ids = np.array(tok.encode(excluded_text), dtype=np.int32)
    print(f"  train {len(train_ids):,} | pre-cutoff val {len(pre_val_ids):,} | "
          f"post-cutoff probe {len(post_ids):,} tokens")

    # ---- Stage 4: pretrain ----------------------------------------------
    print("\n== Stage 4: pretrain tiny transformer ==")
    mcfg = ModelConfig(vocab_size=tok.size, block_size=args.block_size,
                       n_layer=args.n_layer, n_head=args.n_head,
                       d_model=args.d_model, dropout=args.dropout)
    tcfg = TrainConfig(steps=args.steps, batch_size=args.batch_size,
                       seed=args.seed, out_dir=f"{OUT}/checkpoints")
    model, stats = train(train_ids, pre_val_ids, mcfg, tcfg)

    # ---- Stage 5: acceptance probes -------------------------------------
    print("\n== Stage 5: cutoff-integrity acceptance probes ==")
    probes = run_probes(
        model, tok, pre_val_ids, post_ids, admitted_text, excluded_text,
        args.block_size, initial_val_loss=stats["initial_val_loss"],
        final_val_loss=stats["final_val_loss"], seed=args.seed,
    )
    with open(f"{OUT}/probe_report.md", "w") as f:
        f.write(probes.render())
    print(probes.render())

    # ---- machine-readable summary ---------------------------------------
    summary = {
        "cutoff_year": args.cutoff,
        "admitted_docs": [d.prov.doc_id for d in report.admitted],
        "excluded_docs": [d.prov.doc_id for d in report.excluded],
        "corpus_clean": report.clean,
        "vocab_size": tok.size,
        "model_params": model.n_params,
        "train_tokens": int(len(train_ids)),
        "initial_val_loss": stats["initial_val_loss"],
        "final_val_loss": stats["final_val_loss"],
        "pre_entity_mean_nll": probes.pre_entity_mean,
        "post_entity_mean_nll": probes.post_entity_mean,
        "era_gap": probes.era_gap,
        "pre_cutoff_ppl": probes.pre_cutoff_ppl,
        "post_cutoff_ppl": probes.post_cutoff_ppl,
        "cutoff_respected": probes.cutoff_respected,
        "wall_seconds": stats["wall_seconds"],
    }
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    ok = report.clean and probes.cutoff_respected
    print(f"\n{'='*60}\nPIPELINE {'PASS' if ok else 'FAIL'} -- artifacts in {OUT}/")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
