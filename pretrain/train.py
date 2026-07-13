"""Pretraining loop with fixed data order and dense checkpoints (spec section 6).

Reproducibility follows the OLMo/Pythia pattern the spec adopts [PY1, O2]:
deterministic seeding, a fixed data-shuffle order, and dense checkpoints saved
through training (so one can later trace when concepts become linked). At pilot
scale this is a few minutes on CPU.
"""

import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from .model import ModelConfig, TinyGPT


@dataclass
class TrainConfig:
    steps: int = 800
    batch_size: int = 16
    lr: float = 3e-3
    min_lr: float = 3e-4
    warmup: int = 50
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 100
    eval_iters: int = 40
    checkpoint_interval: int = 200
    seed: int = 1
    out_dir: str = "pretrain/checkpoints"


def _get_batch(data: np.ndarray, block_size: int, batch_size: int,
               rng: np.random.Generator):
    ix = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i:i + block_size] for i in ix])
    y = np.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return torch.from_numpy(x).long(), torch.from_numpy(y).long()


def _lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    if step >= cfg.steps:
        return cfg.min_lr
    ratio = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    coeff = 0.5 * (1.0 + np.cos(np.pi * ratio))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


@torch.no_grad()
def estimate_loss(model, data, mcfg, tcfg, rng):
    model.eval()
    losses = []
    for _ in range(tcfg.eval_iters):
        x, y = _get_batch(data, mcfg.block_size, tcfg.batch_size, rng)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def train(train_ids: np.ndarray, val_ids: np.ndarray, mcfg: ModelConfig,
          tcfg: TrainConfig, log=print) -> tuple[TinyGPT, dict]:
    torch.manual_seed(tcfg.seed)
    np.random.seed(tcfg.seed)
    rng = np.random.default_rng(tcfg.seed)        # fixed data-order stream
    eval_rng = np.random.default_rng(tcfg.seed + 999)
    os.makedirs(tcfg.out_dir, exist_ok=True)

    model = TinyGPT(mcfg)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr,
                            weight_decay=tcfg.weight_decay, betas=(0.9, 0.95))
    log(f"model params: {model.n_params:,}  vocab: {mcfg.vocab_size}  "
        f"train tokens: {len(train_ids):,}")

    history = []
    t0 = time.time()
    model.train()
    for step in range(tcfg.steps + 1):
        lr = _lr_at(step, tcfg)
        for g in opt.param_groups:
            g["lr"] = lr

        if step % tcfg.eval_interval == 0 or step == tcfg.steps:
            tr = estimate_loss(model, train_ids, mcfg, tcfg, eval_rng)
            va = estimate_loss(model, val_ids, mcfg, tcfg, eval_rng)
            history.append({"step": step, "train_loss": tr, "val_loss": va, "lr": lr})
            log(f"  step {step:4d}  train {tr:.4f}  val {va:.4f}  "
                f"lr {lr:.2e}  ({time.time()-t0:.0f}s)")

        if step > 0 and (step % tcfg.checkpoint_interval == 0):
            _save(model, mcfg, os.path.join(tcfg.out_dir, f"ckpt_step{step}.pt"))

        if step == tcfg.steps:
            break

        x, y = _get_batch(train_ids, mcfg.block_size, tcfg.batch_size, rng)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        opt.step()

    _save(model, mcfg, os.path.join(tcfg.out_dir, "final.pt"))
    stats = {
        "history": history,
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"],
        "initial_val_loss": history[0]["val_loss"],
        "wall_seconds": time.time() - t0,
    }
    return model, stats


def _save(model: TinyGPT, mcfg: ModelConfig, path: str) -> None:
    torch.save({"model": model.state_dict(), "config": asdict(mcfg)}, path)


def load_model(path: str) -> TinyGPT:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = TinyGPT(ModelConfig(**payload["config"]))
    model.load_state_dict(payload["model"])
    model.eval()
    return model
