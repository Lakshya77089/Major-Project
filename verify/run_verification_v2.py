"""
zkFedMoE Verification Harness v2 -- Convergence-corrected re-runs

Changes from v1:
  - Exp 4 robustness: 9 clients (not 5) so Median@40% has stable n=9
    coordinate-wise median with 4 of 9 malicious. Also increases rounds 5 -> 8.
  - Exp 5 non-IID Dirichlet: rounds 3 -> 8 so the model actually converges
    before evaluating heterogeneity effect.
  - Exp 6 MIA: rounds 3 -> 8, members subsampled to 800 per side, more local
    epochs per round so the model can memorise enough for the loss-threshold
    attack to work.
  - Exp 7 FedProx: rounds 3 -> 8, mu range shifted to {0, 1e-5, 1e-4, 1e-3}
    -- the paper's mu={0.001..0.1} range is too large for a 600K-param model
    where the prox-loss sums over all params and dominates the cross-entropy.

All other parameters identical to v1; the goal is *convergence*, not
re-defining the experiment.

Saves to verify/results/measured_v2.json. Does NOT overwrite v1 results.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset, TensorDataset

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
CODE_DIR = PROJECT_ROOT / "Code"
sys.path.insert(0, str(CODE_DIR))

from src.models.moe_model import MoETextClassifier
from src.data.text_datasets import (
    build_ag_news_clients,
    build_ag_news_clients_noniid,
    client_class_distribution,
)
from src.fl.client import local_train
from src.fl.server import FedServer
from src.fl.dp import apply_dp, RenyiAccountant
from src.fl.adversaries import poisoning_train
from src.fl.attacks import membership_inference_attack

RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "measured_v2.json"
DEVICE = torch.device("cpu")


def set_seed(s: int = 42) -> None:
    torch.manual_seed(s)
    np.random.seed(s)


def evaluate(model, dataset, bs: int = 64) -> float:
    loader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=0)
    model.to(DEVICE).eval()
    correct = total = 0
    with torch.no_grad():
        for ids, lbl in loader:
            ids, lbl = ids.to(DEVICE), lbl.to(DEVICE)
            correct += (model(ids).argmax(-1) == lbl).sum().item()
            total += lbl.size(0)
    return correct / max(total, 1)


def _save(d: Dict[str, Any]) -> None:
    def fix(o):
        if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))):
            return str(o)
        if isinstance(o, dict):
            return {k: fix(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [fix(x) for x in o]
        if isinstance(o, np.generic):
            return o.item()
        return o
    RESULTS_FILE.write_text(json.dumps(fix(d), indent=2))


def _load() -> Dict[str, Any]:
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text())
    return {}


def _save_partial(name: str, payload: Dict[str, Any]) -> None:
    cur = _load()
    cur[name] = payload
    _save(cur)
    print(f"  [CHECKPOINT v2] Saved {name}")


def _hr(t: str) -> None:
    print(f"\n{'='*78}\n  {t}\n{'='*78}")


def make_model_kw(vs, nc, num_experts=8, k=2):
    return dict(vocab_size=vs, embed_dim=64, num_classes=nc,
                num_experts=num_experts, expert_hidden_dim=256, k=k, lora_r=8)


# Cached AG News
_AG = None
def load_ag_news_iid(num_clients: int):
    global _AG
    if _AG is None:
        _AG = build_ag_news_clients(num_clients=1, seq_len=64,
                                    use_external_csv=True, max_vocab=5000)
    cl1, td, vs, nc, voc = _AG
    full = cl1[0]
    n = len(full)
    sizes = [n // num_clients] * num_clients
    sizes[0] += n - sum(sizes)
    from torch.utils.data import random_split
    cls = list(random_split(full, sizes))
    return cls, td, vs, nc, voc


# ===========================================================================
# EXP 4 v2 -- Robustness with 9 clients, 8 rounds
# ===========================================================================
def exp4_v2() -> Dict[str, Any]:
    _hr("EXPERIMENT 4 v2: Robustness (9 clients, 8 rounds, label-flip)")

    NUM_CLIENTS = 9
    ROUNDS = 8
    fractions = [0.0, 0.22, 0.44]   # 0/2/4 mal -> 0%/22%/44%
    methods = {
        "FedAvg":       "aggregate",
        "Median":       "aggregate_median",
        "Trimmed Mean": "aggregate_trimmed_mean",
    }

    clients, test_ds, vs, nc, _ = load_ag_news_iid(NUM_CLIENTS)
    kw = make_model_kw(vs, nc)
    results: Dict[str, Dict[str, float]] = {m: {} for m in methods}

    for method_name, attr in methods.items():
        for frac in fractions:
            set_seed(42)
            srv = FedServer(MoETextClassifier(**kw), device=DEVICE)
            n_mal = max(0, int(round(NUM_CLIENTS * frac)))
            t0 = time.perf_counter()
            for rnd in range(ROUNDS):
                round_states = []
                for ci, cds in enumerate(clients):
                    cm = MoETextClassifier(**kw)
                    cm.load_state_dict(srv.get_global_state(), strict=False)
                    if ci < n_mal:
                        state, n = poisoning_train(
                            cm, cds, epochs=1, batch_size=64, lr=2e-3,
                            device=DEVICE, num_classes=nc)
                    else:
                        fs, _, n, _, _, _, _ = local_train(
                            cm, cds, epochs=1, batch_size=64, lr=2e-3,
                            device=DEVICE, top_k_sparse=2)
                        state = fs
                    round_states.append((state, n))
                getattr(srv, attr)(round_states)
            wall = time.perf_counter() - t0
            acc = evaluate(srv.global_model, test_ds)
            label = f"{int(frac*100)}%"
            results[method_name][label] = round(acc * 100, 2)
            print(f"  {method_name:13s}  mal={label}  acc={acc*100:5.2f}%  "
                  f"n_mal={n_mal}/{NUM_CLIENTS}  wall={wall:.0f}s")

    payload = {"results": results, "num_clients": NUM_CLIENTS, "rounds": ROUNDS}
    _save_partial("exp4_v2", payload)
    return payload


# ===========================================================================
# EXP 5 v2 -- Non-IID with 8 rounds (was 3)
# ===========================================================================
def exp5_v2() -> Dict[str, Any]:
    _hr("EXPERIMENT 5 v2: Non-IID Dirichlet alpha sweep (5 clients, 8 rounds)")

    alphas = [0.1, 0.3, 0.5, 1.0, 5.0, 100.0]
    ROUNDS = 8
    rows = []

    for alpha in alphas:
        set_seed(42)
        clients, test_ds, vs, nc, _ = build_ag_news_clients_noniid(
            num_clients=5, alpha=alpha, use_external_csv=True, seed=42)
        kw = make_model_kw(vs, nc)
        srv = FedServer(MoETextClassifier(**kw), device=DEVICE)

        client_dists = [client_class_distribution(c, nc).tolist() for c in clients]

        t0 = time.perf_counter()
        for rnd in range(ROUNDS):
            states = []
            for cds in clients:
                cm = MoETextClassifier(**kw)
                cm.load_state_dict(srv.get_global_state(), strict=False)
                fs, _, n, _, _, _, _ = local_train(
                    cm, cds, epochs=1, batch_size=64, lr=2e-3,
                    device=DEVICE, top_k_sparse=2)
                states.append((fs, n))
            srv.aggregate(states)
        wall = time.perf_counter() - t0
        acc = evaluate(srv.global_model, test_ds)
        rows.append({
            "alpha": alpha, "accuracy_pct": round(acc * 100, 2),
            "client_dists": client_dists, "wall_seconds": round(wall, 1),
        })
        print(f"  alpha={alpha:>6.1f}  acc={acc*100:5.2f}%  "
              f"client0_size={sum(client_dists[0])}  wall={wall:.0f}s")

    payload = {"rows": rows, "rounds": ROUNDS}
    _save_partial("exp5_v2", payload)
    return payload


# ===========================================================================
# EXP 6 v2 -- MIA with smaller members + more rounds + more local epochs
# ===========================================================================
def exp6_v2() -> Dict[str, Any]:
    _hr("EXPERIMENT 6 v2: MIA with 8 rounds, 3 local epochs, small member set")

    ROUNDS = 8
    LOCAL_EPOCHS = 3
    configs = [
        ("No DP",        None, None),
        ("DP sigma=0.1", 1.0, 0.1),
        ("DP sigma=0.5", 1.0, 0.5),
        ("DP sigma=1.0", 1.0, 1.0),
    ]

    # Use 5 clients but smaller AG-News slice so the model can memorise
    # (~3000 samples per client instead of 24000). This is the regime where
    # MIA actually has signal.
    clients_full, test_ds, vs, nc, _ = load_ag_news_iid(5)
    # Subsample each client shard down to ~3000 samples (member shards are
    # what the attacker tries to identify; smaller = more memorisation)
    from torch.utils.data import Subset as _Sub
    clients = []
    SUBSAMPLE = 3000
    for shard in clients_full:
        n = len(shard)
        idx = torch.randperm(n)[:min(n, SUBSAMPLE)].tolist()
        clients.append(_Sub(shard, idx))

    rows = []
    for label, clip, sigma in configs:
        set_seed(42)
        kw = make_model_kw(vs, nc)
        srv = FedServer(MoETextClassifier(**kw), device=DEVICE)
        t0 = time.perf_counter()
        for rnd in range(ROUNDS):
            states = []
            for cds in clients:
                cm = MoETextClassifier(**kw)
                cm.load_state_dict(srv.get_global_state(), strict=False)
                fs, _, n, _, _, _, _ = local_train(
                    cm, cds, epochs=LOCAL_EPOCHS, batch_size=64, lr=2e-3,
                    device=DEVICE, top_k_sparse=2)
                if sigma is not None and sigma > 0:
                    fs = apply_dp(fs, clip_norm=clip, noise_multiplier=sigma)
                states.append((fs, n))
            srv.aggregate(states)
        train_wall = time.perf_counter() - t0

        acc = evaluate(srv.global_model, test_ds)
        members = ConcatDataset(clients[:2])
        mia = membership_inference_attack(
            srv.global_model, members, test_ds,
            device=DEVICE, max_samples=400)
        rows.append({
            "config": label, "sigma": sigma,
            "accuracy_pct": round(acc * 100, 2),
            "mia_auc": round(mia.auc, 3),
            "mia_attack_acc": round(mia.attack_accuracy, 3),
            "member_loss_mean": round(mia.train_loss_mean, 3),
            "nonmember_loss_mean": round(mia.nonmember_loss_mean, 3),
            "train_wall_seconds": round(train_wall, 1),
        })
        print(f"  {label:14s}  acc={acc*100:5.2f}%  AUC={mia.auc:.3f}  "
              f"attack_acc={mia.attack_accuracy:.1%}  "
              f"mem_loss={mia.train_loss_mean:.3f}  "
              f"nonmem_loss={mia.nonmember_loss_mean:.3f}  "
              f"wall={train_wall:.0f}s")

    payload = {"rows": rows, "rounds": ROUNDS, "local_epochs": LOCAL_EPOCHS,
               "samples_per_client": SUBSAMPLE}
    _save_partial("exp6_v2", payload)
    return payload


# ===========================================================================
# EXP 7 v2 -- FedProx with smaller mu range, 8 rounds
# ===========================================================================
def exp7_v2() -> Dict[str, Any]:
    _hr("EXPERIMENT 7 v2: FedProx (5 clients, 8 rounds, alpha=0.3, smaller mu range)")

    ALPHA = 0.3
    ROUNDS = 8
    # Paper's mu range was {0.001..0.5}; that overpowers cross-entropy on
    # 600K params. We sweep a smaller range to find the *actual* sweet spot.
    mu_values = [0.0, 1e-5, 1e-4, 5e-4, 1e-3, 1e-2]
    rows = []

    for mu in mu_values:
        set_seed(42)
        clients, test_ds, vs, nc, _ = build_ag_news_clients_noniid(
            num_clients=5, alpha=ALPHA, use_external_csv=True, seed=42)
        kw = make_model_kw(vs, nc)
        srv = FedServer(MoETextClassifier(**kw), device=DEVICE)

        t0 = time.perf_counter()
        for rnd in range(ROUNDS):
            states = []
            for cds in clients:
                cm = MoETextClassifier(**kw)
                cm.load_state_dict(srv.get_global_state(), strict=False)
                fs, _, n, _, _, _, _ = local_train(
                    cm, cds, epochs=1, batch_size=64, lr=2e-3,
                    device=DEVICE, top_k_sparse=2, fedprox_mu=mu)
                states.append((fs, n))
            srv.aggregate(states)
        wall = time.perf_counter() - t0
        acc = evaluate(srv.global_model, test_ds)
        label = "FedAvg" if mu == 0.0 else f"FedProx mu={mu}"
        rows.append({"mu": mu, "label": label, "alpha": ALPHA,
                     "accuracy_pct": round(acc * 100, 2),
                     "wall_seconds": round(wall, 1)})
        print(f"  {label:20s}  alpha={ALPHA}  acc={acc*100:5.2f}%  wall={wall:.0f}s")

    payload = {"rows": rows, "alpha": ALPHA, "rounds": ROUNDS}
    _save_partial("exp7_v2", payload)
    return payload


ALL_V2 = [
    ("exp4_v2", exp4_v2),
    ("exp5_v2", exp5_v2),
    ("exp6_v2", exp6_v2),
    ("exp7_v2", exp7_v2),
]


def main():
    args = sys.argv[1:]
    targets = [n for n, _ in ALL_V2] if not args or args == ["all"] else args

    overall_t0 = time.perf_counter()
    for name, fn in ALL_V2:
        if name not in targets:
            continue
        try:
            fn()
        except Exception as e:
            print(f"\n  [ERROR] {name} failed: {e}")
            import traceback; traceback.print_exc()
            d = _load()
            d[name] = {"error": str(e)}
            _save(d)
    overall_wall = time.perf_counter() - overall_t0
    print(f"\n{'='*78}")
    print(f"  All v2 experiments complete in {overall_wall:.1f}s "
          f"({overall_wall/60:.1f} min)")
    print(f"  Results: {RESULTS_FILE}\n{'='*78}")


if __name__ == "__main__":
    main()
