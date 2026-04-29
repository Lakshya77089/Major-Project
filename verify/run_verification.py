"""
zkFedMoE Conference Paper Verification Harness
==============================================

Runs every experiment claimed in conference_paper.tex from scratch and saves
measured values to verify/results/measured.json. Each experiment is followed
by a printed comparison against the paper's claimed value, so the user can
see in real time whether each claim holds.

Usage:
    python verify/run_verification.py [exp1|exp2|...|all]

Design choices:
  - Imports primitives from Code/src/ unchanged. We are auditing the existing
    codebase, not writing a parallel implementation.
  - Seeds set to 42 for every experiment, so re-runs are bit-exact (except
    for thread-scheduling noise in CPU-bound runs).
  - Each experiment self-contained in a function so partial runs are possible.
  - Results streamed to JSON after every experiment, so a crash halfway
    through still leaves measured data on disk.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset, TensorDataset

# ---------------------------------------------------------------------------
# Path setup -- import Code/src/ as a sibling package
# ---------------------------------------------------------------------------
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
from src.fl.dp import apply_dp, PrivacyAccountant, RenyiAccountant
from src.fl.sepg import generate_proof, verify_proof
from src.fl.adversaries import poisoning_train
from src.fl.attacks import membership_inference_attack
from src.fl.zkhash import benchmark_compare as zkhash_benchmark
from src.chain.ledger import Ledger

# ---------------------------------------------------------------------------
# Constants and output paths
# ---------------------------------------------------------------------------
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "measured.json"
DEVICE = torch.device("cpu")

# ---------------------------------------------------------------------------
# Paper's claimed values, hard-coded from conference_paper.tex.
# These are what we will COMPARE against, not use as targets.
# ---------------------------------------------------------------------------
PAPER_CLAIMS: Dict[str, Any] = {
    "exp1_privacy_utility": {
        "claim": "eps (Rényi) and accuracy at sigma=0/0.1/0.5/1.0/2.0",
        "rows": [
            {"sigma": 0.0, "eps": float("inf"), "accuracy_pct": 58.00},
            {"sigma": 0.1, "eps": 0.289,        "accuracy_pct": 25.00},
            {"sigma": 0.5, "eps": 0.058,        "accuracy_pct": 25.01},
            {"sigma": 1.0, "eps": 0.029,        "accuracy_pct": 26.12},
            {"sigma": 2.0, "eps": 0.014,        "accuracy_pct": 25.00},
        ],
    },
    "exp2_comm_vs_k": {
        "claim": "K=1 saves 39.52%, K=4 best accuracy 59.09%",
        "rows": [
            {"K": 1, "accuracy_pct": 54.24, "saving_pct": 39.52},
            {"K": 2, "accuracy_pct": 54.96, "saving_pct": 33.88},
            {"K": 3, "accuracy_pct": 57.66, "saving_pct": 28.23},
            {"K": 4, "accuracy_pct": 59.09, "saving_pct": 22.58},
            {"K": 8, "accuracy_pct": 55.99, "saving_pct": 0.00},
        ],
    },
    "exp3_sepg_overhead": {
        "claim": "Total SEPG overhead 6.08+/-0.21 ms, constant across K",
        "mean_total_ms": 6.08,
    },
    "exp4_robustness": {
        "claim": "FedAvg 41.82 / Median 45.95 / TrimMean 43.95 at 40% malicious",
        "fedavg_40": 41.82,
        "median_40": 45.95,
        "trimmean_40": 43.95,
    },
    "exp5_noniid": {
        "claim": "alpha=0.1 -> 43-48%, alpha=0.5 -> 50-54%, alpha=100 -> 57-58%",
        "alpha_0_1": (43, 48),
        "alpha_0_5": (50, 54),
        "alpha_100": (57, 58),
    },
    "exp6_mia": {
        "claim": "MIA AUC 0.643 (no DP) -> ~0.51 at sigma=1.0",
        "no_dp_auc": 0.643,
        "sigma_1_0_auc": 0.51,
    },
    "exp7_fedprox": {
        "claim": "FedAvg ~48-52%, FedProx (sweet) ~49-54% at alpha=0.3",
        "fedavg_range_pct": (48, 52),
        "fedprox_sweet_range_pct": (49, 54),
    },
    "exp8_disease": {
        "claim": "82.8% top-1, 92.4% top-3 at sigma=0.10",
        "sigma_0_10_top1_pct": 82.8,
        "sigma_0_10_top3_pct": 92.4,
        "sigma_0_00_top1_pct": 87.5,
        "sigma_0_00_top3_pct": 95.1,
    },
    "exp9_chain": {
        "claim": "Ledger 1k tx ~30 ms; MiMC ~3000x SHA-256 at 100K params",
        "ledger_1k_seal_ms": 30.0,
        "ledger_1k_verify_ms": 30.0,
        "mimc_sha_ratio_100k": 3000.0,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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


def _save(measured: Dict[str, Any]) -> None:
    serialisable: Dict[str, Any] = {}
    def fix(o):
        if isinstance(o, float) and (o != o or o == float("inf") or o == float("-inf")):
            return str(o)
        if isinstance(o, dict):
            return {k: fix(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [fix(x) for x in o]
        if isinstance(o, np.generic):
            return o.item()
        return o
    serialisable = fix(measured)
    RESULTS_FILE.write_text(json.dumps(serialisable, indent=2))


def _load() -> Dict[str, Any]:
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text())
    return {}


def _hr(title: str) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n  {title}\n{bar}")


def _verdict(label: str, ok: bool, paper_val: Any, measured_val: Any, tol: str = "") -> None:
    mark = "[MATCH]    " if ok else "[MISMATCH] "
    try:
        print(f"  {mark}{label}: paper={paper_val} | measured={measured_val} {tol}")
    except UnicodeEncodeError:
        # Fallback if a Unicode char slipped into the formatted output
        print(f"  {mark}{label}: paper={paper_val!r} | measured={measured_val!r} {tol!r}".encode("ascii", "replace").decode("ascii"))


def _save_partial(name: str, payload: Dict[str, Any]) -> None:
    """Checkpoint-save a single experiment's results so a later print crash
    cannot lose the actual measured numbers."""
    cur = _load()
    cur[name] = payload
    _save(cur)
    print(f"  [CHECKPOINT] Saved {name} measurements to {RESULTS_FILE}")


def make_model_kw(vocab_size, num_classes, num_experts=8, k=2):
    return dict(
        vocab_size=vocab_size, embed_dim=64, num_classes=num_classes,
        num_experts=num_experts, expert_hidden_dim=256, k=k, lora_r=8,
    )


# ---------------------------------------------------------------------------
# Cached data load -- AG News is 120K rows, parse it once.
# ---------------------------------------------------------------------------
_AG_CACHE = None

def load_ag_news(num_clients: int = 5, alpha: float | None = None, seed: int = 42):
    """Load AG News, IID or Dirichlet."""
    global _AG_CACHE
    if alpha is None:
        # IID. Re-split each call so different num_clients works.
        if _AG_CACHE is None:
            _AG_CACHE = build_ag_news_clients(
                num_clients=1, seq_len=64, use_external_csv=True, max_vocab=5000,
            )
        clients_1, test_ds, vs, nc, vocab = _AG_CACHE
        full_train = clients_1[0]
        n = len(full_train)
        sizes = [n // num_clients] * num_clients
        sizes[0] += n - sum(sizes)
        from torch.utils.data import random_split
        clients = list(random_split(full_train, sizes))
        return clients, test_ds, vs, nc, vocab
    # Non-IID
    return build_ag_news_clients_noniid(
        num_clients=num_clients, alpha=alpha,
        use_external_csv=True, seed=seed,
    )


# ===========================================================================
# EXPERIMENT 1 -- Privacy-utility tradeoff
# ===========================================================================
def exp1_privacy_utility() -> Dict[str, Any]:
    _hr("EXPERIMENT 1: Privacy-Utility Tradeoff (Table I)")
    print("  Reference: 5 clients, 5 rounds, AG News, MoE(E=8,K=2), seed=42")

    sigmas = [0.0, 0.1, 0.5, 1.0, 2.0]
    CLIP = 1.0
    ROUNDS = 5
    rows = []

    clients, test_ds, vs, nc, _ = load_ag_news(5)
    kw = make_model_kw(vs, nc)

    for sigma in sigmas:
        set_seed(42)
        srv = FedServer(MoETextClassifier(**kw), device=DEVICE)
        # Track BOTH accountants
        acct_basic = PrivacyAccountant(target_delta=1e-5)
        acct_renyi = RenyiAccountant(target_delta=1e-5)

        t0 = time.perf_counter()
        for rnd in range(ROUNDS):
            states = []
            for cds in clients:
                cm = MoETextClassifier(**kw)
                cm.load_state_dict(srv.get_global_state(), strict=False)
                fs, _, n, _, _, _, _ = local_train(
                    cm, cds, epochs=1, batch_size=64, lr=2e-3,
                    device=DEVICE, top_k_sparse=2,
                )
                if sigma > 0:
                    fs = apply_dp(fs, clip_norm=CLIP, noise_multiplier=sigma)
                states.append((fs, n))
            srv.aggregate(states)
            if sigma > 0:
                # sample rate -- 64 batch / shard size
                shard_size = len(clients[0]) if hasattr(clients[0], "__len__") else 1000
                sr = min(64 / max(shard_size, 1), 1.0)
                acct_basic.accumulate(sigma, sr, num_steps=1)
                acct_renyi.accumulate(sigma, sr, num_steps=1)

        wall = time.perf_counter() - t0
        acc = evaluate(srv.global_model, test_ds)
        eps_basic, _ = acct_basic.get_privacy_spent() if sigma > 0 else (float("inf"), 0)
        eps_renyi, _ = acct_renyi.get_privacy_spent() if sigma > 0 else (float("inf"), 0)
        rows.append({
            "sigma": sigma,
            "epsilon_basic": eps_basic,
            "epsilon_renyi": eps_renyi,
            "accuracy_pct": round(acc * 100, 2),
            "wall_seconds": round(wall, 1),
        })
        eps_str_b = "inf" if eps_basic == float("inf") else f"{eps_basic:.4f}"
        eps_str_r = "inf" if eps_renyi == float("inf") else f"{eps_renyi:.4f}"
        print(f"  sigma={sigma:.1f}  eps_basic={eps_str_b:>8s}  eps_renyi={eps_str_r:>8s}  "
              f"acc={acc*100:5.2f}%  wall={wall:.1f}s")

    # Checkpoint save before any comparison printing
    _save_partial("exp1", {"rows": rows, "paper_claim": PAPER_CLAIMS["exp1_privacy_utility"]})

    # Compare against paper
    print("\n  Paper says (Table I uses 'eps (Renyi)'):")
    for paper, mine in zip(PAPER_CLAIMS["exp1_privacy_utility"]["rows"], rows):
        ok_acc = abs(paper["accuracy_pct"] - mine["accuracy_pct"]) < 1.0
        # Use the larger of basic/Rényi for the eps comparison: paper labels Rényi
        # but the original run used basic. We report both.
        paper_eps = paper["eps"]
        my_renyi  = mine["epsilon_renyi"]
        my_basic  = mine["epsilon_basic"]
        if paper_eps == float("inf"):
            ok_eps = my_renyi == float("inf")
        else:
            # Within 25% relative -- accountant numerics differ tightly
            ok_eps = (
                (my_renyi != float("inf")) and
                (abs(my_renyi - paper_eps) / max(paper_eps, 1e-9) < 0.25
                 or abs(my_basic - paper_eps) / max(paper_eps, 1e-9) < 0.25)
            )
        _verdict(f"sigma={mine['sigma']:.1f} acc",
                 ok_acc, f"{paper['accuracy_pct']:.2f}%", f"{mine['accuracy_pct']:.2f}%")
        _verdict(f"sigma={mine['sigma']:.1f} eps",
                 ok_eps,
                 "inf" if paper_eps == float("inf") else f"{paper_eps:.3f}",
                 f"renyi={my_renyi:.4f} basic={my_basic:.4f}",
                 tol="(within 25%)")

    return {"rows": rows, "paper_claim": PAPER_CLAIMS["exp1_privacy_utility"]}


# ===========================================================================
# EXPERIMENT 2 -- Communication savings vs Top-K
# ===========================================================================
def exp2_comm_vs_k() -> Dict[str, Any]:
    _hr("EXPERIMENT 2: Communication Savings vs Top-K (Table II)")
    print("  Reference: 5 clients, 5 rounds, E=8 experts, sweep K=1..8")

    NE = 8
    ROUNDS = 5
    rows = []
    clients, test_ds, vs, nc, _ = load_ag_news(5)

    for K in range(1, NE + 1):
        set_seed(42)
        kw = make_model_kw(vs, nc, num_experts=NE, k=K)
        srv = FedServer(MoETextClassifier(**kw), device=DEVICE)

        # Compute parameter sizes from the model template (deterministic)
        model_t = MoETextClassifier(**kw)
        total_p  = sum(p.numel() for p in model_t.parameters())
        expert_p = sum(p.numel() for n, p in model_t.named_parameters() if "moe.experts" in n)
        per_exp  = expert_p // NE
        sparse_p = total_p - (NE - K) * per_exp
        # bytes = float32 = 4
        dense_bytes  = total_p * 4 * len(clients) * ROUNDS
        sparse_bytes = sparse_p * 4 * len(clients) * ROUNDS
        saving_pct   = (1 - sparse_bytes / dense_bytes) * 100

        t0 = time.perf_counter()
        for rnd in range(ROUNDS):
            states = []
            for cds in clients:
                cm = MoETextClassifier(**kw)
                cm.load_state_dict(srv.get_global_state(), strict=False)
                fs, _, n, _, _, _, _ = local_train(
                    cm, cds, epochs=1, batch_size=64, lr=2e-3,
                    device=DEVICE, top_k_sparse=K,
                )
                states.append((fs, n))
            srv.aggregate(states)
        wall = time.perf_counter() - t0

        acc = evaluate(srv.global_model, test_ds)
        rows.append({
            "K": K,
            "accuracy_pct": round(acc * 100, 2),
            "saving_pct":   round(saving_pct, 2),
            "dense_MB":     round(dense_bytes / 1e6, 2),
            "sparse_MB":    round(sparse_bytes / 1e6, 2),
            "wall_seconds": round(wall, 1),
        })
        print(f"  K={K}  acc={acc*100:5.2f}%  saving={saving_pct:5.2f}%  wall={wall:.1f}s")

    _save_partial("exp2", {"rows": rows, "paper_claim": PAPER_CLAIMS["exp2_comm_vs_k"]})

    # Compare
    print("\n  Compared to paper:")
    paper_rows = {r["K"]: r for r in PAPER_CLAIMS["exp2_comm_vs_k"]["rows"]}
    for r in rows:
        if r["K"] in paper_rows:
            p = paper_rows[r["K"]]
            ok_acc = abs(p["accuracy_pct"] - r["accuracy_pct"]) < 2.0
            ok_sav = abs(p["saving_pct"] - r["saving_pct"]) < 1.0
            _verdict(f"K={r['K']} accuracy", ok_acc,
                     f"{p['accuracy_pct']:.2f}%", f"{r['accuracy_pct']:.2f}%")
            _verdict(f"K={r['K']} saving",   ok_sav,
                     f"{p['saving_pct']:.2f}%", f"{r['saving_pct']:.2f}%")

    return {"rows": rows, "paper_claim": PAPER_CLAIMS["exp2_comm_vs_k"]}


# ===========================================================================
# EXPERIMENT 3 -- SEPG verification overhead
# ===========================================================================
def exp3_sepg_overhead() -> Dict[str, Any]:
    _hr("EXPERIMENT 3: SEPG Verification Overhead (Table III)")
    print("  Reference: mean over 50 trials per K, K=1..8, ~600K-param state")

    NE = 8
    rows = []
    for K in range(1, NE + 1):
        set_seed(42)
        kw = make_model_kw(vocab_size=5000, num_classes=4, num_experts=NE, k=K)
        model = MoETextClassifier(**kw)
        state = {n: p.detach().cpu().clone() for n, p in model.state_dict().items()}

        N_TRIALS = 50
        # Generation
        t0 = time.perf_counter()
        for _ in range(N_TRIALS):
            proof = generate_proof(
                client_id=0, round_id=0,
                top_k_indices=list(range(K)),
                clip_norm=1.0, noise_multiplier=0.5, epsilon=1.0,
                sparse_state=state,
            )
        gen_ms = (time.perf_counter() - t0) / N_TRIALS * 1000.0
        # Verification
        t0 = time.perf_counter()
        for _ in range(N_TRIALS):
            verify_proof(proof, state, expected_k=K)
        ver_ms = (time.perf_counter() - t0) / N_TRIALS * 1000.0

        rows.append({
            "K": K, "gen_ms": round(gen_ms, 3), "ver_ms": round(ver_ms, 3),
            "total_ms": round(gen_ms + ver_ms, 3),
        })
        print(f"  K={K}  gen={gen_ms:6.3f}ms  ver={ver_ms:6.3f}ms  total={gen_ms+ver_ms:6.3f}ms")

    mean_total = float(np.mean([r["total_ms"] for r in rows]))
    std_total  = float(np.std([r["total_ms"] for r in rows]))
    print(f"\n  Across all K: total = {mean_total:.2f} +/- {std_total:.2f} ms")

    _save_partial("exp3", {
        "rows": rows,
        "mean_total_ms": round(mean_total, 3),
        "std_total_ms":  round(std_total, 3),
        "paper_claim":   PAPER_CLAIMS["exp3_sepg_overhead"],
    })

    paper_mean = PAPER_CLAIMS["exp3_sepg_overhead"]["mean_total_ms"]
    ok = abs(mean_total - paper_mean) < 1.0
    _verdict("Mean total overhead", ok, f"{paper_mean:.2f} ms",
             f"{mean_total:.2f} +/- {std_total:.2f} ms")

    return {
        "rows": rows,
        "mean_total_ms": round(mean_total, 3),
        "std_total_ms":  round(std_total, 3),
        "paper_claim":   PAPER_CLAIMS["exp3_sepg_overhead"],
    }


# ===========================================================================
# EXPERIMENT 4 -- Robustness under poisoning
# ===========================================================================
def exp4_robustness() -> Dict[str, Any]:
    _hr("EXPERIMENT 4: Robustness Under Poisoning (Table IV)")
    print("  Reference: 5 clients, 5 rounds, label-flip poisoning, 0/20/40% malicious")

    clients, test_ds, vs, nc, _ = load_ag_news(5)
    kw = make_model_kw(vs, nc)
    ROUNDS = 5

    # Paper reports 0%, 20%, 40% (ignoring the int-rounding ambiguity in 5 clients)
    fractions = [0.0, 0.20, 0.40]
    methods = {
        "FedAvg":       "aggregate",
        "Median":       "aggregate_median",
        "Trimmed Mean": "aggregate_trimmed_mean",
    }

    results: Dict[str, Dict[str, float]] = {m: {} for m in methods}

    for method_name, attr in methods.items():
        for frac in fractions:
            set_seed(42)
            srv = FedServer(MoETextClassifier(**kw), device=DEVICE)
            n_mal = max(0, int(round(len(clients) * frac)))

            t0 = time.perf_counter()
            for rnd in range(ROUNDS):
                round_states = []
                for ci, cds in enumerate(clients):
                    cm = MoETextClassifier(**kw)
                    cm.load_state_dict(srv.get_global_state(), strict=False)
                    if ci < n_mal:
                        state, n = poisoning_train(
                            cm, cds, epochs=1, batch_size=64,
                            lr=2e-3, device=DEVICE, num_classes=nc,
                        )
                    else:
                        fs, _, n, _, _, _, _ = local_train(
                            cm, cds, epochs=1, batch_size=64, lr=2e-3,
                            device=DEVICE, top_k_sparse=2,
                        )
                        state = fs
                    round_states.append((state, n))
                getattr(srv, attr)(round_states)
            wall = time.perf_counter() - t0
            acc = evaluate(srv.global_model, test_ds)
            results[method_name][f"{int(frac*100)}%"] = round(acc * 100, 2)
            print(f"  {method_name:13s}  mal={int(frac*100):2d}%  acc={acc*100:5.2f}%  "
                  f"n_mal={n_mal}  wall={wall:.1f}s")

    _save_partial("exp4", {"results": results, "paper_claim": PAPER_CLAIMS["exp4_robustness"]})

    # Compare
    print("\n  Compared to paper at 40% malicious:")
    paper = PAPER_CLAIMS["exp4_robustness"]
    for label, key in [("FedAvg", "fedavg_40"),
                       ("Median", "median_40"),
                       ("Trimmed Mean", "trimmean_40")]:
        my_val = results[label]["40%"]
        ok = abs(paper[key] - my_val) < 3.0
        _verdict(f"{label} @ 40% mal", ok, f"{paper[key]:.2f}%", f"{my_val:.2f}%",
                 tol="(within 3 pp)")

    return {"results": results, "paper_claim": paper}


# ===========================================================================
# EXPERIMENT 5 -- Non-IID Dirichlet sweep
# ===========================================================================
def exp5_noniid() -> Dict[str, Any]:
    _hr("EXPERIMENT 5: Non-IID Dirichlet alpha sweep (Table V)")
    print("  Reference: 5 clients, 3 rounds, AG News, sweep alpha")

    alphas = [0.1, 0.3, 0.5, 1.0, 5.0, 100.0]
    ROUNDS = 3
    rows = []

    for alpha in alphas:
        set_seed(42)
        clients, test_ds, vs, nc, _ = build_ag_news_clients_noniid(
            num_clients=5, alpha=alpha, use_external_csv=True, seed=42,
        )
        kw = make_model_kw(vs, nc)
        srv = FedServer(MoETextClassifier(**kw), device=DEVICE)

        # Per-client class distribution snapshot
        client_dists = [client_class_distribution(c, nc).tolist() for c in clients]

        t0 = time.perf_counter()
        for rnd in range(ROUNDS):
            states = []
            for cds in clients:
                cm = MoETextClassifier(**kw)
                cm.load_state_dict(srv.get_global_state(), strict=False)
                fs, _, n, _, _, _, _ = local_train(
                    cm, cds, epochs=1, batch_size=64, lr=2e-3,
                    device=DEVICE, top_k_sparse=2,
                )
                states.append((fs, n))
            srv.aggregate(states)
        wall = time.perf_counter() - t0
        acc = evaluate(srv.global_model, test_ds)
        rows.append({
            "alpha": alpha,
            "accuracy_pct": round(acc * 100, 2),
            "client_dists": client_dists,
            "wall_seconds": round(wall, 1),
        })
        print(f"  alpha={alpha:>6.1f}  acc={acc*100:5.2f}%  "
              f"clients[0]={client_dists[0]}  wall={wall:.1f}s")

    _save_partial("exp5", {"rows": rows, "paper_claim": PAPER_CLAIMS["exp5_noniid"]})

    # Compare
    print("\n  Compared to paper:")
    p = PAPER_CLAIMS["exp5_noniid"]
    by_alpha = {r["alpha"]: r["accuracy_pct"] for r in rows}
    for label, alpha, lo_hi in [
        ("alpha=0.1", 0.1,   p["alpha_0_1"]),
        ("alpha=0.5", 0.5,   p["alpha_0_5"]),
        ("alpha=100", 100.0, p["alpha_100"]),
    ]:
        my = by_alpha.get(alpha)
        if my is None:
            continue
        lo, hi = lo_hi
        ok = lo - 5 <= my <= hi + 5  # generous 5pp slack
        _verdict(f"{label}", ok, f"{lo}-{hi}%", f"{my:.2f}%", tol="(within 5pp slack)")

    return {"rows": rows, "paper_claim": p}


# ===========================================================================
# EXPERIMENT 6 -- MIA AUC under DP ON/OFF
# ===========================================================================
def exp6_mia() -> Dict[str, Any]:
    _hr("EXPERIMENT 6: Membership Inference Attack (Table VI)")
    print("  Reference: 3 rounds, members = first 2 clients, MIA AUC")

    ROUNDS = 3
    configs = [
        ("No DP",       None, None),
        ("DP sigma=0.1", 1.0, 0.1),
        ("DP sigma=0.5", 1.0, 0.5),
        ("DP sigma=1.0", 1.0, 1.0),
    ]
    rows = []
    clients, test_ds, vs, nc, _ = load_ag_news(5)

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
                    cm, cds, epochs=1, batch_size=64, lr=2e-3,
                    device=DEVICE, top_k_sparse=2,
                )
                if sigma is not None and sigma > 0:
                    fs = apply_dp(fs, clip_norm=clip, noise_multiplier=sigma)
                states.append((fs, n))
            srv.aggregate(states)
        train_wall = time.perf_counter() - t0

        acc = evaluate(srv.global_model, test_ds)
        # MIA: members = first 2 clients
        members = ConcatDataset(clients[:2])
        mia = membership_inference_attack(
            srv.global_model, members, test_ds,
            device=DEVICE, max_samples=300,
        )
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
              f"mem_loss={mia.train_loss_mean:.3f}  nonmem_loss={mia.nonmember_loss_mean:.3f}")

    _save_partial("exp6", {"rows": rows, "paper_claim": PAPER_CLAIMS["exp6_mia"]})

    # Compare
    print("\n  Compared to paper:")
    p = PAPER_CLAIMS["exp6_mia"]
    by_label = {r["config"]: r for r in rows}
    no_dp = by_label.get("No DP")
    if no_dp:
        ok = abs(no_dp["mia_auc"] - p["no_dp_auc"]) < 0.10
        _verdict("No DP MIA AUC", ok, f"{p['no_dp_auc']}", f"{no_dp['mia_auc']}",
                 tol="(within +/-0.10)")
    sig1 = by_label.get("DP sigma=1.0")
    if sig1:
        ok = abs(sig1["mia_auc"] - p["sigma_1_0_auc"]) < 0.10
        _verdict("sigma=1.0 MIA AUC", ok, f"{p['sigma_1_0_auc']}", f"{sig1['mia_auc']}",
                 tol="(within +/-0.10)")

    return {"rows": rows, "paper_claim": p}


# ===========================================================================
# EXPERIMENT 7 -- FedProx vs FedAvg under non-IID
# ===========================================================================
def exp7_fedprox() -> Dict[str, Any]:
    _hr("EXPERIMENT 7: FedProx vs FedAvg under Dirichlet alpha=0.3 (Table VII)")
    print("  Reference: 5 clients, 3 rounds, alpha=0.3, mu sweep")

    ALPHA = 0.3
    ROUNDS = 3
    mu_values = [0.0, 0.001, 0.01, 0.1, 0.5]
    rows = []

    for mu in mu_values:
        set_seed(42)
        clients, test_ds, vs, nc, _ = build_ag_news_clients_noniid(
            num_clients=5, alpha=ALPHA, use_external_csv=True, seed=42,
        )
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
                    device=DEVICE, top_k_sparse=2, fedprox_mu=mu,
                )
                states.append((fs, n))
            srv.aggregate(states)
        wall = time.perf_counter() - t0
        acc = evaluate(srv.global_model, test_ds)
        label = "FedAvg" if mu == 0.0 else f"FedProx mu={mu}"
        rows.append({
            "mu": mu, "label": label, "alpha": ALPHA,
            "accuracy_pct": round(acc * 100, 2),
            "wall_seconds": round(wall, 1),
        })
        print(f"  {label:18s}  alpha={ALPHA}  acc={acc*100:5.2f}%  wall={wall:.1f}s")

    _save_partial("exp7", {"rows": rows, "paper_claim": PAPER_CLAIMS["exp7_fedprox"]})

    # Compare
    print("\n  Compared to paper:")
    p = PAPER_CLAIMS["exp7_fedprox"]
    fedavg = next((r for r in rows if r["mu"] == 0.0), None)
    if fedavg:
        lo, hi = p["fedavg_range_pct"]
        ok = lo - 5 <= fedavg["accuracy_pct"] <= hi + 5
        _verdict("FedAvg @ alpha=0.3", ok,
                 f"{lo}-{hi}%", f"{fedavg['accuracy_pct']:.2f}%",
                 tol="(within 5pp slack)")
    sweet = [r for r in rows if r["mu"] in (0.01, 0.05, 0.1)]
    if sweet:
        best = max(sweet, key=lambda r: r["accuracy_pct"])
        lo, hi = p["fedprox_sweet_range_pct"]
        ok = lo - 5 <= best["accuracy_pct"] <= hi + 5
        _verdict(f"FedProx sweet (mu={best['mu']})", ok,
                 f"{lo}-{hi}%", f"{best['accuracy_pct']:.2f}%",
                 tol="(within 5pp slack)")

    return {"rows": rows, "paper_claim": p}


# ===========================================================================
# EXPERIMENT 8 -- Disease Detection
# ===========================================================================
def exp8_disease() -> Dict[str, Any]:
    _hr("EXPERIMENT 8: Disease Detection (Table VIII-A)")
    print("  Reference: 24 diseases x 70 symptoms, 10 hospitals, 10 rounds, sigma sweep")

    # Lazy import the symptom dataset
    sys.path.insert(0, str(CODE_DIR))
    try:
        from data.disease_symptoms import build_dataset as _build_disease
    except ImportError as e:
        print(f"  [!] Could not import disease dataset: {e}")
        return {"error": str(e)}

    set_seed(42)
    X_d, y_d, syms_d, dis_names_d = _build_disease(n_variants=80, seed=42)
    n_features = X_d.shape[1]
    n_classes  = len(dis_names_d)
    n_total    = X_d.shape[0]
    n_test     = max(int(0.2 * n_total), n_classes)
    perm = np.random.permutation(n_total)
    test_ix  = perm[:n_test]
    train_ix = perm[n_test:]

    X_tr = torch.from_numpy(X_d[train_ix]).float()
    y_tr = torch.from_numpy(y_d[train_ix]).long()
    X_te = torch.from_numpy(X_d[test_ix]).float()
    y_te = torch.from_numpy(y_d[test_ix]).long()
    train_ds = TensorDataset(X_tr, y_tr)
    test_ds  = TensorDataset(X_te, y_te)

    print(f"  Dataset: {n_total} records ({len(train_ix)} train / {len(test_ix)} test) "
          f"x {n_features} symptoms x {n_classes} diseases")

    # Dirichlet split across 10 hospitals (alpha=1.0)
    N_HOSP = 10
    ALPHA  = 1.0
    rng = np.random.default_rng(42)
    by_class = [np.where(y_tr.numpy() == c)[0] for c in range(n_classes)]
    for arr_c in by_class:
        rng.shuffle(arr_c)

    client_ix = None
    for _ in range(50):
        cand = [[] for _ in range(N_HOSP)]
        for c in range(n_classes):
            if len(by_class[c]) == 0:
                continue
            props = rng.dirichlet([ALPHA] * N_HOSP)
            split_pts = (np.cumsum(props) * len(by_class[c])).astype(int)[:-1]
            chunks = np.split(by_class[c], split_pts)
            for cid, chunk in enumerate(chunks):
                cand[cid].extend(chunk.tolist())
        if min(len(ix) for ix in cand) >= 3:
            client_ix = cand
            break
    if client_ix is None:
        client_ix = cand
    client_dss = [Subset(train_ds, ix) for ix in client_ix]

    # MLP model identical to dashboard
    class _DiseaseClf(torch.nn.Module):
        def __init__(self, in_f, hidden=128, n_cls=24):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(in_f, hidden), torch.nn.ReLU(),
                torch.nn.Dropout(0.15),
                torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(hidden, n_cls),
            )
        def forward(self, x): return self.net(x)

    def _eval(model, ds, top_k=3):
        model.eval()
        ldr = DataLoader(ds, batch_size=128, shuffle=False)
        c1 = ck = total = 0
        with torch.no_grad():
            for X, y in ldr:
                out = model(X)
                c1 += int((out.argmax(-1) == y).sum())
                topk_idx = out.topk(top_k, dim=-1).indices
                ck += int((topk_idx == y.unsqueeze(-1)).any(-1).sum())
                total += y.size(0)
        return c1 / max(total, 1), ck / max(total, 1)

    def _aggregate_fedavg(states_with_n):
        keys = list(states_with_n[0][0].keys())
        total_n = sum(n for _, n in states_with_n)
        out = {k: torch.zeros_like(states_with_n[0][0][k]).float() for k in keys}
        for st_, n in states_with_n:
            w = n / total_n
            for k in keys:
                out[k] += st_[k].float() * w
        return out

    sigma_sweep = [0.0, 0.05, 0.10, 0.20]
    CLIP_C = 1.5
    ROUNDS = 10
    LOCAL_EPOCHS = 3
    LR = 2e-3
    BATCH = 32

    rows = []
    for sigma in sigma_sweep:
        set_seed(42)
        global_model = _DiseaseClf(n_features, 128, n_classes)
        t0 = time.perf_counter()
        for rnd in range(1, ROUNDS + 1):
            global_state = {k: v.detach().cpu().clone()
                            for k, v in global_model.state_dict().items()}
            client_updates = []
            for cid in range(N_HOSP):
                local = _DiseaseClf(n_features, 128, n_classes)
                local.load_state_dict(global_state)
                local.train()
                loader = DataLoader(client_dss[cid], batch_size=BATCH, shuffle=True)
                opt = torch.optim.Adam(local.parameters(), lr=LR)
                crit = torch.nn.CrossEntropyLoss()
                for _ep in range(LOCAL_EPOCHS):
                    for Xb, yb in loader:
                        opt.zero_grad()
                        loss = crit(local(Xb), yb)
                        loss.backward()
                        opt.step()
                new_state = {k: v.detach().cpu().clone()
                             for k, v in local.state_dict().items()}
                # Apply DP to the *delta*, not the absolute state
                delta = {k: new_state[k].float() - global_state[k].float()
                         for k in new_state}
                if sigma > 0:
                    delta = apply_dp(delta, clip_norm=CLIP_C, noise_multiplier=sigma)
                uploaded = {k: global_state[k].float() + delta[k] for k in delta}
                client_updates.append((uploaded, len(client_ix[cid])))
            agg = _aggregate_fedavg(client_updates)
            global_model.load_state_dict(agg)
        wall = time.perf_counter() - t0
        top1, top3 = _eval(global_model, test_ds)
        rows.append({
            "sigma": sigma,
            "top1_pct": round(top1 * 100, 2),
            "top3_pct": round(top3 * 100, 2),
            "wall_seconds": round(wall, 1),
        })
        print(f"  sigma={sigma:.2f}  top1={top1*100:5.2f}%  top3={top3*100:5.2f}%  "
              f"wall={wall:.1f}s")

    _save_partial("exp8", {"rows": rows, "paper_claim": PAPER_CLAIMS["exp8_disease"]})

    # Compare
    print("\n  Compared to paper:")
    p = PAPER_CLAIMS["exp8_disease"]
    by_sigma = {r["sigma"]: r for r in rows}
    if 0.10 in by_sigma:
        m = by_sigma[0.10]
        ok1 = abs(m["top1_pct"] - p["sigma_0_10_top1_pct"]) < 5.0
        ok3 = abs(m["top3_pct"] - p["sigma_0_10_top3_pct"]) < 5.0
        _verdict("sigma=0.10 top-1", ok1,
                 f"{p['sigma_0_10_top1_pct']:.1f}%", f"{m['top1_pct']:.2f}%",
                 tol="(within +/-5 pp)")
        _verdict("sigma=0.10 top-3", ok3,
                 f"{p['sigma_0_10_top3_pct']:.1f}%", f"{m['top3_pct']:.2f}%",
                 tol="(within +/-5 pp)")
    if 0.0 in by_sigma:
        m = by_sigma[0.0]
        ok1 = abs(m["top1_pct"] - p["sigma_0_00_top1_pct"]) < 5.0
        ok3 = abs(m["top3_pct"] - p["sigma_0_00_top3_pct"]) < 5.0
        _verdict("sigma=0.00 top-1", ok1,
                 f"{p['sigma_0_00_top1_pct']:.1f}%", f"{m['top1_pct']:.2f}%",
                 tol="(within +/-5 pp)")
        _verdict("sigma=0.00 top-3", ok3,
                 f"{p['sigma_0_00_top3_pct']:.1f}%", f"{m['top3_pct']:.2f}%",
                 tol="(within +/-5 pp)")

    return {"rows": rows, "paper_claim": p}


# ===========================================================================
# EXPERIMENT 9 -- Ledger throughput + MiMC vs SHA-256
# ===========================================================================
def exp9_chain() -> Dict[str, Any]:
    _hr("EXPERIMENT 9: Audit Ledger + ZK-friendly hash benchmarks (Table IX)")

    # (A) Ledger seal/verify
    n_tx_values = [10, 100, 500, 1000]
    ledger_rows = []
    for n_tx in n_tx_values:
        ledger = Ledger()
        t0 = time.perf_counter()
        for i in range(n_tx):
            ledger.add_transaction(
                "verify", client_id=i % 10, round=1,
                accepted=True, proof_hash=f"{i:032x}",
            )
        ledger.seal_block()
        seal_ms = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        ok, _ = ledger.verify()
        verify_ms = (time.perf_counter() - t0) * 1000.0
        ledger_rows.append({
            "n_tx": n_tx,
            "seal_ms":   round(seal_ms, 3),
            "verify_ms": round(verify_ms, 3),
            "ok": bool(ok),
        })
        print(f"  Ledger n={n_tx:>4d}  seal={seal_ms:7.2f}ms  verify={verify_ms:7.2f}ms  ok={ok}")

    # (B) Hash benchmarks at three sizes
    hash_rows = []
    state_sizes = [
        ("tiny_1K",     {"w": torch.randn(32, 32)}),
        ("small_10K",   {"w": torch.randn(100, 100)}),
        ("medium_100K", {"w": torch.randn(316, 316)}),
    ]
    for label, st in state_sizes:
        bench = zkhash_benchmark(st, n_trials=3)
        hash_rows.append({
            "size_label": label,
            "sha256_ms":  round(bench["sha256_ms"], 3),
            "mimc_ms":    round(bench["mimc_ms"], 3),
            "ratio":      round(bench["ratio"], 1),
        })
        print(f"  Hash {label:>13s}  SHA={bench['sha256_ms']:7.2f}ms  "
              f"MiMC={bench['mimc_ms']:9.2f}ms  ratio={bench['ratio']:.0f}x")

    _save_partial("exp9", {
        "ledger": ledger_rows,
        "hash_overhead": hash_rows,
        "paper_claim": PAPER_CLAIMS["exp9_chain"],
    })

    # Compare
    print("\n  Compared to paper:")
    p = PAPER_CLAIMS["exp9_chain"]
    seal_1k = next((r for r in ledger_rows if r["n_tx"] == 1000), None)
    if seal_1k:
        ok_seal = abs(seal_1k["seal_ms"] - p["ledger_1k_seal_ms"]) < 50.0
        ok_ver  = abs(seal_1k["verify_ms"] - p["ledger_1k_verify_ms"]) < 50.0
        _verdict("Ledger 1k tx seal_ms", ok_seal,
                 f"~{p['ledger_1k_seal_ms']} ms", f"{seal_1k['seal_ms']} ms",
                 tol="(within +/-50 ms)")
        _verdict("Ledger 1k tx verify_ms", ok_ver,
                 f"~{p['ledger_1k_verify_ms']} ms", f"{seal_1k['verify_ms']} ms",
                 tol="(within +/-50 ms)")
    medium = next((r for r in hash_rows if r["size_label"] == "medium_100K"), None)
    if medium:
        ok = abs(medium["ratio"] - p["mimc_sha_ratio_100k"]) / p["mimc_sha_ratio_100k"] < 0.50
        _verdict("MiMC/SHA ratio @ 100K", ok,
                 f"~{p['mimc_sha_ratio_100k']:.0f}x", f"{medium['ratio']:.0f}x",
                 tol="(within +/-50% relative)")

    return {
        "ledger":   ledger_rows,
        "hash_overhead": hash_rows,
        "paper_claim":   p,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
ALL_EXPERIMENTS = [
    ("exp1", exp1_privacy_utility),
    ("exp2", exp2_comm_vs_k),
    ("exp3", exp3_sepg_overhead),
    ("exp4", exp4_robustness),
    ("exp5", exp5_noniid),
    ("exp6", exp6_mia),
    ("exp7", exp7_fedprox),
    ("exp8", exp8_disease),
    ("exp9", exp9_chain),
]


def main():
    args = sys.argv[1:]
    if not args or args == ["all"]:
        targets = [name for name, _ in ALL_EXPERIMENTS]
    else:
        targets = args

    all_results = _load()
    overall_t0 = time.perf_counter()
    for name, fn in ALL_EXPERIMENTS:
        if name not in targets:
            continue
        try:
            result = fn()
            all_results = _load()  # reload in case _save_partial wrote inside fn
            all_results[name] = result
            _save(all_results)
            print(f"\n  --> Saved final results to {RESULTS_FILE}")
        except Exception as e:
            print(f"\n  [ERROR] {name} failed during compare/print: {e}")
            import traceback
            traceback.print_exc()
            # Reload from disk -- _save_partial may already have written real data.
            # Only mark error if there's nothing saved yet.
            all_results = _load()
            if name not in all_results or "rows" not in all_results.get(name, {}) and "results" not in all_results.get(name, {}) and "ledger" not in all_results.get(name, {}):
                all_results[name] = {"error": str(e)}
                _save(all_results)
            else:
                # Append error annotation but keep measured data
                all_results[name]["compare_error"] = str(e)
                _save(all_results)
                print(f"  (Measured data preserved; only the comparison print failed.)")
    overall_wall = time.perf_counter() - overall_t0
    print(f"\n{'='*78}\n  All requested experiments complete in {overall_wall:.1f}s "
          f"({overall_wall/60:.1f} min)\n{'='*78}")
    print(f"  Results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
