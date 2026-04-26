"""
zkFedMoE — Experiment Suite
============================
Generates the 4 core plots described in the report:
  1. Accuracy vs epsilon (privacy-utility tradeoff)
  2. Communication vs K (sparse savings sweep)
  3. Verification overhead vs K
  4. Accuracy vs % malicious clients (robustness)

Usage:
    python -m experiments.run_all_experiments
"""

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

# Allow running from Code/ directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import MoETextClassifier
from src.data import (
    build_ag_news_clients,
    build_ag_news_clients_noniid,
    client_class_distribution,
)
from src.fl.client import local_train
from src.fl.server import FedServer
from src.fl.dp import apply_dp, PrivacyAccountant, RenyiAccountant
from src.fl.sepg import generate_proof, verify_proof
from src.fl.adversaries import poisoning_train, freerider_train, sybil_clones
from src.fl.attacks import membership_inference_attack

PLOT_DIR = Path(__file__).resolve().parents[1] / "plots"
PLOT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cpu")


def set_seed(s=42):
    torch.manual_seed(s)


def evaluate(model, dataset, bs=64):
    loader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=0)
    model.to(DEVICE).eval()
    c = t = 0
    with torch.no_grad():
        for ids, lbl in loader:
            ids, lbl = ids.to(DEVICE), lbl.to(DEVICE)
            c += (model(ids).argmax(-1) == lbl).sum().item()
            t += lbl.size(0)
    return c / max(t, 1)


# ============================================================
# Shared: load AG News once
# ============================================================
def load_data(num_clients=5):
    return build_ag_news_clients(
        num_clients=num_clients, seq_len=64,
        use_external_csv=True, max_vocab=5000)


def make_model_kw(vocab_size, num_classes, num_experts=8, k=2):
    return dict(vocab_size=vocab_size, embed_dim=64, num_classes=num_classes,
                num_experts=num_experts, expert_hidden_dim=256, k=k, lora_r=8)


def fl_round(server, clients, model_kw, lr=2e-3, dp_clip=None, dp_noise=None, top_k=2):
    """Run one federated round. Returns (dense_states, sparse_states, expert_usages)."""
    dense_states, sparse_states = [], []
    for cds in clients:
        cm = MoETextClassifier(**model_kw)
        cm.load_state_dict(server.get_global_state(), strict=False)
        fs, ss, n, db, sb, tki, eu = local_train(
            cm, cds, epochs=1, batch_size=64, lr=lr,
            device=DEVICE, top_k_sparse=top_k)

        # Apply DP if requested
        if dp_clip is not None and dp_noise is not None:
            fs = apply_dp(fs, clip_norm=dp_clip, noise_multiplier=dp_noise)
            ss = apply_dp(ss, clip_norm=dp_clip, noise_multiplier=dp_noise)

        dense_states.append((fs, n))
        sparse_states.append((ss, n))

    return dense_states, sparse_states


# ============================================================
# EXPERIMENT 1: Accuracy vs Epsilon (Privacy-Utility Tradeoff)
# ============================================================
def experiment_privacy_utility():
    print("\n" + "=" * 60)
    print("  Experiment 1: Accuracy vs Epsilon (Privacy-Utility)")
    print("=" * 60)

    clients, test_ds, vs, nc, vocab = load_data(5)
    kw = make_model_kw(vs, nc)
    ROUNDS = 5
    CLIP_NORM = 1.0

    noise_mults = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]
    results = []

    for nm in noise_mults:
        set_seed()
        srv = FedServer(MoETextClassifier(**kw), device=DEVICE)
        acc_tracker = PrivacyAccountant(target_delta=1e-5)

        for rnd in range(ROUNDS):
            ds, _ = fl_round(srv, clients, kw, dp_clip=CLIP_NORM if nm > 0 else None,
                             dp_noise=nm if nm > 0 else None)
            srv.aggregate(ds)
            if nm > 0:
                sample_rate = 64 / (len(clients[0]) if hasattr(clients[0], '__len__') else 1000)
                acc_tracker.accumulate(nm, sample_rate, num_steps=1)

        acc = evaluate(srv.global_model, test_ds)
        eps, delta = acc_tracker.get_privacy_spent() if nm > 0 else (float("inf"), 0)
        eps_label = f"{eps:.2f}" if eps < 100 else "inf"
        results.append({"noise_mult": nm, "epsilon": eps, "accuracy": acc})
        print(f"  noise={nm:.1f}  eps={eps_label}  acc={acc:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    eps_vals = [r["epsilon"] if r["epsilon"] < 100 else 50 for r in results]
    accs = [r["accuracy"] for r in results]
    labels = [f"nm={r['noise_mult']}" for r in results]
    ax.plot(eps_vals, accs, "o-", linewidth=2, markersize=8)
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (eps_vals[i], accs[i]), textcoords="offset points",
                    xytext=(5, 8), fontsize=9)
    ax.set_xlabel("Privacy Budget (epsilon)", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("Privacy-Utility Tradeoff", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "exp1_privacy_utility.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: plots/exp1_privacy_utility.png")
    return results


# ============================================================
# EXPERIMENT 2: Communication vs K
# ============================================================
def experiment_comm_vs_k():
    print("\n" + "=" * 60)
    print("  Experiment 2: Communication Cost vs Top-K")
    print("=" * 60)

    clients, test_ds, vs, nc, vocab = load_data(5)
    NE = 8
    ROUNDS = 5
    results = []

    for K in range(1, NE + 1):
        set_seed()
        kw = make_model_kw(vs, nc, num_experts=NE, k=min(K, NE))
        srv = FedServer(MoETextClassifier(**kw), device=DEVICE)

        total_dense = total_sparse = 0
        for rnd in range(ROUNDS):
            ds, ss = fl_round(srv, clients, kw, top_k=K)
            srv.aggregate(ds)
            for (_, _), (s, _) in zip(ds, ss):
                pass
            # Compute bytes from first client
            cm = MoETextClassifier(**kw)
            total_p = sum(p.numel() for p in cm.parameters())
            expert_p = sum(p.numel() for n, p in cm.named_parameters() if "moe.experts" in n)
            per_exp = expert_p // NE
            dense_p = total_p
            sparse_p = total_p - (NE - K) * per_exp
            total_dense += dense_p * 4 * len(clients)
            total_sparse += sparse_p * 4 * len(clients)

        acc = evaluate(srv.global_model, test_ds)
        saving = (1 - total_sparse / total_dense) * 100
        results.append({"K": K, "accuracy": acc, "dense_MB": total_dense / 1e6,
                        "sparse_MB": total_sparse / 1e6, "saving_pct": saving})
        print(f"  K={K}  acc={acc:.4f}  saving={saving:.1f}%")

    # Plot
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ks = [r["K"] for r in results]
    savings = [r["saving_pct"] for r in results]
    accs = [r["accuracy"] for r in results]

    color1 = "#4C72B0"
    ax1.bar(ks, savings, color=color1, alpha=0.7, label="Comm Saving %")
    ax1.set_xlabel("Top-K (experts sent)", fontsize=12)
    ax1.set_ylabel("Communication Saving (%)", fontsize=12, color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "#DD8452"
    ax2.plot(ks, accs, "o-", color=color2, linewidth=2, markersize=8, label="Accuracy")
    ax2.set_ylabel("Test Accuracy", fontsize=12, color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0, 1)

    ax1.set_title("Communication Savings vs Top-K", fontsize=13)
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.88))
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "exp2_comm_vs_k.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: plots/exp2_comm_vs_k.png")
    return results


# ============================================================
# EXPERIMENT 3: Verification Overhead vs K
# ============================================================
def experiment_verification_overhead():
    print("\n" + "=" * 60)
    print("  Experiment 3: Verification Overhead vs Top-K")
    print("=" * 60)

    NE = 8
    results = []

    for K in range(1, NE + 1):
        kw = make_model_kw(vocab_size=5000, num_classes=4, num_experts=NE, k=K)
        model = MoETextClassifier(**kw)
        state = {n: p.detach().cpu().clone() for n, p in model.state_dict().items()}

        # Measure proof generation time
        N_TRIALS = 50
        t0 = time.perf_counter()
        for _ in range(N_TRIALS):
            proof = generate_proof(
                client_id=0, round_id=0,
                top_k_indices=list(range(K)),
                clip_norm=1.0, noise_multiplier=0.5, epsilon=1.0,
                sparse_state=state)
        gen_time = (time.perf_counter() - t0) / N_TRIALS * 1000  # ms

        # Measure verification time
        t0 = time.perf_counter()
        for _ in range(N_TRIALS):
            verify_proof(proof, state, expected_k=K)
        ver_time = (time.perf_counter() - t0) / N_TRIALS * 1000  # ms

        results.append({"K": K, "gen_ms": gen_time, "ver_ms": ver_time,
                        "total_ms": gen_time + ver_time})
        print(f"  K={K}  gen={gen_time:.2f}ms  verify={ver_time:.2f}ms  total={gen_time + ver_time:.2f}ms")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ks = [r["K"] for r in results]
    gen = [r["gen_ms"] for r in results]
    ver = [r["ver_ms"] for r in results]
    ax.bar(ks, gen, label="Proof Generation", color="#4C72B0", alpha=0.8)
    ax.bar(ks, ver, bottom=gen, label="Proof Verification", color="#DD8452", alpha=0.8)
    ax.set_xlabel("Top-K (experts in proof)", fontsize=12)
    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_title("SEPG Verification Overhead vs Top-K", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "exp3_verification_overhead.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: plots/exp3_verification_overhead.png")
    return results


# ============================================================
# EXPERIMENT 4: Robustness Under Attacks
# ============================================================
def experiment_robustness():
    print("\n" + "=" * 60)
    print("  Experiment 4: Robustness Under Attacks")
    print("=" * 60)

    clients, test_ds, vs, nc, vocab = load_data(5)
    kw = make_model_kw(vs, nc)
    ROUNDS = 5

    malicious_fractions = [0.0, 0.1, 0.2, 0.3, 0.4]
    agg_methods = {
        "FedAvg": "aggregate",
        "Median": "aggregate_median",
        "Trimmed Mean": "aggregate_trimmed_mean",
    }

    all_results = {}

    for method_name, method_attr in agg_methods.items():
        method_results = []
        for mf in malicious_fractions:
            set_seed()
            srv = FedServer(MoETextClassifier(**kw), device=DEVICE)
            n_mal = max(0, int(len(clients) * mf))

            for rnd in range(ROUNDS):
                round_states = []
                for ci, cds in enumerate(clients):
                    cm = MoETextClassifier(**kw)
                    cm.load_state_dict(srv.get_global_state(), strict=False)

                    if ci < n_mal:
                        # Malicious: poisoning attack
                        state, n = poisoning_train(
                            cm, cds, epochs=1, batch_size=64,
                            lr=2e-3, device=DEVICE, num_classes=nc)
                    else:
                        # Honest
                        fs, _, n, _, _, _, _ = local_train(
                            cm, cds, epochs=1, batch_size=64,
                            lr=2e-3, device=DEVICE)
                        state = fs

                    round_states.append((state, n))

                getattr(srv, method_attr)(round_states)

            acc = evaluate(srv.global_model, test_ds)
            method_results.append({"malicious_pct": mf * 100, "accuracy": acc})
            print(f"  {method_name:15s}  mal={mf*100:.0f}%  acc={acc:.4f}")

        all_results[method_name] = method_results

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"FedAvg": "#4C72B0", "Median": "#55A868", "Trimmed Mean": "#DD8452"}
    for method_name, res_list in all_results.items():
        xs = [r["malicious_pct"] for r in res_list]
        ys = [r["accuracy"] for r in res_list]
        ax.plot(xs, ys, "o-", label=method_name, color=colors.get(method_name, "gray"),
                linewidth=2, markersize=8)
    ax.set_xlabel("Malicious Clients (%)", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("Robustness: FedAvg vs Median vs Trimmed Mean", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "exp4_robustness.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: plots/exp4_robustness.png")
    return all_results


# ============================================================
# EXPERIMENT 5: Non-IID data (Dirichlet alpha sweep)
# ============================================================
def experiment_noniid():
    print("\n" + "=" * 60)
    print("  Experiment 5: Non-IID Data (Dirichlet alpha sweep)")
    print("=" * 60)

    ROUNDS = 3
    alphas = [0.1, 0.3, 0.5, 1.0, 5.0, 100.0]
    results = []

    for alpha in alphas:
        set_seed()
        clients, test_ds, vs, nc, _ = build_ag_news_clients_noniid(
            num_clients=5, alpha=alpha, use_external_csv=True, seed=42,
        )
        kw = make_model_kw(vs, nc)
        srv = FedServer(MoETextClassifier(**kw), device=DEVICE)

        # Record per-client class distribution for this alpha
        client_dists = []
        for c in clients:
            dist = client_class_distribution(c, nc).tolist()
            client_dists.append(dist)

        for rnd in range(ROUNDS):
            ds, _ = fl_round(srv, clients, kw)
            srv.aggregate(ds)

        acc = evaluate(srv.global_model, test_ds)
        results.append({
            "alpha": alpha, "accuracy": acc, "client_dists": client_dists,
        })
        print(f"  alpha={alpha:>5.1f}  acc={acc:.4f}  "
              f"clients[0]={client_dists[0]}")

    # Plot: alpha (log scale) vs accuracy
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = [r["alpha"] for r in results]
    ys = [r["accuracy"] for r in results]
    ax.plot(xs, ys, "o-", linewidth=2, markersize=8, color="#4C72B0")
    ax.set_xscale("log")
    ax.set_xlabel("Dirichlet α (log scale)  — lower = more non-IID", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("Non-IID Robustness: Accuracy vs Dirichlet α", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "exp5_noniid.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: plots/exp5_noniid.png")
    return results


# ============================================================
# EXPERIMENT 6: Membership Inference Attack (DP ON vs OFF)
# ============================================================
def experiment_mia():
    print("\n" + "=" * 60)
    print("  Experiment 6: Membership Inference Attack")
    print("=" * 60)

    from torch.utils.data import ConcatDataset

    ROUNDS = 3
    results = []
    # Wider range so we see the effect emerge with slightly more rounds
    configs = [
        ("No DP", None, None),
        ("DP σ=0.1", 1.0, 0.1),
        ("DP σ=0.5", 1.0, 0.5),
        ("DP σ=1.0", 1.0, 1.0),
    ]

    clients_base, test_ds_base, vs, nc, _ = load_data(5)

    for label, clip, sigma in configs:
        set_seed()
        kw = make_model_kw(vs, nc)
        srv = FedServer(MoETextClassifier(**kw), device=DEVICE)

        for rnd in range(ROUNDS):
            ds, _ = fl_round(
                srv, clients_base, kw,
                dp_clip=clip, dp_noise=sigma,
            )
            srv.aggregate(ds)

        acc = evaluate(srv.global_model, test_ds_base)

        # MIA: members = first 2 clients, non-members = test set
        members = ConcatDataset(clients_base[:2])
        mia = membership_inference_attack(
            srv.global_model, members, test_ds_base,
            device=DEVICE, max_samples=300,
        )
        results.append({
            "config": label,
            "clip": clip, "sigma": sigma,
            "accuracy": acc,
            "mia_auc": mia.auc,
            "mia_acc": mia.attack_accuracy,
            "train_loss": mia.train_loss_mean,
            "nonmember_loss": mia.nonmember_loss_mean,
        })
        print(f"  {label:>12s}  acc={acc:.4f}  "
              f"MIA AUC={mia.auc:.3f}  (member_loss={mia.train_loss_mean:.3f}, "
              f"nonmem_loss={mia.nonmember_loss_mean:.3f})")

    # Plot: MIA AUC with DP ON vs OFF
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [r["config"] for r in results]
    aucs = [r["mia_auc"] for r in results]
    accs = [r["accuracy"] for r in results]

    x = range(len(labels))
    ax.bar([i - 0.2 for i in x], aucs, width=0.4,
           color="#DD8452", label="MIA AUC (lower=better privacy)")
    ax.bar([i + 0.2 for i in x], accs, width=0.4,
           color="#4C72B0", label="Test Accuracy")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5,
               label="MIA random baseline (0.5)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title("Membership Inference: DP reduces leakage", fontsize=13)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "exp6_mia.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: plots/exp6_mia.png")
    return results


# ============================================================
# EXPERIMENT 7: FedProx vs FedAvg under non-IID data
# ============================================================
def experiment_fedprox():
    """
    Compare vanilla FedAvg vs. FedProx (mu>0) under Dirichlet non-IID.

    Hypothesis: at small alpha (skewed data), FedProx's proximal term
    keeps clients from drifting and improves convergence vs. vanilla FedAvg.
    """
    print("\n" + "=" * 60)
    print("  Experiment 7: FedProx vs FedAvg under non-IID data")
    print("=" * 60)

    ROUNDS = 3
    alpha = 0.3  # strong non-IID
    mu_values = [0.0, 0.001, 0.01, 0.1, 0.5]
    results = []

    for mu in mu_values:
        set_seed()
        clients, test_ds, vs, nc, _ = build_ag_news_clients_noniid(
            num_clients=5, alpha=alpha, use_external_csv=True, seed=42,
        )
        kw = make_model_kw(vs, nc)
        srv = FedServer(MoETextClassifier(**kw), device=DEVICE)

        for rnd in range(ROUNDS):
            states = []
            for ci, cds in enumerate(clients):
                cm = MoETextClassifier(**kw)
                cm.load_state_dict(srv.get_global_state(), strict=False)
                fs, _, n, _, _, _, _ = local_train(
                    cm, cds, epochs=1, batch_size=64, lr=2e-3,
                    device=DEVICE, top_k_sparse=2, fedprox_mu=mu,
                )
                states.append((fs, n))
            srv.aggregate(states)

        acc = evaluate(srv.global_model, test_ds)
        label = "FedAvg" if mu == 0.0 else f"FedProx mu={mu}"
        results.append({"mu": mu, "label": label, "alpha": alpha, "accuracy": acc})
        print(f"  {label:>20s}  alpha={alpha}  acc={acc:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [r["label"] for r in results]
    accs   = [r["accuracy"] for r in results]
    colors = ["#4C72B0" if r["mu"] == 0.0 else "#DD8452" for r in results]
    ax.bar(labels, accs, color=colors, alpha=0.85)
    ax.axhline(y=accs[0], color="#4C72B0", linestyle="--", alpha=0.4,
               label=f"FedAvg baseline = {accs[0]:.3f}")
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title(f"FedProx vs FedAvg at non-IID Dirichlet alpha={alpha}",
                 fontsize=13)
    ax.set_ylim(0, max(accs) * 1.15)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "exp7_fedprox.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: plots/exp7_fedprox.png")
    return results


# ============================================================
# EXPERIMENT 8: Ledger throughput + MiMC vs SHA-256
# ============================================================
def experiment_chain_overhead():
    """
    Two micro-benchmarks for the on-chain layer:
      (A) Pure-Python ledger throughput: how many transactions per second
          the Ledger can ingest, seal, and verify.
      (B) ZK-friendly hash overhead: SHA-256 vs MiMC for SEPG-sized state.
    """
    print("\n" + "=" * 60)
    print("  Experiment 8: Ledger throughput + MiMC vs SHA-256")
    print("=" * 60)

    from src.chain import Ledger
    from src.fl.zkhash import zkhash_benchmark

    # (A) Ledger throughput
    n_tx_values = [10, 100, 500, 1000]
    ledger_results = []
    for n_tx in n_tx_values:
        ledger = Ledger()
        t0 = time.perf_counter()
        for i in range(n_tx):
            ledger.add_transaction(
                "verify", client_id=i % 10, round=1, accepted=True,
                proof_hash=f"{i:032x}",
            )
        ledger.seal_block()
        seal_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        ok, _ = ledger.verify()
        verify_ms = (time.perf_counter() - t0) * 1000.0

        ledger_results.append({
            "n_tx": n_tx, "seal_ms": seal_ms, "verify_ms": verify_ms,
            "tx_per_sec_seal": n_tx / max(seal_ms, 1e-6) * 1000.0,
            "ok": ok,
        })
        print(f"  Ledger n={n_tx:>4d}  seal={seal_ms:>7.2f}ms  "
              f"verify={verify_ms:>7.2f}ms  ok={ok}")

    # (B) MiMC vs SHA-256 for an SEPG-sized state (~600K params)
    print()
    state_sizes = [
        ("tiny (1K)",    {"w": torch.randn(32, 32)}),
        ("small (10K)",  {"w": torch.randn(100, 100)}),
        ("medium (100K)",{"w": torch.randn(316, 316)}),
    ]
    hash_results = []
    for label, st in state_sizes:
        bench = zkhash_benchmark(st, n_trials=3)
        hash_results.append({
            "size_label": label,
            "sha256_ms": bench["sha256_ms"],
            "mimc_ms":   bench["mimc_ms"],
            "ratio":     bench["ratio"],
        })
        print(f"  Hash {label:>15s}  SHA={bench['sha256_ms']:>7.2f}ms  "
              f"MiMC={bench['mimc_ms']:>9.2f}ms  ratio={bench['ratio']:.0f}x")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # (A) Ledger throughput
    ns = [r["n_tx"] for r in ledger_results]
    seals = [r["seal_ms"] for r in ledger_results]
    vers  = [r["verify_ms"] for r in ledger_results]
    ax1.plot(ns, seals, "o-", color="#4C72B0", linewidth=2, label="Seal")
    ax1.plot(ns, vers,  "s-", color="#DD8452", linewidth=2, label="Verify")
    ax1.set_xlabel("Transactions per block", fontsize=12)
    ax1.set_ylabel("Time (ms)", fontsize=12)
    ax1.set_title("Pure-Python Ledger Throughput", fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # (B) Hash comparison
    sizes = [r["size_label"] for r in hash_results]
    sha = [r["sha256_ms"] for r in hash_results]
    mimc = [r["mimc_ms"] for r in hash_results]
    x = np.arange(len(sizes))
    ax2.bar(x - 0.2, sha,  width=0.4, color="#55A868", label="SHA-256 (current)")
    ax2.bar(x + 0.2, mimc, width=0.4, color="#8172B2", label="MiMC (ZK-friendly)")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(sizes)
    ax2.set_ylabel("Time per hash (ms)", fontsize=12)
    ax2.set_title("ZK-friendly Hash Overhead", fontsize=13)
    ax2.legend(fontsize=11)
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "exp8_chain_overhead.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: plots/exp8_chain_overhead.png")

    return {"ledger": ledger_results, "hash_overhead": hash_results}


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("\n  zkFedMoE Experiment Suite")
    print("  ========================\n")

    r1 = experiment_privacy_utility()
    r2 = experiment_comm_vs_k()
    r3 = experiment_verification_overhead()
    r4 = experiment_robustness()
    r5 = experiment_noniid()
    r6 = experiment_mia()
    r7 = experiment_fedprox()
    r8 = experiment_chain_overhead()

    # Save all results
    all_results = {
        "privacy_utility": r1,
        "comm_vs_k": r2,
        "verification_overhead": r3,
        "robustness": r4,
        "noniid_dirichlet": r5,
        "mia_dp_on_off": r6,
        "fedprox_vs_fedavg": r7,
        "chain_overhead": r8,
    }
    with open(PLOT_DIR / "experiment_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("  ALL EXPERIMENTS COMPLETE")
    print("=" * 60)
    print("\n  Generated plots:")
    for p in sorted(PLOT_DIR.glob("exp*.png")):
        print(f"    {p.name}")
    print(f"\n  Results: plots/experiment_results.json")
