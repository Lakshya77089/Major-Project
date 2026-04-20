"""
zkFedMoE Demo Script
====================
Run this to demonstrate the full system to your teacher.

It runs:
  1. Phase 1 — Baseline FL on small corpus (fast, ~30 sec)
  2. Phase 2 — AG News dense vs sparse comparison (~3 min)
  3. Generates plots saved to plots/ directory

Usage:
    python run_demo.py
"""

import os
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models import MoETextClassifier
from src.fl import FedServer, local_train
from src.data import build_ag_news_clients


# ---- Helpers ----

def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model: nn.Module, dataset: Dataset, batch_size: int, device: torch.device) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = model.to(device)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for input_ids, labels in loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            preds = model(input_ids).argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / max(total, 1)


PLOT_DIR = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)


# ==================================================================
# PHASE 1
# ==================================================================

def run_phase1():
    print("=" * 65)
    print("  PHASE 1: Baseline FL with MoE + LoRA (small corpus)")
    print("=" * 65)
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clients, test_ds, vocab_size, num_classes, _vocab = build_ag_news_clients(
        num_clients=4, seq_len=64, use_external_csv=False, repeat=50,
    )
    train_ds = ConcatDataset(clients)
    print(f"  {len(train_ds)} train, {len(test_ds)} test, "
          f"vocab={vocab_size}, classes={num_classes}, clients={len(clients)}")
    print("-" * 65)

    model_kw = dict(vocab_size=vocab_size, embed_dim=64, num_classes=num_classes,
                    num_experts=4, expert_hidden_dim=128, k=2, lora_r=8)
    server = FedServer(MoETextClassifier(**model_kw), device=device)

    rounds, train_accs, test_accs = [], [], []

    for rnd in range(1, 16):
        states = []
        for cds in clients:
            cm = MoETextClassifier(**model_kw)
            cm.load_state_dict(server.get_global_state(), strict=False)
            fs, _, n, _, _, _, _ = local_train(cm, cds, epochs=2, batch_size=16,
                                              lr=5e-4, device=device)
            states.append((fs, n))
        server.aggregate(states)

        tr = evaluate(server.global_model, train_ds, 32, device)
        te = evaluate(server.global_model, test_ds, 32, device)
        rounds.append(rnd)
        train_accs.append(tr)
        test_accs.append(te)
        print(f"  Round {rnd:2d}:  train={tr:.4f}  test={te:.4f}")

    print("-" * 65)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rounds, train_accs, "o-", label="Train Accuracy", linewidth=2)
    ax.plot(rounds, test_accs, "s--", label="Test Accuracy", linewidth=2)
    ax.axhline(y=0.25, color="red", linestyle=":", label="Random Baseline (25%)")
    ax.set_xlabel("Federated Round", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Phase 1: Federated Learning Convergence (MoE + LoRA)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "phase1_accuracy.png", dpi=150)
    plt.close(fig)
    print(f"  Plot saved: plots/phase1_accuracy.png")

    return {"rounds": rounds, "train_acc": train_accs, "test_acc": test_accs}


# ==================================================================
# PHASE 2
# ==================================================================

def run_phase2():
    print()
    print("=" * 65)
    print("  PHASE 2: AG News — Dense vs Sparse Communication")
    print("=" * 65)
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NUM_EXPERTS = 8
    TOP_K = 2
    ROUNDS = 8

    clients, test_ds, vocab_size, num_classes, _vocab = build_ag_news_clients(
        num_clients=5, seq_len=64, use_external_csv=True, max_vocab=5000,
    )
    train_ds = ConcatDataset(clients)
    print(f"  {len(train_ds):,} train, {len(test_ds):,} test, "
          f"vocab={vocab_size}, classes={num_classes}")

    model_kw = dict(vocab_size=vocab_size, embed_dim=64, num_classes=num_classes,
                    num_experts=NUM_EXPERTS, expert_hidden_dim=256,
                    k=TOP_K, lora_r=8)

    m = MoETextClassifier(**model_kw)
    total_p = sum(p.numel() for p in m.parameters())
    expert_p = sum(p.numel() for n, p in m.named_parameters() if "moe.experts" in n)
    print(f"  Params: total={total_p:,}  experts={expert_p:,} ({expert_p/total_p*100:.0f}%)")

    # Both servers start identical
    dense_model = MoETextClassifier(**model_kw)
    sparse_model = MoETextClassifier(**model_kw)
    sparse_model.load_state_dict(dense_model.state_dict())
    srv_dense = FedServer(dense_model, device=device)
    srv_sparse = FedServer(sparse_model, device=device)

    print("-" * 65)
    print(f"  {'Rnd':>3}  {'Dense':>8} {'Sparse':>8}  "
          f"{'Dense KB':>10} {'Sparse KB':>10} {'Saving':>7}")
    print("-" * 65)

    rnd_list, dense_accs, sparse_accs = [], [], []
    dense_kb_list, sparse_kb_list = [], []

    for rnd in range(1, ROUNDS + 1):
        d_states, s_states = [], []
        rd_bytes = rs_bytes = 0

        for cds in clients:
            cm = MoETextClassifier(**model_kw)
            cm.load_state_dict(srv_dense.get_global_state(), strict=False)
            fs, ss, n, db, sb, _, _ = local_train(
                cm, cds, epochs=1, batch_size=64, lr=2e-3,
                device=device, top_k_sparse=TOP_K)
            rd_bytes += db
            rs_bytes += sb
            d_states.append((fs, n))
            s_states.append((ss, n))

        srv_dense.aggregate(d_states)
        srv_sparse.aggregate(s_states)

        da = evaluate(srv_dense.global_model, test_ds, 64, device)
        sa = evaluate(srv_sparse.global_model, test_ds, 64, device)
        dk = rd_bytes / 1024
        sk = rs_bytes / 1024
        sav = (1 - sk / dk) * 100 if dk > 0 else 0

        rnd_list.append(rnd)
        dense_accs.append(da)
        sparse_accs.append(sa)
        dense_kb_list.append(dk)
        sparse_kb_list.append(sk)

        print(f"  {rnd:3d}  {da:8.4f} {sa:8.4f}  "
              f"{dk:10.1f} {sk:10.1f} {sav:6.1f}%")

    print("-" * 65)
    saving_pct = (1 - sparse_kb_list[-1] / dense_kb_list[-1]) * 100
    print(f"  Communication saving: {saving_pct:.1f}% "
          f"(Top-{TOP_K} of {NUM_EXPERTS} experts)")

    # ---- Plot 1: Accuracy comparison ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rnd_list, dense_accs, "o-", label="Dense (all experts)", linewidth=2)
    ax.plot(rnd_list, sparse_accs, "s--", label=f"Sparse (Top-{TOP_K}/{NUM_EXPERTS})",
            linewidth=2)
    ax.axhline(y=0.25, color="red", linestyle=":", label="Random Baseline (25%)")
    ax.set_xlabel("Federated Round", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("Phase 2: Dense vs Sparse FL on AG News (120K samples)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "phase2_accuracy.png", dpi=150)
    plt.close(fig)
    print(f"  Plot saved: plots/phase2_accuracy.png")

    # ---- Plot 2: Communication cost bar chart ----
    fig, ax = plt.subplots(figsize=(7, 5))
    total_dense = sum(dense_kb_list) / 1024
    total_sparse = sum(sparse_kb_list) / 1024
    bars = ax.bar(["Dense\n(all experts)", f"Sparse\n(Top-{TOP_K}/{NUM_EXPERTS})"],
                  [total_dense, total_sparse],
                  color=["#4C72B0", "#55A868"], width=0.5)
    ax.bar_label(bars, fmt="%.1f MB", fontsize=12, padding=3)
    ax.set_ylabel("Total Communication (MB)", fontsize=12)
    ax.set_title(f"Communication Cost: {ROUNDS} Rounds x {len(clients)} Clients",
                 fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "phase2_communication.png", dpi=150)
    plt.close(fig)
    print(f"  Plot saved: plots/phase2_communication.png")

    # ---- Plot 3: Parameter breakdown pie chart ----
    embed_p = sum(p.numel() for n, p in m.named_parameters() if n.startswith("embedding"))
    other_p = total_p - embed_p - expert_p
    fig, ax = plt.subplots(figsize=(6, 6))
    sizes = [embed_p, expert_p, other_p]
    labels = [f"Embedding\n{embed_p:,} ({embed_p/total_p*100:.0f}%)",
              f"MoE Experts\n{expert_p:,} ({expert_p/total_p*100:.0f}%)",
              f"Router+LoRA\n{other_p:,} ({other_p/total_p*100:.0f}%)"]
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    ax.pie(sizes, labels=labels, colors=colors, startangle=90,
           textprops={"fontsize": 11})
    ax.set_title("Model Parameter Distribution", fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "model_params.png", dpi=150)
    plt.close(fig)
    print(f"  Plot saved: plots/model_params.png")

    return {
        "rounds": rnd_list, "dense_acc": dense_accs, "sparse_acc": sparse_accs,
        "dense_kb": dense_kb_list, "sparse_kb": sparse_kb_list,
        "saving_pct": saving_pct,
    }


# ==================================================================
# MAIN
# ==================================================================

if __name__ == "__main__":
    print()
    print("  zkFedMoE Demo — Federated MoE + LoRA Prototype")
    print("  =============================================")
    print()

    r1 = run_phase1()
    r2 = run_phase2()

    # Save results as JSON
    results = {"phase1": r1, "phase2": r2}
    with open(PLOT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print()
    print("=" * 65)
    print("  ALL DONE — Results and plots saved in plots/ directory")
    print("=" * 65)
    print()
    print("  Generated files:")
    for p in sorted(PLOT_DIR.iterdir()):
        print(f"    {p.name}")
    print()
