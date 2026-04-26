"""
Hospital Federated Learning Demo (zkFedMoE Edge-Device Simulation)
==================================================================

Yeh demo dikhata hai ki real hospital scenario me FL kaise kaam karta hai:

  1.  Ek public hospital patient dataset (Pima Indians Diabetes from UCI)
      load karte hain. Yeh "global data" hai.

  2.  Iss data ko N alag-alag hospitals (edge devices) me TODA jaata hai.
      Har hospital ke paas SIRF apna shard hai - dusre hospitals ka data
      kabhi nahi dekh sakta. (Realistic privacy constraint.)

  3.  Har round me:
        * Server bhejega current global model -> sab hospitals
        * Har hospital APNE LOCAL data pe model train karega
        * Hospital wapas SIRF MODEL UPDATE bhejega (raw data nahi!)
        * Server saare updates ka FedAvg karke naya global model banayega
        * Sabko broadcast karega
        * Repeat

  4.  Console pe har step dikhayenge taaki traceable rahe ki:
        * data har hospital ke paas alag hai
        * raw data NEVER leaves the hospital
        * sirf gradients/weights flow karte hain

Run karne ke liye:
    python hospital_fl_demo.py
    python hospital_fl_demo.py --clients 5 --rounds 8 --alpha 0.3

Bina internet bhi chalega -- offline fallback dataset built in.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset

# ============================================================================
# CONFIG / DATASET
# ============================================================================

# Pima Indians Diabetes (UCI ML Repository / Kaggle, public domain)
# 768 patients, 8 features, binary outcome (1 = diabetes positive)
PIMA_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "pima-indians-diabetes.data.csv"
)

PIMA_FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]

# Tiny offline fallback: 32 hand-crafted plausible rows so demo always runs.
OFFLINE_FALLBACK = """
6,148,72,35,0,33.6,0.627,50,1
1,85,66,29,0,26.6,0.351,31,0
8,183,64,0,0,23.3,0.672,32,1
1,89,66,23,94,28.1,0.167,21,0
0,137,40,35,168,43.1,2.288,33,1
5,116,74,0,0,25.6,0.201,30,0
3,78,50,32,88,31.0,0.248,26,1
10,115,0,0,0,35.3,0.134,29,0
2,197,70,45,543,30.5,0.158,53,1
8,125,96,0,0,0.0,0.232,54,1
4,110,92,0,0,37.6,0.191,30,0
10,168,74,0,0,38.0,0.537,34,1
10,139,80,0,0,27.1,1.441,57,0
1,189,60,23,846,30.1,0.398,59,1
5,166,72,19,175,25.8,0.587,51,1
7,100,0,0,0,30.0,0.484,32,1
0,118,84,47,230,45.8,0.551,31,1
7,107,74,0,0,29.6,0.254,31,1
1,103,30,38,83,43.3,0.183,33,0
1,115,70,30,96,34.6,0.529,32,1
3,126,88,41,235,39.3,0.704,27,0
8,99,84,0,0,35.4,0.388,50,0
7,196,90,0,0,39.8,0.451,41,1
9,119,80,35,0,29.0,0.263,29,1
11,143,94,33,146,36.6,0.254,51,1
10,125,70,26,115,31.1,0.205,41,1
7,147,76,0,0,39.4,0.257,43,1
1,97,66,15,140,23.2,0.487,22,0
13,145,82,19,110,22.2,0.245,57,0
5,117,92,0,0,34.1,0.337,38,0
5,109,75,26,0,36.0,0.546,60,0
3,158,76,36,245,31.6,0.851,28,1
"""


def download_pima(cache_path: Path) -> str:
    """Try downloading; fall back to offline data."""
    if cache_path.exists():
        print(f"[data] Cached file found at {cache_path}")
        return cache_path.read_text()
    try:
        print(f"[data] Downloading Pima Diabetes from {PIMA_URL} ...")
        with urllib.request.urlopen(PIMA_URL, timeout=10) as r:
            text = r.read().decode("utf-8")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text)
        print(f"[data] Downloaded {len(text.splitlines())} lines, cached to {cache_path}")
        return text
    except Exception as exc:
        print(f"[data] Download failed ({exc}); using offline fallback (32 rows).")
        return OFFLINE_FALLBACK.strip()


def parse_csv(text: str) -> Tuple[np.ndarray, np.ndarray]:
    rows = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Skip header if present
        parts = line.split(",")
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            continue
        rows.append(vals)
    arr = np.array(rows, dtype=np.float32)
    X, y = arr[:, :8], arr[:, 8].astype(np.int64)
    return X, y


def standardize(X: np.ndarray) -> np.ndarray:
    """Standard z-score normalisation -- important for medical features
    that have wildly different scales (Glucose ~100, DiabetesPedigree ~0.5)."""
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-8
    return (X - mu) / sd


# ============================================================================
# CLIENT PARTITIONING
# ============================================================================
def dirichlet_split_indices(
    labels: np.ndarray, num_clients: int, alpha: float, seed: int = 42,
    min_size: int = 5,
) -> List[List[int]]:
    """
    Hospitals me data alag-alag hota hai (kuch hospitals zyada diabetic
    patients dekhte hain, kuch kam). Iss reality ko simulate karne ke liye
    Dirichlet(alpha) partitioning use karte hain.

    alpha small  -> very heterogeneous (each hospital skewed)
    alpha large  -> nearly IID
    """
    rng = np.random.default_rng(seed)
    n_classes = int(labels.max()) + 1
    by_class = [np.where(labels == c)[0] for c in range(n_classes)]
    for arr in by_class:
        rng.shuffle(arr)

    for _attempt in range(30):
        client_indices: List[List[int]] = [[] for _ in range(num_clients)]
        for c in range(n_classes):
            proportions = rng.dirichlet([alpha] * num_clients)
            split_pts = (np.cumsum(proportions) * len(by_class[c])).astype(int)[:-1]
            chunks = np.split(by_class[c], split_pts)
            for cid, chunk in enumerate(chunks):
                client_indices[cid].extend(chunk.tolist())
        if min(len(ix) for ix in client_indices) >= min_size:
            break

    for ix in client_indices:
        rng.shuffle(ix)
    return client_indices


# ============================================================================
# MODEL
# ============================================================================
class DiabetesClassifier(nn.Module):
    """A small MLP suitable for tabular medical features.
    Same architecture for every client and the global model."""

    def __init__(self, in_features: int = 8, hidden: int = 32, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def get_state(model: nn.Module):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def state_l2_diff(s1, s2) -> float:
    """L2 norm of the difference between two state dicts (a measure of how
    much a client's local update changed the model)."""
    sq = 0.0
    for k in s1:
        sq += float((s1[k].float() - s2[k].float()).pow(2).sum().item())
    return sq ** 0.5


# ============================================================================
# LOCAL TRAINING (one hospital, one round)
# ============================================================================
def local_train(
    model: nn.Module, dataset: TensorDataset, epochs: int, lr: float,
    batch_size: int,
) -> Tuple[dict, float, int]:
    """
    Train one round of local SGD on a hospital's private data.
    Returns (state_dict, final_loss, n_samples).
    Raw data NEVER leaves this function.
    """
    model.train()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    n_samples = 0
    last_loss = 0.0
    for _ep in range(epochs):
        for X, y in loader:
            opt.zero_grad()
            logits = model(X)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            last_loss = loss.item()
            n_samples += y.size(0)

    return get_state(model), last_loss, n_samples


# ============================================================================
# SERVER AGGREGATION (FedAvg)
# ============================================================================
def fedavg(states_with_weights):
    """Sample-count-weighted average of client states."""
    total = sum(n for _, n in states_with_weights)
    keys = list(states_with_weights[0][0].keys())
    agg = {k: torch.zeros_like(states_with_weights[0][0][k]).float() for k in keys}
    for state, n in states_with_weights:
        w = n / total
        for k in keys:
            agg[k] += state[k].float() * w
    return agg


# ============================================================================
# EVALUATION
# ============================================================================
def evaluate(model: nn.Module, dataset: TensorDataset) -> Tuple[float, float]:
    """Return (accuracy, average loss)."""
    model.eval()
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    crit = nn.CrossEntropyLoss(reduction="sum")
    correct = total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for X, y in loader:
            out = model(X)
            loss_sum += crit(out, y).item()
            preds = out.argmax(dim=-1)
            correct += int((preds == y).sum())
            total += y.size(0)
    return correct / max(total, 1), loss_sum / max(total, 1)


# ============================================================================
# CLIENT CLASS DISTRIBUTION (for the heterogeneity report)
# ============================================================================
def class_distribution(y: np.ndarray, n_classes: int = 2) -> List[int]:
    return [int((y == c).sum()) for c in range(n_classes)]


# ============================================================================
# MAIN DEMO
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clients", type=int, default=4,
                        help="Number of edge hospitals (default: 4)")
    parser.add_argument("--rounds", type=int, default=10,
                        help="Federated rounds (default: 10)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Dirichlet alpha for non-IID split (default: 0.5)")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Local epochs per round (default: 2)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--batch", type=int, default=16,
                        help="Local batch size (default: 16)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-download", action="store_true",
                        help="Skip download attempt; use offline fallback only")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 72)
    print("  HOSPITAL FEDERATED LEARNING DEMO  (zkFedMoE edge-device simulation)")
    print("=" * 72)

    # ---- 1.  Load global hospital dataset --------------------------
    cache_path = Path(__file__).resolve().parent / "data" / "pima_diabetes.csv"
    if args.no_download:
        print("[data] --no-download set, using offline fallback (32 rows).")
        text = OFFLINE_FALLBACK.strip()
    else:
        text = download_pima(cache_path)
    X, y = parse_csv(text)
    X = standardize(X)
    n_total = X.shape[0]
    n_test = max(int(0.2 * n_total), 8)
    perm = np.random.permutation(n_total)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    X_train = torch.from_numpy(X[train_idx]).float()
    y_train = torch.from_numpy(y[train_idx]).long()
    X_test = torch.from_numpy(X[test_idx]).float()
    y_test = torch.from_numpy(y[test_idx]).long()
    test_ds = TensorDataset(X_test, y_test)

    print(f"\n[data] Dataset loaded: {n_total} rows total")
    print(f"        train: {len(train_idx)}, test: {len(test_idx)}")
    print(f"        features: {PIMA_FEATURES}")
    print(f"        class balance (train): {class_distribution(y[train_idx])}")

    # ---- 2.  Partition train data across N hospitals (non-IID) -----
    print(f"\n[partition] Splitting {len(train_idx)} train rows across "
          f"{args.clients} hospitals (Dirichlet alpha={args.alpha})")
    shard_indices = dirichlet_split_indices(
        labels=y[train_idx], num_clients=args.clients, alpha=args.alpha,
        seed=args.seed, min_size=5,
    )

    client_datasets = []
    for cid, ix in enumerate(shard_indices):
        Xc = X_train[ix]
        yc = y_train[ix]
        client_datasets.append(TensorDataset(Xc, yc))
        dist = class_distribution(yc.numpy())
        bar = "*" * dist[0] + "#" * dist[1]
        print(f"    Hospital {cid}: n={len(ix):3d}  classes "
              f"[non-diabetic={dist[0]:2d}, diabetic={dist[1]:2d}]  {bar}")

    print("\n[partition]  Notice: each hospital has a different mix of "
          "diabetic vs non-diabetic patients.\n"
          "             This simulates real-world heterogeneity. No hospital\n"
          "             ever sees another hospital's records.")

    # ---- 3.  Initialise global model ------------------------------
    global_model = DiabetesClassifier(in_features=8, num_classes=2)
    print(f"\n[server] Global model initialised: "
          f"{sum(p.numel() for p in global_model.parameters())} parameters")

    initial_acc, initial_loss = evaluate(global_model, test_ds)
    print(f"[server] Initial (untrained) test accuracy: {initial_acc:.2%}, "
          f"loss: {initial_loss:.4f}\n")

    # ---- 4.  Federated training loop ------------------------------
    history = []
    for rnd in range(1, args.rounds + 1):
        round_start = time.perf_counter()
        print("-" * 72)
        print(f"ROUND {rnd}/{args.rounds}")
        print("-" * 72)
        print(f"[server -> all hospitals] Broadcasting global model "
              f"(theta_t)")

        global_state = get_state(global_model)
        client_updates = []  # list of (state, n)
        client_losses = []

        for cid in range(args.clients):
            # Hospital loads global model and trains LOCALLY on its OWN data
            local_model = DiabetesClassifier(in_features=8, num_classes=2)
            local_model.load_state_dict(global_state)

            new_state, last_loss, n_samples = local_train(
                local_model, client_datasets[cid],
                epochs=args.epochs, lr=args.lr, batch_size=args.batch,
            )

            # Measure how much this client's update changed the model
            delta = state_l2_diff(global_state, new_state)
            print(f"  Hospital {cid}: trained on {n_samples} private records "
                  f"-> uploads update (||delta||_2 = {delta:.4f}, "
                  f"local_loss = {last_loss:.4f})")
            client_updates.append((new_state, n_samples))
            client_losses.append(last_loss)

        # Server aggregates -- FedAvg
        print(f"[server] Received {len(client_updates)} updates "
              f"(NO raw patient data received).")
        print(f"[server] Aggregating via FedAvg (sample-weighted mean)...")
        new_global_state = fedavg(client_updates)
        global_model.load_state_dict(new_global_state)

        # Evaluate new global model on held-out test set
        acc, test_loss = evaluate(global_model, test_ds)
        round_time = time.perf_counter() - round_start
        history.append({
            "round": rnd, "test_acc": acc, "test_loss": test_loss,
            "client_avg_loss": float(np.mean(client_losses)),
            "round_time_s": round_time,
        })

        print(f"[server] Updated global model.")
        print(f"         Test accuracy : {acc:.2%}")
        print(f"         Test loss     : {test_loss:.4f}")
        print(f"         Round time    : {round_time:.2f}s")
        print()

    # ---- 5.  Final summary ----------------------------------------
    print("=" * 72)
    print("  FINAL SUMMARY")
    print("=" * 72)
    print(f"  Initial (untrained) accuracy : {initial_acc:.2%}")
    print(f"  Final accuracy after FL      : {history[-1]['test_acc']:.2%}")
    print(f"  Improvement                  : {(history[-1]['test_acc'] - initial_acc) * 100:+.2f} pp")
    print(f"  Total rounds                 : {args.rounds}")
    print(f"  Total hospitals (clients)    : {args.clients}")
    print(f"  Non-IID alpha                : {args.alpha}")
    print()
    print("  Per-round trace:")
    print(f"  {'Round':>5} {'Test Acc':>10} {'Test Loss':>10} {'Local Loss':>10} {'Time(s)':>8}")
    for h in history:
        print(f"  {h['round']:>5d} {h['test_acc']:>10.2%} {h['test_loss']:>10.4f} "
              f"{h['client_avg_loss']:>10.4f} {h['round_time_s']:>8.2f}")

    print()
    print("  Important takeaway:")
    print("    *  Global accuracy improved without any single hospital")
    print("       sharing its raw patient records.")
    print("    *  Each hospital trained ONLY on its own slice of the data.")
    print("    *  Server saw model updates (deltas), not patient identities.")
    print("    *  This is exactly the privacy contract real federated learning")
    print("       provides; zkFedMoE adds DP, SEPG, and secure aggregation on top.")
    print()
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[abort] Interrupted by user.")
        sys.exit(1)
