"""
zkFedMoE Verification Harness v3 -- New positive results

Two experiments:
  exp7_v3 = FedProx with NORMALISED proximal term (param-count-normalised),
           which is required for the 600K-param MoE. We expect this to
           recover or beat FedAvg under non-IID Dirichlet alpha=0.3.

  exp8b_v1 = Disease single-symptom rescue. Train two otherwise identical
           Disease-Detection models, one with sparse_subset_prob=0.0 (no
           rescue) and one with sparse_subset_prob=0.35 (rescue). Then
           query each with three single-symptom inputs and record the
           top-1 disease for each.

Saves to verify/results/measured_v3.json. Does NOT touch v1 or v2.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
CODE_DIR = PROJECT_ROOT / "Code"
sys.path.insert(0, str(CODE_DIR))

from src.models.moe_model import MoETextClassifier
from src.data.text_datasets import build_ag_news_clients_noniid
from src.fl.client import local_train
from src.fl.server import FedServer
from src.fl.dp import apply_dp

RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "measured_v3.json"
DEVICE = torch.device("cpu")


def set_seed(s: int = 42) -> None:
    torch.manual_seed(s)
    np.random.seed(s)


def evaluate(model, dataset, bs=64) -> float:
    loader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=0)
    model.to(DEVICE).eval()
    c = t = 0
    with torch.no_grad():
        for ids, lbl in loader:
            ids, lbl = ids.to(DEVICE), lbl.to(DEVICE)
            c += (model(ids).argmax(-1) == lbl).sum().item()
            t += lbl.size(0)
    return c / max(t, 1)


def _save(d: Dict[str, Any]) -> None:
    def fix(o):
        if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))):
            return str(o)
        if isinstance(o, dict): return {k: fix(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [fix(x) for x in o]
        if isinstance(o, np.generic): return o.item()
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
    print(f"  [CHECKPOINT v3] Saved {name}")


def _hr(t): print(f"\n{'='*78}\n  {t}\n{'='*78}")


def make_model_kw(vs, nc, num_experts=8, k=2):
    return dict(vocab_size=vs, embed_dim=64, num_classes=nc,
                num_experts=num_experts, expert_hidden_dim=256, k=k, lora_r=8)


# ===========================================================================
# EXP 7 v3 -- FedProx with NORMALISED proximal term
# ===========================================================================
def exp7_v3() -> Dict[str, Any]:
    _hr("EXPERIMENT 7 v3: FedProx (NORMALISED prox, alpha=0.3, 8 rounds)")
    print("  Tests whether parameter-count-normalised FedProx recovers")
    print("  the positive result on a 600K-param MoE.")

    ALPHA = 0.3
    ROUNDS = 8
    # When normalised, mu can be much larger because prox is now O(1) instead
    # of O(N_params). We sweep a wide range to find the new sweet spot.
    mu_values = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
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
                    device=DEVICE, top_k_sparse=2,
                    fedprox_mu=mu, fedprox_normalise=(mu > 0),
                )
                states.append((fs, n))
            srv.aggregate(states)
        wall = time.perf_counter() - t0
        acc = evaluate(srv.global_model, test_ds)
        label = "FedAvg" if mu == 0.0 else f"FedProx-norm mu={mu}"
        rows.append({
            "mu": mu, "label": label, "alpha": ALPHA,
            "accuracy_pct": round(acc * 100, 2),
            "wall_seconds": round(wall, 1),
            "normalised": (mu > 0),
        })
        print(f"  {label:25s}  alpha={ALPHA}  acc={acc*100:5.2f}%  wall={wall:.0f}s")

    payload = {"rows": rows, "alpha": ALPHA, "rounds": ROUNDS,
               "note": "FedProx with proximal term divided by N_params"}
    _save_partial("exp7_v3", payload)
    return payload


# ===========================================================================
# EXP 8B v1 -- Single-symptom rescue table
# ===========================================================================
def exp8b_v1() -> Dict[str, Any]:
    _hr("EXPERIMENT 8B: Single-symptom rescue (sparse augmentation on/off)")

    sys.path.insert(0, str(CODE_DIR))
    from data.disease_symptoms import build_dataset, DISEASES, SYMPTOMS

    sym2ix = {s: i for i, s in enumerate(SYMPTOMS)}
    ROUNDS = 10
    LOCAL_EPOCHS = 3
    LR = 2e-3
    BATCH = 32
    N_HOSP = 10
    ALPHA = 1.0

    test_symptoms = ["cough", "itching", "headache"]

    class _Clf(torch.nn.Module):
        def __init__(self, n_in, n_out):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(n_in, 128), torch.nn.ReLU(),
                torch.nn.Dropout(0.15),
                torch.nn.Linear(128, 128), torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(128, n_out),
            )
        def forward(self, x): return self.net(x)

    def train_disease_model(sparse_subset_prob: float):
        set_seed(42)
        X, y, syms, dis_names = build_dataset(
            n_variants=80, seed=42,
            sparse_subset_prob=sparse_subset_prob,
        )
        n_features = X.shape[1]
        n_classes = len(dis_names)
        n_total = X.shape[0]
        n_test = max(int(0.2 * n_total), n_classes)

        perm = np.random.permutation(n_total)
        train_ix = perm[n_test:]
        test_ix = perm[:n_test]

        X_tr = torch.from_numpy(X[train_ix]).float()
        y_tr = torch.from_numpy(y[train_ix]).long()
        X_te = torch.from_numpy(X[test_ix]).float()
        y_te = torch.from_numpy(y[test_ix]).long()
        train_ds = TensorDataset(X_tr, y_tr)
        test_ds = TensorDataset(X_te, y_te)

        # Dirichlet split
        rng = np.random.default_rng(42)
        by_class = [np.where(y_tr.numpy() == c)[0] for c in range(n_classes)]
        for arr_c in by_class: rng.shuffle(arr_c)
        cand = None
        for _ in range(50):
            c2 = [[] for _ in range(N_HOSP)]
            for c in range(n_classes):
                if len(by_class[c]) == 0: continue
                props = rng.dirichlet([ALPHA] * N_HOSP)
                pts = (np.cumsum(props) * len(by_class[c])).astype(int)[:-1]
                for cid, chunk in enumerate(np.split(by_class[c], pts)):
                    c2[cid].extend(chunk.tolist())
            if min(len(ix) for ix in c2) >= 3:
                cand = c2; break
        if cand is None: cand = c2
        client_dss = [Subset(train_ds, ix) for ix in cand]

        global_model = _Clf(n_features, n_classes)

        for rnd in range(ROUNDS):
            global_state = {k: v.detach().cpu().clone()
                            for k, v in global_model.state_dict().items()}
            updates = []
            for cid in range(N_HOSP):
                local = _Clf(n_features, n_classes)
                local.load_state_dict(global_state)
                local.train()
                loader = DataLoader(client_dss[cid], batch_size=BATCH, shuffle=True)
                opt = torch.optim.Adam(local.parameters(), lr=LR)
                crit = torch.nn.CrossEntropyLoss()
                for _ep in range(LOCAL_EPOCHS):
                    for Xb, yb in loader:
                        opt.zero_grad()
                        crit(local(Xb), yb).backward()
                        opt.step()
                updates.append(({k: v.detach().cpu().clone()
                                 for k, v in local.state_dict().items()},
                                len(cand[cid])))
            # FedAvg
            keys = list(updates[0][0].keys())
            tot = sum(n for _, n in updates)
            agg = {k: torch.zeros_like(updates[0][0][k]).float() for k in keys}
            for st_, n in updates:
                w = n / tot
                for k in keys: agg[k] += st_[k].float() * w
            global_model.load_state_dict(agg)
        return global_model, syms, dis_names, n_features

    def predict_top1(model, n_features, sym_name, syms, dis_names):
        if sym_name not in sym2ix:
            return ("(unknown symptom)", 0.0)
        vec = np.zeros(n_features, dtype=np.float32)
        vec[sym2ix[sym_name]] = 1.0
        x = torch.from_numpy(vec).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=-1).squeeze(0).numpy()
        idx = int(np.argmax(probs))
        return (dis_names[idx], float(probs[idx]))

    # No rescue
    print("  Training without sparse augmentation (sparse_subset_prob=0.0)...")
    t0 = time.perf_counter()
    model_no, syms, dis_names, nf = train_disease_model(sparse_subset_prob=0.0)
    print(f"  Done in {time.perf_counter()-t0:.0f}s")
    no_results = {sym: predict_top1(model_no, nf, sym, syms, dis_names)
                  for sym in test_symptoms}

    # With rescue
    print("  Training WITH sparse augmentation (sparse_subset_prob=0.35)...")
    t0 = time.perf_counter()
    model_yes, _, _, _ = train_disease_model(sparse_subset_prob=0.35)
    print(f"  Done in {time.perf_counter()-t0:.0f}s")
    yes_results = {sym: predict_top1(model_yes, nf, sym, syms, dis_names)
                   for sym in test_symptoms}

    # Print and save
    print()
    print(f"  {'Symptom':<14} | {'Without aug.':<32} | {'With aug.':<32}")
    print(f"  {'-'*14} | {'-'*32} | {'-'*32}")
    rescue_rows = []
    for sym in test_symptoms:
        nname, nprob = no_results[sym]
        yname, yprob = yes_results[sym]
        rescue_rows.append({
            "symptom":      sym,
            "without_aug":  {"top1": nname, "prob": round(nprob * 100, 2)},
            "with_aug":     {"top1": yname, "prob": round(yprob * 100, 2)},
        })
        print(f"  {sym:<14} | {nname[:25]:<25}{nprob*100:>5.1f}%  | "
              f"{yname[:25]:<25}{yprob*100:>5.1f}%")

    payload = {"rows": rescue_rows, "rounds": ROUNDS, "n_hospitals": N_HOSP,
               "test_symptoms": test_symptoms,
               "note": ("sparse_subset_prob controls fraction of training rows "
                        "that are 1-3-symptom presentations; the rescue is the "
                        "claim that adding such rows fixes the OOD failure on "
                        "single-symptom queries.")}
    _save_partial("exp8b_v1", payload)
    return payload


ALL_V3 = [
    ("exp7_v3", exp7_v3),
    ("exp8b_v1", exp8b_v1),
]


def main():
    args = sys.argv[1:]
    targets = [n for n, _ in ALL_V3] if not args or args == ["all"] else args

    overall_t0 = time.perf_counter()
    for name, fn in ALL_V3:
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
    print(f"  All v3 experiments complete in {overall_wall:.1f}s "
          f"({overall_wall/60:.1f} min)")
    print(f"  Results: {RESULTS_FILE}\n{'='*78}")


if __name__ == "__main__":
    main()
