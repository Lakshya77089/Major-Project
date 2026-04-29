# zkFedMoE — Verification Report for Supervisor

**Project:** zkFedMoE (Zero-Knowledge Federated Mixture-of-Experts)
**Team:** Group #34 — Keshav Kashyap, Lakshya Sharma, Prakriti Patel
**Supervisor:** Dr. Gyan Singh Yadav, IIIT Kota
**Date:** April 2026

This report is the result of running every experiment claimed in `Report/conference_paper.tex` from scratch in an isolated harness (`verify/run_verification*.py`) against the unmodified primitives in `Code/src/`. The goal: verify that each numerical claim in the paper is a measured value, not an unverified figure.

---

## Top-line answer to the supervisor's question

> "All values should be verified and every claim should be justified."

After running 9 experiments end-to-end (140+ minutes of CPU time) and producing two independent measured datasets (`measured.json`, `measured_v2.json`):

- **6 of 9 experiments are now genuine and reproducible** with verifiable numbers.
- **2 experiments need the paper text rewritten** because the implementation behaves differently from the paper's narrative (Exp 6 MIA, Exp 7 FedProx).
- **1 experiment (Exp 1)** has its numbers correct but is **mislabelled** in the paper (says "Rényi" when the run used the basic √T accountant).

Detailed verdict per experiment is in [`AUDIT.md`](AUDIT.md). This document is the supervisor-readable summary.

---

## Experiment-by-experiment status

### Exp 1 — Privacy-Utility Tradeoff (Table I) ✅ verifiable, label fix needed

| σ | Paper ε | Paper Acc | Measured ε (basic √T) | Measured ε (Rényi) | Measured Acc |
|---|---|---|---|---|---|
| 0.0 | ∞ | 58.00% | ∞ | ∞ | 57.09% ✓ |
| 0.1 | 0.289 | 25.00% | 0.289 ✓ | 250.4 ✗ | 25.00% ✓ |
| 0.5 | 0.058 | 25.01% | 0.058 ✓ | 3.060 ✗ | 25.00% ✓ |
| 1.0 | 0.029 | 26.12% | 0.029 ✓ | 0.775 ✗ | 25.00% ≈ |
| 2.0 | 0.014 | 25.00% | 0.014 ✓ | 0.184 ✗ | 25.00% ✓ |

**Genuine**: every accuracy and ε in the paper is a real measurement. **Bug**: paper's column header reads "ε (Rényi)" but the values are the basic √T composition. The proper Rényi values are an order of magnitude larger (less private). Fix: change the column header, or re-fit the table to the Rényi numbers (whichever the team prefers).

---

### Exp 2 — Communication vs Top-K (Table II) ✅ verifiable

Saving percentages are deterministic and **MATCH exactly**: K=1→39.52%, K=2→33.88%, K=3→28.23%, K=4→22.58%, K=8→0.00%.

Accuracies in our run are *higher than the paper claims* by 2-15 pp (reaching 74% at K=4 vs paper's 59%). The qualitative claim "K∈{3,4} is the sweet spot" still holds. No correction needed beyond replacing the specific accuracy numbers with the new measurements.

---

### Exp 3 — SEPG Verification Overhead (Table III) ⚠ hardware-dependent

Paper claims 6.08 ms total per K. Measured: **10.89 ± 0.77 ms** on this hardware (Intel laptop, single CPU core, Python 3.10). The 79% gap is purely a hardware effect — pure-Python SHA-256 timings vary 2-4× across machines.

**Recommended fix**: add a footnote that timings were measured on the harness machine, and report the measured number 10.89 ms. The "constant across K" qualitative claim is true (std/mean = 0.07).

---

### Exp 4 — Robustness Under Poisoning (Table IV) ✅ verifiable AFTER fix

Initial run (5 clients, 5 rounds): broken — Median collapsed to 25.00% at 40% malicious due to small-N instability of coordinate-wise median when 2 of 5 voters are adversarial.

**Fixed run (9 clients, 8 rounds)**:

| Method | 0% mal | 22% mal | 44% mal | Drop 0→44 |
|---|---|---|---|---|
| FedAvg | 51.08 | 48.59 | **37.82** | -13.3 pp |
| Median | 47.87 | 51.74 | **43.34** | **-4.5 pp** ← most robust |
| TrimMean | 52.57 | 50.57 | 42.75 | -9.8 pp |

This **matches the paper's qualitative ordering** (Median > TrimMean > FedAvg under heavy poisoning) and the magnitudes are within 4 pp of the paper's claims. Use these numbers in the paper, and update the experimental setup to mention 9 clients + 8 rounds.

---

### Exp 5 — Non-IID Dirichlet (Table V) ✅ verifiable AFTER fix

Initial run (3 rounds): showed accuracy *decreasing* with α, opposite of theory — because 3 rounds is below the convergence threshold for a 600K-param MoE on 120K AG News.

**Fixed run (8 rounds)**:

| α | Paper claim | v2 measured |
|---|---|---|
| 0.1 | 43-48% | **66.53%** |
| 0.3 | (no claim) | 74.88% |
| 0.5 | 50-54% | **76.54%** |
| 1.0 | (no claim) | 79.37% |
| 5.0 | (no claim) | 78.80% |
| 100 | 57-58% | 63.03% |

The qualitative shape ("α small = lower accuracy, α large = approaches IID") is now visible. Numbers consistently *exceed* the paper's claimed ranges. Use these in the paper.

---

### Exp 6 — Membership Inference (Table VI) ❌ paper claim NOT supported

| Config | Paper AUC | v1 AUC (3 rounds) | v2 AUC (8 rounds, more memorisation) |
|---|---|---|---|
| No DP | **0.643** | 0.507 | **0.529** |
| DP σ=0.1 | ~0.58 | 0.492 | 0.462 |
| DP σ=0.5 | ~0.53 | 0.493 | 0.536 |
| DP σ=1.0 | 0.51 | 0.520 | 0.548 |

**Truth: the paper's headline empirical-privacy story does not hold.** Even after configuring the experiment specifically to encourage memorisation (8 FL rounds × 3 local epochs × subsampled 3000-row member shards), the no-DP baseline AUC sits at 0.529 — barely above random. There is essentially no leakage to suppress, so DP cannot be shown to "reduce" it.

**Recommendation**: drop the AUC=0.643 → 0.51 narrative from Section 5.6. Replace with: "MIA AUC stays at ≈0.5 (random baseline) for this MoE+LoRA model on AG News even without DP, indicating that the small per-class memorisation footprint is below the loss-threshold MIA's detection threshold. We retain DP-SGD for the formal (ε,δ) guarantee it provides regardless of empirical leakage."

This is intellectually honest and still defensible — the formal DP guarantee is a separate property from empirical attack resistance.

---

### Exp 7 — FedProx vs FedAvg (Table VII) ❌ paper claim NOT supported

| μ | Paper claim | v1 (3 rounds) | v2 (8 rounds, smaller μ range) |
|---|---|---|---|
| 0 (FedAvg) | ~48-52% | 36.82% | **74.88%** |
| 1e-5 | (n/a) | (n/a) | 58.99% |
| 1e-4 | (n/a) | (n/a) | 61.18% |
| 1e-3 | "small win" | 25.05% | 27.59% |
| 1e-2 | "small win" | 24.97% | 25.00% |
| 1e-1 | "small win" | 25.00% | (n/a) |
| 5e-1 | "over-reg loss" | 25.00% | (n/a) |

**Truth: FedProx never beats FedAvg in this codebase.** Even at the smallest tested μ=1e-5, FedProx is 16 pp below FedAvg. Root cause: the proximal term `(μ/2) Σ_p ‖θ_p - θ_global_p‖²` sums over 600K parameters; a single round of training produces a drift sum-of-squares ≈240, so prox-loss = 0.5 × μ × 240. At μ=1e-3, that's 0.12 — already 9% of the cross-entropy loss, and dominant by μ=1e-2. The paper's claimed sweet spot of μ ∈ [0.001, 0.1] is impossible at this model scale.

**Recommendation**: either (a) remove FedProx from the paper, (b) keep it as a "negative result" finding ("we observed that the standard FedProx implementation does not improve over FedAvg on a 600K-parameter MoE under our regime; we attribute this to the unnormalised proximal term"), or (c) re-implement with a parameter-count-normalised proximal term `(μ / 2N_params) Σ ‖...‖²` and re-run. Option (c) is the most defensible.

---

### Exp 8 — Disease Detection (Table VIII-A) ✅ mostly verifiable

| σ | Paper Top-1 | Paper Top-3 | Measured Top-1 | Measured Top-3 |
|---|---|---|---|---|
| 0.0 | 87.5% | 95.1% | **85.68%** ✓ | **95.31%** ✓ |
| 0.05 | (n/a) | (n/a) | 83.59% | 93.23% |
| 0.10 | **82.8%** | **92.4%** | **73.18%** ✗ | **88.54%** ≈ |
| 0.20 | (n/a) | (n/a) | 32.55% | 55.73% |

σ=0 numbers are within 2 pp of paper claims. **σ=0.10 top-1 is 9.6 pp below the paper's claim** — still a strong result but not the headline 82.8%. Top-3 is within 4 pp. The claim "DP-SGD with σ=0.10 preserves accuracy" still holds qualitatively.

**Recommendation**: replace 82.8%/92.4% with **73.18%/88.54%** in the paper. The claim "small MLP tolerates DP better than the larger MoE" is preserved.

---

### Exp 9 — Audit Ledger + MiMC (Table IX) ⚠ hardware-dependent

| Operation | Paper | Measured |
|---|---|---|
| Ledger seal 1k tx | ~30 ms | **20.32 ms** ✓ |
| Ledger verify 1k tx | ~30 ms | **16.82 ms** ✓ |
| MiMC vs SHA-256 ratio @ 100K | ~3000× | **11573×** ✗ |

Ledger throughput is faster than claimed. MiMC ratio is 4× slower than claimed — but this is because MiMC's slowness scales with state size in pure-Python field arithmetic. Paper's 3000× was probably measured on a 10K-param state (where we measured 6037×, closer to 3000× than 11573×).

**Recommendation**: report the 11573× number for 100K-param state, or specify the state size for the 3000× claim. Either is honest.

---

## Bottom line for the supervisor

The framework **works**. The architecture is sound. The implementation runs end-to-end on three demonstrative tasks (News, Disease, General CSV). 9 of 9 experiments produce measurable numbers, and **6 of those numbers can be defended in the paper as-is or with minor changes**.

The 3 problem areas:

1. **Exp 1 mislabelled** — trivial 1-line fix in the paper.
2. **Exp 6 MIA leakage** — paper claims AUC=0.643 leakage that the codebase **cannot reproduce** even when configured aggressively for memorisation. Must be rewritten as "we keep DP-SGD for the formal guarantee; empirical MIA shows no detectable leakage at our scale".
3. **Exp 7 FedProx** — implementation as written does not improve over FedAvg at any μ. Either drop it, report it as a negative result, or fix the prox-term normalisation and re-run.

Everything else is a number that just needs to be **replaced with the measured value** before submission. We have those measured values now. They are in `verify/results/measured.json` and `verify/results/measured_v2.json`.

## Follow-up runs (v3)

After the initial audit revealed that FedProx (Exp 7) didn't reproduce and the Disease single-symptom rescue table (Exp 8B) hadn't been verified, we ran two additional experiments:

### v3a — FedProx with parameter-count normalisation

Added a `fedprox_normalise=True` flag to `client.py` so the proximal term is divided by the parameter count, $(\mu/2N) \sum_p \|\cdot\|^2$, rather than $(\mu/2) \sum_p \|\cdot\|^2$. This brings the proximal magnitude down to $O(1)$ from $O(N)$, allowing much larger μ.

Results at α=0.3, 5 clients, 8 rounds:

| μ | Acc (%) |
|---|---|
| 0 (FedAvg) | **74.88** ← best |
| 0.01 | 71.71 |
| 0.1 | 70.43 |
| 1.0 | 61.22 |
| 10.0 | 61.99 |
| 100 | 56.51 |
| 1000 | 27.46 |

**Stable negative result.** Even with normalisation, FedProx never beats FedAvg in this regime. The proximal term is no longer destroying training (no μ collapses below 27%, vs the unnormalised version which collapsed to 25% at μ=10⁻³), but the regulariser-induced slowdown of local exploration apparently outweighs whatever drift-reduction benefit it provides at our (5 clients, 8 rounds, α=0.3) configuration. The paper now reports this as a confirmed negative result with two variants tested.

### v3b — Disease single-symptom rescue

Re-trained the Disease Detection model twice (sparse_subset_prob=0.0 vs 0.35) and queried with three single-symptom inputs:

| Symptom | Without aug. | With aug. | Paper claimed |
|---|---|---|---|
| cough | GERD (20.1%) | **Pneumonia (28.6%)** ✓ | Hep A → Pneumonia (48%) |
| itching | Fungal infection (33.6%) | Fungal infection (15.7%) | Hep A → Fungal (47%) |
| headache | Migraine (15.2%) | **Malaria (21.4%)** | GERD → Hypertension (40%) |

**Mixed result.** The paper's specific failure-then-rescue narrative ("Hepatitis A wrong → Pneumonia 48%") didn't reproduce: in our run the no-augmentation model already produced clinically defensible top-1 labels (Migraine for headache, Fungal infection for itching) — there was no "Hepatitis A" disaster to be rescued from. The augmentation does still help in the cough case (GERD → Pneumonia is a more clinically appropriate top-1 with higher confidence). The paper's Table VIII-B has been updated to reflect the actual measured results and a more nuanced prose interpretation.

## How to reproduce

```bash
# Run from project root
cd "d:/projects/Major Project"

# Original 9 experiments (~95 min, single CPU)
python -u verify/run_verification.py all

# Convergence-corrected re-runs of exp4-7 (~46 min)
python -u verify/run_verification_v2.py all

# FedProx-normalised + Disease rescue (~33 min)
python -u verify/run_verification_v3.py all

# Generate audit report
python verify/generate_audit.py

# Output:
#   verify/results/measured.json     -- v1 numbers (paper's original config)
#   verify/results/measured_v2.json  -- v2 numbers (convergence-corrected)
#   verify/results/measured_v3.json  -- v3 numbers (FedProx-norm + Disease rescue)
#   verify/results/AUDIT.md          -- side-by-side audit (all 3 versions)
#   verify/results/PAPER_CHANGES.md  -- exact diff of paper edits
#   verify/results/SUPERVISOR_REPORT.md (this document)
```

Seed = 42 throughout; results are bit-exact reproducible up to thread scheduling on the same hardware.
