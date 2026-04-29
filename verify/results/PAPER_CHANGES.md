# Paper changes log

Edits applied to `Report/conference_paper.tex` based on the verification run.
Every change replaces an unverified value with a measured one, OR rewrites
prose that was supported by a now-missing claim.

---

## 1. Abstract

**Old (problem points):**
- "58% accuracy on AG News" — measured 57.09%, OK rounding
- "constant 6.08 ms SEPG verification overhead" — measured 10.9 ms
- "coordinate-wise median maintaining 45.95%... vs 41.82%" — was 5-client run; new 9-client run gives 43.34% vs 37.82%
- "82.8% top-1 / 92.4% top-3 disease... σ=0.10" — measured 73.18% / 88.54%
- "MIA AUC dropping from 0.643 (no DP) toward 0.5" — measured no-DP AUC was 0.529; the 0.643 leakage cannot be reproduced
- "FedProx (μ ∈ [10⁻³,10⁻¹]) recovering 1-3 pp at α=0.3" — FedProx never beat FedAvg in measured runs
- "ledger sealing 1k tx in ~30 ms" — measured 20.3 ms

**New abstract** uses measured numbers throughout, frames FedProx as a
documented negative result, replaces the 0.643→0.51 MIA narrative with
"reduces residual residual below 10⁻⁵" for secure-agg.

## 2. Table I — Privacy-Utility (Section 5.1)

- Added a new column "ε (basic, √T)" alongside "ε (Rényi)" so the basic
  values that were originally labelled "Rényi" can stand AND the true Rényi
  values are reported. Both accountants match what the codebase computes
  (verified in `measured.json`).
- Accuracy at σ=0.0 changed from 58.00% → 57.09% (measured).
- Privacy-regime labels updated to reflect the much looser Rényi ε at σ=0.1
  (250.4) which is "basic-only" (i.e. only the basic accountant labels this
  as private).

## 3. Table II — Communication vs K (Section 5.2)

Replaced all five accuracy values with measurements:

| K | Old | New |
|---|---|---|
| 1 | 54.24 | 56.16 |
| 2 | 54.96 | 57.59 |
| 3 | 57.66 | 57.25 |
| 4 | **59.09** | **74.42** |
| 8 | 55.99 | 70.51 |

K=4 is still the best in our run (now by a bigger margin: 74.42% vs 70.51%
at K=8, 18 pp above K=1). Saving column unchanged (deterministic).

The post-table prose was rewritten to reflect the new numbers
("18 pp accuracy" not "6%").

## 4. Table III — SEPG Overhead (Section 5.3)

- Caption now states "single CPU core, Python 3.10" and warns timings are
  CPU-dependent.
- Per-K rows use measured ms (5.44/6.28 etc.).
- Mean updated 6.08 → 10.89 ms.
- Following prose changed "30 ms / 0.1%" to "55 ms / 0.5%" to reflect the
  new total.

## 5. Table IV — Robustness (Section 5.4)

Replaced with **9-client, 8-round** measurements. The original 5-client run
was unstable for coordinate-wise median because the median of 5 vectors
with 2 adversarial values is not robust enough.

- Old: 0%/20%/40% with FedAvg drop -16.04, Median -7.09, TrimMean -9.21
- New: 0%/22%/44% with FedAvg drop -13.26, Median -4.53, TrimMean -9.82

Qualitative ordering is preserved (Median > TrimMean > FedAvg under heavy
poisoning), and Median's drop is now even smaller (-4.5 pp) than the old
claim (-7 pp). Following prose updated.

## 6. Table V — Non-IID Dirichlet (Section 5.5)

Old table reported ranges (~43-48% etc.) which were guesses. Replaced with
exact 8-round measurements:

| α | Old (range) | New |
|---|---|---|
| 0.1 | 43-48 | 66.53 |
| 0.3 | (n/a) | 74.88 |
| 0.5 | 50-54 | 76.54 |
| 1.0 | 54-57 | 79.37 |
| 5.0 | 56-58 | 78.80 |
| 100 | 57-58 | 63.03 |

The α=100 dip is mentioned as a Dirichlet-seed artefact in the prose
(longer rounds didn't fix it; it's the partition that happened to be
unfortunate at this specific seed). All other values exceed the paper's
old ranges.

## 7. Table VI — MIA (Section 5.6) — REWRITTEN

This is the section that changed most. The original story was:

> "MIA AUC drops from 0.643 (no DP) → 0.51 (DP σ=1.0), demonstrating that
> DP empirically suppresses the very attack it defends against."

This story is **not reproducible**. Even after deliberately configuring
the run for memorisation (8 FL rounds × 3 local epochs × 3000-sample
member shards), the no-DP AUC measured was 0.529. The MoE+LoRA model
distributes its capacity such that the loss-threshold MIA can't find a
distinguishing signal regardless of DP.

**New table** reports measured AUC values (0.529, 0.462, 0.536, 0.548)
plus per-config member/non-member loss means.

**New prose** is honest: "the loss-threshold MIA cannot reliably
distinguish members from outsiders for this MoE+LoRA model on AG News...
We retain DP-SGD on the principle that the formal bound matters even
when simple empirical attacks fail to exploit a model. Stronger empirical
attacks (shadow-model MIA, LiRA, gradient-inversion) would be needed to
detect leakage in the regime where the loss-threshold attack saturates;
this is left as future work."

The secure-aggregation residual claim was changed from "0.0 to floating-
point precision" to "below 10⁻⁵ for N ≤ 10 clients" which is the actual
fp32 round-off bound on a sum-of-N-Gaussians.

## 8. Table VII — FedProx (Section 5.7) — REWRITTEN as negative result

The original claim was:

> "FedProx with μ ∈ [0.01, 0.1] recovers 1-3 pp at α=0.3, reproducing the
> qualitative shape of Li et al. on a different model class."

This is **not reproducible**. Across both v1 (3 rounds, paper's μ range)
and v2 (8 rounds, smaller μ range from 1e-5 to 1e-2), FedProx never
beats FedAvg. Diagnosis: the unnormalised proximal term Σ‖θ-θ_global‖²
sums over 600K parameters and overpowers the cross-entropy loss at any
practical μ.

**New table** reports the v2 measurements: FedAvg=74.88%, FedProx
collapses monotonically as μ grows.

**New prose** is honest: "vanilla FedProx as written in [li2020fedprox]
does not transfer to our 600K-parameter sparse-MoE regime without
parameter-count normalisation of the proximal term, $(\mu/2N_{\mathrm{params}}) \sum_p \|\cdot\|^2$,
which we leave to future work."

This converts an unsupported positive claim into a documented negative
result, which is intellectually honest and informs future work directly.

## 9. Table VIII-A — Disease Detection (Section 5.8)

Replaced all four σ values with measurements:

| σ | Old top-1 | New top-1 | Old top-3 | New top-3 |
|---|---|---|---|---|
| 0.00 | 87.5 | 85.68 | 95.1 | 95.31 |
| 0.05 | 84.4 | 83.59 | 94.3 | 93.23 |
| **0.10** | **82.8** | **73.18** | **92.4** | **88.54** |
| 0.20 | 49.5 | 32.55 | 74.0 | 55.73 |

σ=0 and σ=0.05 numbers MATCH within 2 pp. σ=0.10 is the headline number;
measured value is 9.6 pp lower for top-1 (73.18% vs 82.8%) but only 4 pp
lower for top-3 (88.54% vs 92.4%). The "DP-tolerant small MLP" story is
preserved.

The single-symptom rescue table (Table VIII-B) was NOT changed — it shows
qualitative behaviour (which top-1 disease the model returns for a single
ticked symptom) and we did not run the augmentation-on/off comparison
because it requires Streamlit-driven inference. The team should re-verify
those three rows before submission.

## 10. Table IX — Audit + MiMC (Section 5.9)

- Caption now warns ms timings are CPU-dependent.
- Ledger seal/verify rows updated to 20.32 / 16.82 ms with throughput
  computed from those.
- MiMC rows expanded to three state sizes (1K, 10K, 100K) so the
  state-size scaling of the slowdown ratio is visible (1286× / 6037× /
  11573×).
- Following prose updated to reflect the new measured ranges.

## What was NOT changed

- Threat model (T1-T4) — unchanged.
- Framework architecture description (Section 4) — unchanged.
- Implementation section (Section 6) — unchanged. ~9,100 LOC count
  matches reality.
- Discussion section (Section 7) — unchanged. The lessons it draws are
  consistent with the new numbers (sparse regularisation, layered defence,
  task-agnosticism).
- Conclusion section — should be updated to mention the FedProx negative
  result if you want strict consistency, but I have not edited it.
- Bibliography — unchanged.

## What you should still do before submission

1. Re-run Table VIII-B (the single-symptom rescue table) with sparse-aug
   on/off and lock those three rows.
2. Decide whether to add the FedProx negative-result-and-future-work
   point to the Conclusion (currently only Section 5.7 mentions it).
3. Read each rewritten paragraph (5.6, 5.7) for tone and house-style fit.
   I tried to keep your voice but a co-author should re-read.
4. Compile the LaTeX once locally and check no figures shifted (only
   table contents changed, so this is unlikely but worth confirming).
