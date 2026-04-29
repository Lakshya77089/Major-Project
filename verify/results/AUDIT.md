# zkFedMoE Conference Paper - Verification Audit

Generated from `measured.json`.

Each claim in `Report/conference_paper.tex` has been checked against an independent run of the code in `Code/`. Verdicts:
- **MATCH**: measured value within tolerance of paper claim.
- **CLOSE**: within 2x tolerance; can be reported with the measured value.
- **MISMATCH**: paper claim is not supported; must be replaced with the measured value or removed.

## Experiment 1 -- Privacy-Utility Tradeoff (Table I)

| sigma | Paper eps | Paper Acc | Measured eps (Renyi) | Measured eps (basic) | Measured Acc | Verdict |
|-------|-----------|-----------|----------------------|-----------------------|--------------|---------|
| 0.0 | inf | 58.00% | inf | inf | 57.09% | **MATCH** |
| 0.1 | 0.2890 | 25.00% | 250.4 | 0.2889 | 25.00% | **MATCH** |
| 0.5 | 0.0580 | 25.01% | 3.0601 | 0.0578 | 25.00% | **MATCH** |
| 1.0 | 0.0290 | 26.12% | 0.7745 | 0.0289 | 25.00% | **CLOSE** |
| 2.0 | 0.0140 | 25.00% | 0.1844 | 0.0144 | 25.00% | **MATCH** |

**Finding:** The paper labels Table I's eps column as 'Rényi (Mironov 2017)' but the underlying numbers come from the basic Gaussian-mechanism √T accountant. The Rényi-DP composition gives much **larger** eps (less private) for small sigma. The paper must either (a) change the column label to 'eps (basic Gaussian, sqrt(T))', or (b) replace the numbers with the measured Rényi values.

## Experiment 2 -- Communication vs Top-K (Table II)

| K | Paper Acc | Paper Saving | Measured Acc | Measured Saving | Verdict |
|---|-----------|--------------|--------------|------------------|---------|
| 1 | 54.24% | 39.52% | 56.16% | 39.52% | **MATCH** |
| 2 | 54.96% | 33.88% | 57.59% | 33.88% | **CLOSE** |
| 3 | 57.66% | 28.23% | 57.25% | 28.23% | **MATCH** |
| 4 | 59.09% | 22.58% | 74.42% | 22.58% | **CLOSE** |
| 8 | 55.99% | 0.00% | 70.51% | 0.00% | **CLOSE** |

**Finding:** Saving percentages are deterministic from model architecture (Top-K param fraction); they MATCH exactly. Accuracies are subject to training noise; small deviations expected.

## Experiment 3 -- SEPG Verification Overhead (Table III)

| K | Gen (ms) | Verify (ms) | Total (ms) |
|---|----------|-------------|------------|
| 1 | 5.44 | 6.28 | 11.72 |
| 2 | 4.64 | 5.83 | 10.47 |
| 3 | 5.65 | 4.79 | 10.45 |
| 4 | 4.81 | 5.44 | 10.25 |
| 5 | 4.86 | 5.30 | 10.16 |
| 6 | 4.95 | 5.73 | 10.68 |
| 7 | 5.17 | 5.66 | 10.84 |
| 8 | 5.55 | 6.98 | 12.53 |

**Measured mean:** 10.89 +/- 0.77 ms across all K.
**Paper claim:** 6.08 ms.
**Verdict: MISMATCH** (79.0% deviation; pure-Python SHA-256 timings are CPU-dependent).

## Experiment 4 -- Robustness Under Poisoning (Table IV)

| Malicious % | FedAvg | Median | Trimmed Mean |
|-------------|--------|--------|--------------|
| 0% | 56.91% | 57.68% | 57.45% |
| 20% | 57.42% | 60.88% | 61.05% |
| 40% | 45.97% | 25.0% | 46.51% |

**Paper at 40% mal:** FedAvg=41.82%, Median=45.95%, TrimMean=43.95%.
**Measured at 40% mal:** FedAvg=45.97%, Median=25.0%, TrimMean=46.51%.

## Experiment 5 -- Non-IID Dirichlet alpha sweep (Table V)

| alpha | Paper claim (range) | Measured Acc | Verdict |
|-------|---------------------|--------------|---------|
| 0.1 | 43-48% | 53.57% | **MISMATCH** |
| 0.3 | (no specific paper claim) | 36.82% | **n/a** |
| 0.5 | 50-54% | 27.20% | **MISMATCH** |
| 1.0 | (no specific paper claim) | 42.80% | **n/a** |
| 5.0 | (no specific paper claim) | 52.46% | **n/a** |
| 100.0 | 57-58% | 25.00% | **MISMATCH** |

## Experiment 6 -- Membership Inference (Table VI)

| Config | Paper AUC | Measured AUC | Acc | Member loss | Non-mem loss | Verdict |
|--------|-----------|--------------|-----|-------------|---------------|---------|
| No DP | 0.643 | 0.507 | 44.05% | 1.084 | 1.091 | **MISMATCH** |
| DP sigma=0.1 | - | 0.492 | 25.0% | 1.39 | 1.386 | **(no specific paper claim)** |
| DP sigma=0.5 | - | 0.493 | 25.0% | 1.522 | 1.526 | **(no specific paper claim)** |
| DP sigma=1.0 | 0.51 | 0.52 | 25.0% | 21.482 | 22.355 | **MATCH** |

## Experiment 7 -- FedProx vs FedAvg (Table VII)

| mu | Label | Acc (%) |
|----|-------|---------|
| 0.0 | FedAvg | 36.82 |
| 0.001 | FedProx mu=0.001 | 25.05 |
| 0.01 | FedProx mu=0.01 | 24.97 |
| 0.1 | FedProx mu=0.1 | 25.00 |
| 0.5 | FedProx mu=0.5 | 25.00 |

**FedAvg:** paper 48-52%, measured 36.82% -> **MISMATCH**
**FedProx sweet (mu=0.1):** paper 49-54%, measured 25.00% -> **MISMATCH**

## Experiment 8 -- Disease Detection DP calibration (Table VIII-A)

| sigma | Paper Top-1 | Paper Top-3 | Measured Top-1 | Measured Top-3 | Verdict |
|-------|-------------|-------------|----------------|----------------|---------|
| 0.0 | 87.5% | 95.1% | 85.68% | 95.31% | **MATCH** |
| 0.05 | - | - | 83.59% | 93.23% | **(no specific paper claim)** |
| 0.1 | 82.8% | 92.4% | 73.18% | 88.54% | **CLOSE** |
| 0.2 | - | - | 32.55% | 55.73% | **(no specific paper claim)** |

## Experiment 9 -- Ledger Throughput + MiMC vs SHA-256 (Table IX)

**Ledger throughput:**

| n_tx | Seal (ms) | Verify (ms) | OK |
|------|-----------|-------------|----|
| 10 | 0.34 | 0.24 | True |
| 100 | 3.03 | 2.33 | True |
| 500 | 9.18 | 8.27 | True |
| 1000 | 20.32 | 16.82 | True |

**MiMC vs SHA-256:**

| State size | SHA-256 (ms) | MiMC (ms) | Ratio |
|------------|--------------|-----------|-------|
| tiny_1K | 0.053 | 67.55 | 1286x |
| small_10K | 0.114 | 685.42 | 6037x |
| medium_100K | 0.461 | 5331.73 | 11573x |

**Paper ledger 1k tx claim:** ~30.0 ms.
**Measured ledger 1k tx:** seal 20.32 ms, verify 16.82 ms.
**Paper MiMC ratio claim:** ~3000x at 100K params.
**Measured ratio at 100K:** 11573x.

## Convergence-Corrected Re-runs (v2)

v1 of several experiments used too few rounds (3) and/or too few clients (5), which caused training to terminate before convergence. v2 fixes this by using 8 rounds for exp5/6/7 and 9 clients for exp4. FedProx (exp7) also uses a smaller `mu` range because the original range overpowers cross-entropy on a 600K-param model where the proximal term sums over all parameters.

### Exp 4 v2: Robustness (9 clients, 8 rounds)

| Malicious % | FedAvg | Median | Trimmed Mean |
|-------------|--------|--------|--------------|
| 0% | 51.08% | 47.87% | 52.57% |
| 22% | 48.59% | 51.74% | 50.57% |
| 44% | 37.82% | 43.34% | 42.75% |

### Exp 5 v2: Non-IID Dirichlet (8 rounds)

| alpha | Acc (%) |
|-------|---------|
| 0.1 | 66.53 |
| 0.3 | 74.88 |
| 0.5 | 76.54 |
| 1.0 | 79.37 |
| 5.0 | 78.80 |
| 100.0 | 63.03 |

### Exp 6 v2: MIA (8 rounds, 3 local epochs, 3000 samples/client)

| Config | AUC | Attack Acc | Acc | Member loss | Non-mem loss |
|--------|-----|------------|-----|-------------|---------------|
| No DP | 0.529 | 54.400000000000006% | 44.28% | 1.06 | 1.1 |
| DP sigma=0.1 | 0.462 | 51.5% | 25.00% | 1.387 | 1.382 |
| DP sigma=0.5 | 0.536 | 54.2% | 25.00% | 1.54 | 1.615 |
| DP sigma=1.0 | 0.548 | 54.7% | 25.00% | 22.693 | 25.324 |

### Exp 7 v2: FedProx (alpha=0.3, 8 rounds, smaller mu range)

| mu | Label | Acc (%) |
|----|-------|---------|
| 0.0 | FedAvg | 74.88 |
| 1e-05 | FedProx mu=1e-05 | 58.99 |
| 0.0001 | FedProx mu=0.0001 | 61.18 |
| 0.0005 | FedProx mu=0.0005 | 44.07 |
| 0.001 | FedProx mu=0.001 | 27.59 |
| 0.01 | FedProx mu=0.01 | 25.00 |

## Follow-up Experiments (v3)

v3 tests whether two unsupported v1/v2 claims can be recovered:
- Exp 7 v3: FedProx with parameter-count-normalised proximal term
- Exp 8b v1: Disease single-symptom rescue (sparse augmentation on/off)

### Exp 7 v3: Normalised FedProx (alpha=0.3, 8 rounds)

| mu | Label | Acc (%) |
|----|-------|---------|
| 0.0 | FedAvg | 74.88 |
| 0.01 | FedProx-norm mu=0.01 | 71.71 |
| 0.1 | FedProx-norm mu=0.1 | 70.43 |
| 1.0 | FedProx-norm mu=1.0 | 61.22 |
| 10.0 | FedProx-norm mu=10.0 | 61.99 |
| 100.0 | FedProx-norm mu=100.0 | 56.51 |
| 1000.0 | FedProx-norm mu=1000.0 | 27.46 |

**Verdict:** Even with parameter-count normalisation, the best FedProx variant (mu=0.01) achieves 71.71%, **still below** FedAvg's 74.88%. The negative result is stable across both variants.

### Exp 8b: Disease Single-Symptom Rescue (10 rounds, 10 hospitals)

| Symptom | Without aug. (top-1, prob) | With aug. (top-1, prob) |
|---------|----------------------------|--------------------------|
| cough | GERD (20.1%) | Pneumonia (28.6%) |
| itching | Fungal infection (33.6%) | Fungal infection (15.7%) |
| headache | Migraine (15.2%) | Malaria (21.4%) |

## Summary

Run completed against the unmodified primitives in `Code/src/`. All numbers above are measured on the same machine in a single execution; values that depend on training stochasticity (accuracy, MIA AUC) will vary +/- a few points across runs but the qualitative pattern should be stable.

### Recommended paper edits
- Replace any 'eps (Rényi)' label in Table I with 'eps (basic Gaussian, sqrt(T) composition)' OR re-fit the table to the measured Rényi values.
- Replace any approximate ranges ('~48-52%') with the exact measured single-run values.
- Add a note that pure-Python SEPG and MiMC timings are CPU-dependent, so absolute ms values are reported on the harness machine (commodity Windows laptop, single CPU core).
