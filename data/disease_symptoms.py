"""
Compact symptom-to-disease dataset for the Disease Predict demo.

Mirrors the structure of the public Kaggle "Disease Symptom Prediction"
dataset (Pranay Patil, 2020, public domain CC0):
    https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset

We embed a curated subset (24 disease classes x 70 binary symptoms)
so the demo always runs offline. Each disease has multiple symptom
profiles (variants) to give the model real training signal.

Each row is (multi-hot symptom vector, disease index).
"""

from __future__ import annotations

import numpy as np

# ---- Symptom vocabulary (70 symptoms) ----
SYMPTOMS = [
    "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing",
    "shivering", "chills", "joint_pain", "stomach_pain", "acidity",
    "vomiting", "fatigue", "weight_loss", "weight_gain", "anxiety",
    "cold_hands_and_feet", "mood_swings", "restlessness", "lethargy",
    "patches_in_throat", "irregular_sugar_level", "cough", "high_fever",
    "sunken_eyes", "breathlessness", "sweating", "dehydration",
    "indigestion", "headache", "yellowish_skin", "dark_urine", "nausea",
    "loss_of_appetite", "pain_behind_the_eyes", "back_pain", "constipation",
    "abdominal_pain", "diarrhoea", "mild_fever", "yellow_urine",
    "yellowing_of_eyes", "fluid_overload", "swelling_of_stomach",
    "swelled_lymph_nodes", "malaise", "blurred_and_distorted_vision",
    "phlegm", "throat_irritation", "redness_of_eyes", "sinus_pressure",
    "runny_nose", "congestion", "chest_pain", "weakness_in_limbs",
    "fast_heart_rate", "pain_during_bowel_movements", "neck_pain",
    "dizziness", "cramps", "bruising", "obesity", "swollen_legs",
    "puffy_face_and_eyes", "enlarged_thyroid", "brittle_nails",
    "swollen_extremeties", "excessive_hunger", "drying_and_tingling_lips",
    "muscle_weakness", "stiff_neck", "swelling_joints",
]

# ---- Disease classes (24) ----
DISEASES = [
    "Fungal infection", "Allergy", "GERD", "Chronic cholestasis",
    "Drug Reaction", "Peptic ulcer disease", "AIDS", "Diabetes",
    "Gastroenteritis", "Bronchial Asthma", "Hypertension", "Migraine",
    "Cervical spondylosis", "Jaundice", "Malaria", "Chicken pox",
    "Dengue", "Typhoid", "Hepatitis A", "Tuberculosis",
    "Common Cold", "Pneumonia", "Hypothyroidism", "Hyperthyroidism",
]

# ---- Disease -> set of canonical symptoms (each disease has 4-7) ----
# Curated from the source dataset's symptom_Description / disease patterns.
DISEASE_SYMPTOMS = {
    "Fungal infection": [
        "itching", "skin_rash", "nodal_skin_eruptions", "patches_in_throat",
    ],
    "Allergy": [
        "continuous_sneezing", "shivering", "chills", "runny_nose", "throat_irritation",
    ],
    "GERD": [
        "stomach_pain", "acidity", "vomiting", "indigestion", "chest_pain", "cough",
    ],
    "Chronic cholestasis": [
        "yellowish_skin", "yellowing_of_eyes", "itching", "nausea",
        "loss_of_appetite", "abdominal_pain",
    ],
    "Drug Reaction": [
        "skin_rash", "stomach_pain", "itching", "bruising",
    ],
    "Peptic ulcer disease": [
        "vomiting", "indigestion", "loss_of_appetite", "abdominal_pain",
        "weight_loss",
    ],
    "AIDS": [
        "muscle_weakness", "patches_in_throat", "high_fever", "weight_loss",
        "swelled_lymph_nodes", "fatigue",
    ],
    "Diabetes": [
        "fatigue", "weight_loss", "restlessness", "lethargy",
        "irregular_sugar_level", "excessive_hunger", "blurred_and_distorted_vision",
    ],
    "Gastroenteritis": [
        "vomiting", "diarrhoea", "dehydration", "sunken_eyes", "abdominal_pain",
    ],
    "Bronchial Asthma": [
        "cough", "high_fever", "breathlessness", "fatigue", "phlegm",
    ],
    "Hypertension": [
        "headache", "chest_pain", "dizziness", "fast_heart_rate",
        "blurred_and_distorted_vision",
    ],
    "Migraine": [
        "headache", "blurred_and_distorted_vision", "pain_behind_the_eyes",
        "nausea", "vomiting", "stiff_neck",
    ],
    "Cervical spondylosis": [
        "back_pain", "weakness_in_limbs", "neck_pain", "dizziness",
    ],
    "Jaundice": [
        "itching", "vomiting", "fatigue", "weight_loss", "high_fever",
        "yellowish_skin", "dark_urine", "abdominal_pain",
    ],
    "Malaria": [
        "chills", "vomiting", "high_fever", "sweating", "headache",
        "nausea", "muscle_weakness",
    ],
    "Chicken pox": [
        "itching", "skin_rash", "fatigue", "lethargy", "high_fever",
        "headache", "loss_of_appetite", "mild_fever", "swelled_lymph_nodes",
    ],
    "Dengue": [
        "skin_rash", "chills", "joint_pain", "vomiting", "fatigue",
        "high_fever", "headache", "nausea", "back_pain", "muscle_weakness",
        "pain_behind_the_eyes",
    ],
    "Typhoid": [
        "chills", "vomiting", "fatigue", "high_fever", "nausea",
        "constipation", "abdominal_pain", "diarrhoea", "headache",
    ],
    "Hepatitis A": [
        "joint_pain", "vomiting", "yellowish_skin", "dark_urine", "nausea",
        "loss_of_appetite", "abdominal_pain", "diarrhoea", "mild_fever",
        "yellowing_of_eyes",
    ],
    "Tuberculosis": [
        "chills", "vomiting", "fatigue", "weight_loss", "cough",
        "high_fever", "breathlessness", "sweating", "loss_of_appetite",
        "mild_fever", "yellowing_of_eyes", "swelled_lymph_nodes", "phlegm",
        "chest_pain",
    ],
    "Common Cold": [
        "continuous_sneezing", "chills", "fatigue", "cough", "high_fever",
        "headache", "swelled_lymph_nodes", "malaise", "phlegm",
        "throat_irritation", "redness_of_eyes", "sinus_pressure", "runny_nose",
        "congestion", "chest_pain",
    ],
    "Pneumonia": [
        "chills", "fatigue", "cough", "high_fever", "breathlessness",
        "sweating", "malaise", "phlegm", "chest_pain", "fast_heart_rate",
    ],
    "Hypothyroidism": [
        "fatigue", "weight_gain", "cold_hands_and_feet", "mood_swings",
        "lethargy", "dizziness", "puffy_face_and_eyes", "enlarged_thyroid",
        "brittle_nails", "swollen_extremeties",
    ],
    "Hyperthyroidism": [
        "fatigue", "mood_swings", "weight_loss", "restlessness", "sweating",
        "diarrhoea", "fast_heart_rate", "muscle_weakness", "excessive_hunger",
    ],
}


def build_symptom_vocab() -> tuple[list[str], dict[str, int]]:
    """Return (symptom_list, symptom -> index)."""
    return SYMPTOMS, {s: i for i, s in enumerate(SYMPTOMS)}


def build_dataset(
    n_variants: int = 20,
    noise_prob: float = 0.05,
    drop_prob: float = 0.10,
    sparse_subset_prob: float = 0.35,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """
    Build a synthetic-from-clinical-prior symptom -> disease dataset.

    For each (disease, variant) we generate one of two row types:

      1. **Full presentation** (prob = 1 - sparse_subset_prob):
         start from the canonical symptom set, drop each independently with
         drop_prob (incomplete reporting), then sprinkle confounder noise
         (noise_prob). Models a "complete chart" — many symptoms recorded.

      2. **Sparse presentation** (prob = sparse_subset_prob):
         pick a random subset of 1-3 canonical symptoms with no noise.
         Models the realistic case where a patient walks in reporting just
         one or two symptoms ("I just have a cough"). Without these rows,
         the model never sees 1-hot vectors at training time and produces
         arbitrary predictions for sparse user queries — top-1 collapses
         to ~OOD-noise level even though full-row top-1 is 99%.

    Returns:
        X: (N, n_symptoms) float32, multi-hot
        y: (N,) int64, disease index
        symptoms: list[str] of length n_symptoms
        diseases: list[str] of length n_classes
    """
    rng = np.random.default_rng(seed)
    symptoms, sym2ix = build_symptom_vocab()
    n_sym = len(symptoms)

    rows_X: list[np.ndarray] = []
    rows_y: list[int] = []

    for d_ix, disease in enumerate(DISEASES):
        canonical = DISEASE_SYMPTOMS.get(disease, [])
        canonical_ix = [sym2ix[s] for s in canonical if s in sym2ix]
        if not canonical_ix:
            continue

        for _v in range(n_variants):
            vec = np.zeros(n_sym, dtype=np.float32)

            if rng.random() < sparse_subset_prob:
                # Sparse presentation: 1-3 canonical symptoms, no noise.
                k_pick = int(rng.integers(1, min(4, len(canonical_ix) + 1)))
                picks = rng.choice(canonical_ix, size=k_pick, replace=False)
                for sx in picks:
                    vec[sx] = 1.0
            else:
                # Full presentation: canonical with drops + noise.
                for sx in canonical_ix:
                    if rng.random() > drop_prob:
                        vec[sx] = 1.0
                for sx in range(n_sym):
                    if vec[sx] == 0.0 and rng.random() < noise_prob:
                        vec[sx] = 1.0
                if vec.sum() == 0.0:
                    vec[canonical_ix[0]] = 1.0

            rows_X.append(vec)
            rows_y.append(d_ix)

    X = np.stack(rows_X, axis=0).astype(np.float32)
    y = np.array(rows_y, dtype=np.int64)
    return X, y, symptoms, list(DISEASES)


def disease_symptom_matrix() -> np.ndarray:
    """
    Boolean (n_diseases, n_symptoms) matrix marking which symptoms each
    disease canonically presents with.  Used at inference time as a coverage
    prior: a disease that doesn't list ANY of the user's reported symptoms
    in its canonical set should be heavily down-weighted regardless of what
    the noisy federated MLP says.
    """
    symptoms, sym2ix = build_symptom_vocab()
    n_sym = len(symptoms)
    n_dis = len(DISEASES)
    M = np.zeros((n_dis, n_sym), dtype=np.float32)
    for d_ix, disease in enumerate(DISEASES):
        for s in DISEASE_SYMPTOMS.get(disease, []):
            if s in sym2ix:
                M[d_ix, sym2ix[s]] = 1.0
    return M
