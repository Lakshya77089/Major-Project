"""
ZK-friendly hash primitives for zkFedMoE (F8).

SHA-256 is great for *integrity* but expensive to prove inside a zk-SNARK
circuit (lots of bit-twiddling, ~25K constraints per hash).  ZK-friendly
hashes like MiMC and Poseidon use only field-multiplication and addition,
making them 10-100x cheaper to prove.

This module provides a pure-Python MiMC-Feistel implementation as a drop-in
replacement for SHA-256 inside SEPG proofs.  We do NOT replace the default
``_hash_state`` in sepg.py; instead this is exposed as an *alternative*
hashing function that future ZK-SNARK integration would use.

Reference: Albrecht et al., "MiMC: Efficient Encryption and Cryptographic
Hashing with Minimal Multiplicative Complexity", ASIACRYPT 2016.

Caveats:
  * This is for demonstration / future ZK upgrade path only.  Pure-Python
    field arithmetic is *slow* compared to SHA-256.
  * The chosen field is BN254 scalar field (commonly used in Ethereum
    ZK-SNARKs); 254-bit prime.
"""

from __future__ import annotations

import hashlib
import io
from typing import Dict, Iterable

import torch


# BN254 scalar field prime (used in Groth16/PLONK on Ethereum)
BN254_PRIME = 21888242871839275222246405745257275088548364400416034343698204186575808495617

# Number of MiMC rounds for collision resistance at 128-bit security
MIMC_ROUNDS = 91

# Round constants: deterministically derived from SHA-256 of round indices
# (so they are pseudo-random but reproducible).
def _round_constants(p: int = BN254_PRIME, n: int = MIMC_ROUNDS) -> list:
    cs = [0]  # first round constant is 0 by convention
    for i in range(1, n):
        digest = hashlib.sha256(f"mimc_round_{i}".encode("utf-8")).digest()
        cs.append(int.from_bytes(digest, "big") % p)
    return cs


_RC = _round_constants()


def mimc_permute(x: int, k: int = 0, p: int = BN254_PRIME) -> int:
    """
    MiMC permutation in a prime field.  x is the input element, k is the key.

    For each round i:  x <- (x + k + RC_i)^7 mod p
    Final output:      x + k mod p
    """
    for rc in _RC:
        t = (x + k + rc) % p
        # cube + cube + cube = power of 7 with squarings; here use ** 7 directly
        x = pow(t, 7, p)
    return (x + k) % p


def mimc_hash(elements: Iterable[int], p: int = BN254_PRIME) -> int:
    """
    Sponge-style absorb of a list of field elements using the MiMC permutation.

    Returns a single field element (the hash).
    """
    state = 0
    key = 0
    for elem in elements:
        # Reduce input element modulo p
        e = elem % p
        # Sponge absorb: XOR-style addition + permutation
        state = (state + e) % p
        state = mimc_permute(state, key, p)
        key = state
    return state


def _bytes_to_field_elements(buf: bytes, p: int = BN254_PRIME) -> list:
    """Split bytes into 31-byte chunks, each interpreted as a field element."""
    chunk = 31  # 31 bytes = 248 bits, fits inside BN254 (254 bits)
    elems = []
    for i in range(0, len(buf), chunk):
        sub = buf[i:i + chunk]
        elems.append(int.from_bytes(sub, "big") % p)
    return elems


def mimc_hash_state(state: Dict[str, torch.Tensor]) -> str:
    """
    ZK-friendly replacement for the SHA-256 ``_hash_state`` in sepg.py.

    Same input format (dict of named tensors), same deterministic
    serialisation, but uses MiMC for the final hash.  Returns a hex string
    so it can be a drop-in replacement.
    """
    buf = io.BytesIO()
    for key in sorted(state.keys()):
        buf.write(key.encode("utf-8"))
        buf.write(state[key].cpu().float().numpy().tobytes())
    elements = _bytes_to_field_elements(buf.getvalue())
    digest = mimc_hash(elements)
    return f"{digest:064x}"


def benchmark_compare(state: Dict[str, torch.Tensor], n_trials: int = 5) -> Dict[str, float]:
    """Helper for the dashboard / experiments: timing of SHA-256 vs MiMC."""
    import time
    # SHA-256
    sha_t0 = time.perf_counter()
    for _ in range(n_trials):
        h = hashlib.sha256()
        for key in sorted(state.keys()):
            h.update(key.encode("utf-8"))
            h.update(state[key].cpu().float().numpy().tobytes())
        _ = h.hexdigest()
    sha_ms = (time.perf_counter() - sha_t0) / n_trials * 1000.0

    # MiMC
    mimc_t0 = time.perf_counter()
    for _ in range(n_trials):
        _ = mimc_hash_state(state)
    mimc_ms = (time.perf_counter() - mimc_t0) / n_trials * 1000.0

    return {"sha256_ms": sha_ms, "mimc_ms": mimc_ms, "ratio": mimc_ms / max(sha_ms, 1e-9)}
