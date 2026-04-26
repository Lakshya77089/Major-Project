"""
Hash-chained append-only ledger for zkFedMoE (F9).

Stand-in for the "blockchain layer" promised in IAS-ZKC.  This is a pure-
Python in-memory ledger with the *integrity* properties of a public chain
(append-only, tamper-evident via Merkle roots and per-block hashes) but
without the cost or complexity of Ethereum, gas, wallets, or smart contracts.

What it stores:
    - Client registrations (DID-style identifier)
    - Resource bid commitments + reveals
    - SEPG proof verifications (accepted / rejected) per round
    - Reputation updates per client
    - Reward distributions

Each block contains:
    - block_id (sequential)
    - prev_hash (links to previous block)
    - timestamp
    - tx_list (list of transactions)
    - merkle_root (root of Merkle tree over transactions)
    - block_hash = SHA256(block_id || prev_hash || timestamp || merkle_root)

A tampered transaction breaks the merkle_root, which breaks block_hash, which
breaks every subsequent block (chain integrity).

This module is **pure Python, zero external dependencies**, and is designed
to be transparent for educational and demonstration purposes.  It is NOT a
distributed consensus protocol -- there is no PoW/PoS/BFT here.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ============================================================================
# Transaction
# ============================================================================

@dataclass
class Transaction:
    """A single ledger transaction."""
    tx_type: str           # "register" | "bid_commit" | "bid_reveal" | "verify" | "reputation" | "reward"
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)

    def serialise(self) -> str:
        """Canonical JSON serialisation (sorted keys for determinism)."""
        return json.dumps(
            {"tx_type": self.tx_type, "payload": self.payload, "timestamp": self.timestamp},
            sort_keys=True, default=str,
        )

    def hash(self) -> str:
        return hashlib.sha256(self.serialise().encode("utf-8")).hexdigest()


# ============================================================================
# Merkle Root
# ============================================================================

def merkle_root(tx_hashes: List[str]) -> str:
    """
    Compute a Merkle root over an ordered list of transaction hashes.

    Empty list -> all-zeros (32 bytes hex).
    Pairs of hashes are concatenated and re-hashed; if odd count, the last
    is duplicated (Bitcoin-style).
    """
    if not tx_hashes:
        return "0" * 64
    layer = list(tx_hashes)
    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer.append(layer[-1])
        next_layer = []
        for i in range(0, len(layer), 2):
            h = hashlib.sha256((layer[i] + layer[i + 1]).encode("utf-8")).hexdigest()
            next_layer.append(h)
        layer = next_layer
    return layer[0]


# ============================================================================
# Block
# ============================================================================

@dataclass
class Block:
    block_id: int
    prev_hash: str
    timestamp: float
    tx_list: List[Transaction]
    merkle_root_hex: str
    block_hash: str

    @staticmethod
    def build(block_id: int, prev_hash: str, tx_list: List[Transaction]) -> "Block":
        ts = time.time()
        tx_hashes = [t.hash() for t in tx_list]
        mr = merkle_root(tx_hashes)
        header = f"{block_id}|{prev_hash}|{ts:.6f}|{mr}".encode("utf-8")
        bh = hashlib.sha256(header).hexdigest()
        return Block(
            block_id=block_id,
            prev_hash=prev_hash,
            timestamp=ts,
            tx_list=tx_list,
            merkle_root_hex=mr,
            block_hash=bh,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "block_id": self.block_id,
            "prev_hash": self.prev_hash,
            "timestamp": self.timestamp,
            "merkle_root": self.merkle_root_hex,
            "block_hash": self.block_hash,
            "n_tx": len(self.tx_list),
            "tx_types": [t.tx_type for t in self.tx_list],
        }


# ============================================================================
# Ledger
# ============================================================================

GENESIS_PREV_HASH = "0" * 64


class Ledger:
    """
    Append-only, hash-chained ledger.

    Transactions are buffered in a mempool until ``seal_block`` is called,
    which packages them into a new Block and appends it to the chain.
    """

    def __init__(self):
        self.blocks: List[Block] = []
        self._mempool: List[Transaction] = []
        # Bootstrap with genesis block (always block_id=0, no transactions)
        genesis = Block.build(0, GENESIS_PREV_HASH, [])
        self.blocks.append(genesis)

    # ---- public API ----

    def add_transaction(self, tx_type: str, **payload) -> Transaction:
        tx = Transaction(tx_type=tx_type, payload=dict(payload))
        self._mempool.append(tx)
        return tx

    def seal_block(self) -> Block:
        """Package mempool into a new block and append to chain."""
        prev_hash = self.blocks[-1].block_hash
        block_id = len(self.blocks)
        new_block = Block.build(block_id, prev_hash, list(self._mempool))
        self.blocks.append(new_block)
        self._mempool.clear()
        return new_block

    def latest_block(self) -> Block:
        return self.blocks[-1]

    def height(self) -> int:
        """Number of blocks in the chain (including genesis)."""
        return len(self.blocks)

    def total_transactions(self) -> int:
        return sum(len(b.tx_list) for b in self.blocks)

    # ---- queries ----

    def filter_by_type(self, tx_type: str) -> List[Transaction]:
        out = []
        for b in self.blocks:
            for t in b.tx_list:
                if t.tx_type == tx_type:
                    out.append(t)
        return out

    def filter_by_client(self, client_id: int) -> List[Transaction]:
        out = []
        for b in self.blocks:
            for t in b.tx_list:
                if t.payload.get("client_id") == client_id:
                    out.append(t)
        return out

    # ---- chain integrity ----

    def verify(self) -> tuple:
        """Re-hash every block; return (ok: bool, reason: str)."""
        return verify_chain(self.blocks)

    def to_summary(self) -> List[Dict[str, Any]]:
        return [b.to_dict() for b in self.blocks]


# ============================================================================
# Standalone chain verifier
# ============================================================================

def verify_chain(blocks: List[Block]) -> tuple:
    """
    Walk the chain and verify every link.  Returns (ok, reason).
    """
    if not blocks:
        return False, "empty chain"
    if blocks[0].prev_hash != GENESIS_PREV_HASH:
        return False, "genesis prev_hash != all-zeros"
    for i, b in enumerate(blocks):
        # Re-derive merkle root from transactions
        tx_hashes = [t.hash() for t in b.tx_list]
        mr = merkle_root(tx_hashes)
        if mr != b.merkle_root_hex:
            return False, f"block {i}: merkle_root mismatch"
        # Re-derive block hash
        header = f"{b.block_id}|{b.prev_hash}|{b.timestamp:.6f}|{b.merkle_root_hex}".encode("utf-8")
        bh = hashlib.sha256(header).hexdigest()
        if bh != b.block_hash:
            return False, f"block {i}: block_hash mismatch"
        # Link to previous
        if i > 0 and b.prev_hash != blocks[i - 1].block_hash:
            return False, f"block {i}: prev_hash != blocks[{i-1}].block_hash"
    return True, "chain ok"
