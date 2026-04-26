"""
Commitment-based resource bidding for zkFedMoE (F4).

Stand-in for the "ZK-committed resource bids" promised in IAS-ZKC.  Uses a
Pedersen-style hiding commitment built from SHA-256: each client commits
``hash(bandwidth || compute || storage || nonce)``, then later reveals the
plaintext for the server to verify.

The commitment scheme is *binding* (cannot change the bid after committing)
and *hiding* (server cannot reverse-engineer the bid before reveal because
of the random nonce).  Real ZK-SNARK based commitments give the same two
properties; this is the lightweight cryptographic equivalent that runs in
pure Python with no external dependencies.

Usage flow per round:
    1.  Each client picks (bandwidth, compute, storage), generates a random
        nonce, and computes the commitment hash.  Sends commitment + client_id.
    2.  Server collects all commitments.  At this stage the server knows
        nothing about the bids (hiding property).
    3.  Each client reveals (bid, nonce).  Server runs ``verify_bid`` which
        re-hashes and checks the commitment matches.  Tampered or fabricated
        post-hoc bids are caught.
    4.  Server runs an auction (e.g. select clients with highest combined
        compute + bandwidth offer for the next round).
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ResourceBid:
    """Plaintext resource offer from one client."""
    client_id: int
    bandwidth_mbps: float
    compute_gflops: float
    storage_mb: float
    nonce: str  # hex string -- random per bid

    def serialise(self) -> bytes:
        return (
            f"{self.client_id}|"
            f"{self.bandwidth_mbps:.6f}|"
            f"{self.compute_gflops:.6f}|"
            f"{self.storage_mb:.6f}|"
            f"{self.nonce}"
        ).encode("utf-8")


@dataclass
class BidCommitment:
    """Binding+hiding commitment to a bid (sent before reveal)."""
    client_id: int
    round_id: int
    commitment_hash: str          # SHA-256 of bid serialisation
    timestamp_committed: float = 0.0


def fresh_nonce(num_bytes: int = 16) -> str:
    """Cryptographically random hex nonce."""
    return secrets.token_hex(num_bytes)


def commit_bid(bid: ResourceBid, round_id: int) -> BidCommitment:
    """Produce a hiding commitment for a bid.

    Hiding requires the nonce in ``bid`` to be unpredictable; use
    :func:`fresh_nonce` to generate it.
    """
    digest = hashlib.sha256(bid.serialise()).hexdigest()
    return BidCommitment(
        client_id=bid.client_id,
        round_id=round_id,
        commitment_hash=digest,
    )


def verify_bid(commitment: BidCommitment, revealed: ResourceBid) -> tuple:
    """
    Verify that a revealed bid matches the earlier commitment.

    Returns (passed: bool, reason: str).
    """
    if revealed.client_id != commitment.client_id:
        return False, (
            f"client_id mismatch: revealed {revealed.client_id} "
            f"vs committed {commitment.client_id}"
        )
    digest = hashlib.sha256(revealed.serialise()).hexdigest()
    if digest != commitment.commitment_hash:
        return False, (
            "commitment hash mismatch — bid altered or wrong nonce"
        )
    if revealed.bandwidth_mbps < 0 or revealed.compute_gflops < 0 or revealed.storage_mb < 0:
        return False, "negative resource offer"
    return True, "ok"


@dataclass
class AuctionResult:
    accepted_client_ids: List[int]
    winning_score: float
    rejected_count: int
    leaderboard: List[tuple]  # list of (client_id, score)


def run_auction(
    commitments: Dict[int, BidCommitment],
    revealed_bids: Dict[int, ResourceBid],
    top_n: int = 5,
    weights: Optional[Dict[str, float]] = None,
) -> AuctionResult:
    """
    Verify all reveals, score winning bids, return top-N clients.

    score = w_b * bandwidth + w_c * compute + w_s * storage
    Default weights treat all three resources equally (1/3 each).
    """
    if weights is None:
        weights = {"bandwidth": 1 / 3, "compute": 1 / 3, "storage": 1 / 3}

    leaderboard: List[tuple] = []
    rejected = 0
    for cid, com in commitments.items():
        if cid not in revealed_bids:
            rejected += 1
            continue
        ok, _ = verify_bid(com, revealed_bids[cid])
        if not ok:
            rejected += 1
            continue
        bid = revealed_bids[cid]
        score = (
            weights["bandwidth"] * bid.bandwidth_mbps
            + weights["compute"] * bid.compute_gflops
            + weights["storage"] * bid.storage_mb / 1000.0  # storage in GB equiv
        )
        leaderboard.append((cid, float(score)))

    leaderboard.sort(key=lambda x: -x[1])
    winners = [cid for cid, _ in leaderboard[:top_n]]
    winning_score = leaderboard[0][1] if leaderboard else 0.0
    return AuctionResult(
        accepted_client_ids=winners,
        winning_score=winning_score,
        rejected_count=rejected,
        leaderboard=leaderboard,
    )
