"""
SHA-256 Tamper-Evident Hash Chain Audit Ledger.
Provides an immutable cryptographic trail of fraud signals, decisions, and challenges.
"""

from dataclasses import dataclass, asdict
import hashlib
import json
import time
from typing import Dict, List, Optional


@dataclass(frozen=True)
class LedgerBlock:
    index: int
    timestamp_epoch: float
    session_id: str
    caller_hash: str
    vpa_hash: str
    acoustic_feature_hash: str
    action_level: int
    reason_codes: List[str]
    previous_hash: str
    block_hash: str


class HashChainLedger:
    """
    Append-only SHA-256 audit log. Any alteration of historical records invalidates
    all subsequent block hashes in the chain.
    """
    GENESIS_HASH = "0" * 64

    def __init__(self):
        self.chain: List[LedgerBlock] = []

    def append_event(
        self,
        session_id: str,
        caller_number: str,
        vpa: str,
        acoustic_vector_bytes: bytes,
        action_level: int,
        reason_codes: List[str],
    ) -> LedgerBlock:
        prev_hash = self.chain[-1].block_hash if self.chain else self.GENESIS_HASH
        idx = len(self.chain)

        caller_hash = hashlib.sha256(caller_number.encode("utf-8")).hexdigest()
        vpa_hash = hashlib.sha256(vpa.encode("utf-8")).hexdigest()
        acoustic_hash = hashlib.sha256(acoustic_vector_bytes).hexdigest()
        ts = time.time()

        content = f"{idx}:{ts}:{session_id}:{caller_hash}:{vpa_hash}:{acoustic_hash}:{action_level}:{','.join(sorted(reason_codes))}:{prev_hash}"
        block_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        block = LedgerBlock(
            index=idx,
            timestamp_epoch=ts,
            session_id=session_id,
            caller_hash=caller_hash,
            vpa_hash=vpa_hash,
            acoustic_feature_hash=acoustic_hash,
            action_level=action_level,
            reason_codes=reason_codes,
            previous_hash=prev_hash,
            block_hash=block_hash,
        )
        self.chain.append(block)
        return block

    def verify_integrity(self) -> bool:
        """Verify the full cryptographic integrity of the ledger chain."""
        for i, block in enumerate(self.chain):
            expected_prev = self.chain[i - 1].block_hash if i > 0 else self.GENESIS_HASH
            if block.previous_hash != expected_prev:
                return False

            content = (
                f"{block.index}:{block.timestamp_epoch}:{block.session_id}:"
                f"{block.caller_hash}:{block.vpa_hash}:{block.acoustic_feature_hash}:"
                f"{block.action_level}:{','.join(sorted(block.reason_codes))}:{block.previous_hash}"
            )
            recalculated = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if block.block_hash != recalculated:
                return False

        return True
