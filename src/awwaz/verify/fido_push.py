"""
Method V1: Hardware-Bound FIDO2 / StrongBox Push Verification.
Generates and verifies cryptographic challenges bound to the victim's hardware secure element.
"""

from dataclasses import dataclass
import hashlib
import os
import time
from typing import Optional
from cryptography.hazmat.primitives.asymmetric import ed25519


@dataclass(frozen=True)
class HardwareChallenge:
    challenge_id: str
    nonce: str
    timestamp_epoch: int
    amount_inr: float
    payee_vpa_hash: str
    ttl_seconds: int = 60


@dataclass(frozen=True)
class HardwareAssertion:
    challenge_id: str
    signature_bytes: bytes
    authenticator_data: bytes
    client_data_json: str


class FidoStrongBoxPush:
    """
    Mints out-of-band hardware push challenges and verifies Android StrongBox / iOS Secure Enclave assertions.
    """
    def __init__(self, default_ttl_seconds: int = 60):
        self.default_ttl = default_ttl_seconds
        self._active_challenges = {}

    def mint_challenge(self, amount_inr: float, payee_vpa: str) -> HardwareChallenge:
        """Create a cryptographic challenge for mobile hardware authentication."""
        nonce = os.urandom(32).hex()
        vpa_hash = hashlib.sha256(payee_vpa.encode("utf-8")).hexdigest()
        challenge_id = hashlib.sha256(f"{nonce}:{time.time()}".encode("utf-8")).hexdigest()[:16]

        challenge = HardwareChallenge(
            challenge_id=challenge_id,
            nonce=nonce,
            timestamp_epoch=int(time.time()),
            amount_inr=amount_inr,
            payee_vpa_hash=vpa_hash,
            ttl_seconds=self.default_ttl,
        )
        self._active_challenges[challenge_id] = challenge
        return challenge

    def verify_assertion(
        self,
        assertion: HardwareAssertion,
        public_key: ed25519.Ed25519PublicKey,
    ) -> bool:
        """
        Verify the biometric hardware signature against the challenge.
        """
        challenge = self._active_challenges.get(assertion.challenge_id)
        if not challenge:
            return False

        # Verify TTL
        if (time.time() - challenge.timestamp_epoch) > challenge.ttl_seconds:
            return False

        message = f"{challenge.challenge_id}:{challenge.nonce}:{challenge.amount_inr}:{challenge.payee_vpa_hash}".encode("utf-8")
        try:
            public_key.verify(assertion.signature_bytes, message)
            del self._active_challenges[assertion.challenge_id]
            return True
        except Exception:
            return False
