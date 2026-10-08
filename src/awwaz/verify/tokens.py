"""
Ed25519 Cryptographic Verification Tokens.
Generates and verifies tamper-proof authorization tokens minted upon out-of-band resolution.
"""

from dataclasses import dataclass
import json
import time
from typing import Optional
from cryptography.hazmat.primitives.asymmetric import ed25519


@dataclass(frozen=True)
class AuthorizationToken:
    token_id: str
    payer_vpa: str
    payee_vpa: str
    amount_inr: float
    verification_method: str     # e.g., "METHOD_V1_FIDO2_STRONGBOX", "METHOD_V2_FAMILY_BEACON"
    issued_at_epoch: int
    expires_at_epoch: int
    signature_hex: str
    public_key_hex: str


class Ed25519TokenSigner:
    """
    Signs and validates tamper-evident authorization release tokens for payment execution.
    """
    def __init__(self, private_key: Optional[ed25519.Ed25519PrivateKey] = None):
        self.private_key = private_key or ed25519.Ed25519PrivateKey.generate()
        self.public_key = self.private_key.public_key()

    def mint_token(
        self,
        token_id: str,
        payer_vpa: str,
        payee_vpa: str,
        amount_inr: float,
        verification_method: str,
        validity_seconds: int = 300,
    ) -> AuthorizationToken:
        now = int(time.time())
        expires = now + validity_seconds

        payload = {
            "token_id": token_id,
            "payer_vpa": payer_vpa,
            "payee_vpa": payee_vpa,
            "amount_inr": amount_inr,
            "verification_method": verification_method,
            "issued_at": now,
            "expires_at": expires,
        }
        canonical_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
        signature = self.private_key.sign(canonical_bytes)

        return AuthorizationToken(
            token_id=token_id,
            payer_vpa=payer_vpa,
            payee_vpa=payee_vpa,
            amount_inr=amount_inr,
            verification_method=verification_method,
            issued_at_epoch=now,
            expires_at_epoch=expires,
            signature_hex=signature.hex(),
            public_key_hex=self.public_key.public_bytes_raw().hex(),
        )

    def verify_token(self, token: AuthorizationToken) -> bool:
        if time.time() > token.expires_at_epoch:
            return False

        payload = {
            "token_id": token.token_id,
            "payer_vpa": token.payer_vpa,
            "payee_vpa": token.payee_vpa,
            "amount_inr": token.amount_inr,
            "verification_method": token.verification_method,
            "issued_at": token.issued_at_epoch,
            "expires_at": token.expires_at_epoch,
        }
        canonical_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
        try:
            pub_key = ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(token.public_key_hex))
            pub_key.verify(bytes.fromhex(token.signature_hex), canonical_bytes)
            return True
        except Exception:
            return False
