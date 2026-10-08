"""
Safety Invariant Test: The Emergency Rescue Path (Section 7.2 of Master Blueprint).
Guarantees that genuine life-threatening emergencies are never blocked by false positives.
"""

import pytest

try:
    from awwaz.verify.family_beacon import (
        FamilyLifelineBeaconEngine,
        FamilyResponseStatus,
    )
    from awwaz.policy.action_ladder import GraduatedActionLadder, ActionLevel
    from awwaz.verify.tokens import Ed25519TokenSigner
except ImportError:
    from src.awwaz.verify.family_beacon import (
        FamilyLifelineBeaconEngine,
        FamilyResponseStatus,
    )
    from src.awwaz.policy.action_ladder import GraduatedActionLadder, ActionLevel
    from src.awwaz.verify.tokens import Ed25519TokenSigner


def test_real_emergency_confirmation_releases_and_accelerates():
    """
    CRITICAL HUMAN SAFETY INVARIANT:
    If a son or family member actually suffered an accident and confirms 'YES_REAL_EMERGENCY',
    the system MUST NOT freeze money. It must accelerate release and offer emergency concierge.
    """
    engine = FamilyLifelineBeaconEngine(default_ttl_seconds=45)
    ladder = GraduatedActionLadder()

    beacon = engine.dispatch_beacon(
        target_user_id="mother_user_123",
        relative_phone="+919876543210",
        caller_identity="Son (Rohan)",
        suspected_amount=50000.0,
    )

    # Relative taps 'YES, THIS IS ME - REAL EMERGENCY'
    resolution = engine.resolve_beacon(beacon.beacon_id, FamilyResponseStatus.YES_REAL_EMERGENCY)

    assert resolution.action_directive == "FLIP_TO_ASSISTED_EMERGENCY_RELEASE"
    assert resolution.should_accelerate_funds is True
    assert resolution.should_block_imposter is False
    assert "EMERGENCY CONFIRMED" in resolution.victim_screen_banner_text

    # Action ladder flips hold directly to Level 0 pass-through release
    new_level = ladder.handle_verification_outcome(
        current_level=ActionLevel.OUT_OF_BAND_STEP_UP,
        outcome="EMERGENCY_CONFIRMED",
    )
    assert new_level == ActionLevel.PASS_THROUGH


def test_scam_exposed_when_relative_confirms_safe():
    """
    When relative confirms 'I AM 100% SAFE', cognitive capture is shattered
    and the imposter transfer is blocked.
    """
    engine = FamilyLifelineBeaconEngine(default_ttl_seconds=45)

    beacon = engine.dispatch_beacon(
        target_user_id="mother_user_123",
        relative_phone="+919876543210",
        caller_identity="Son (Rohan)",
        suspected_amount=50000.0,
    )

    resolution = engine.resolve_beacon(beacon.beacon_id, FamilyResponseStatus.I_AM_SAFE_SCAM)

    assert resolution.action_directive == "CONFIRMED_VOICE_CLONING_SCAM"
    assert resolution.should_accelerate_funds is False
    assert resolution.should_block_imposter is True
    assert "CONFIRMED SAFE" in resolution.victim_screen_banner_text


def test_ed25519_signed_token_minted_on_release():
    """Verified payments produce valid cryptographically signed Ed25519 release tokens."""
    signer = Ed25519TokenSigner()

    token = signer.mint_token(
        token_id="tok_rel_999",
        payer_vpa="mother@upi",
        payee_vpa="hospital@upi",
        amount_inr=50000.0,
        verification_method="METHOD_V2_FAMILY_BEACON",
        validity_seconds=300,
    )

    assert signer.verify_token(token) is True
    assert token.amount_inr == 50000.0
