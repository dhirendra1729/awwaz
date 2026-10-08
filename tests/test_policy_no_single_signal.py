"""
Safety Invariant Test: No Single Signal Family Can Escalate Beyond Observation.
Reference: Section 10.2 of AWWAZ Master Blueprint.
"""

import pytest
try:
    from awwaz.policy.decision_engine import ZeroTrustPolicyEngine, EvaluatedSignal, SignalFamily
except ImportError:
    from src.awwaz.policy.decision_engine import ZeroTrustPolicyEngine, EvaluatedSignal, SignalFamily


def test_single_acoustic_signal_cannot_escalate():
    """
    CRITICAL SAFETY INVARIANT:
    Even with an extreme 0.999 spoof probability on a 100,000 INR transfer,
    a single acoustic detector alone CANNOT trigger friction, blocks, or challenges.
    """
    engine = ZeroTrustPolicyEngine(high_impact_amount_threshold=5000.0)

    extreme_acoustic_signal = EvaluatedSignal(
        family=SignalFamily.ACOUSTIC,
        name="VOCODER_PHASE_ANOMALY_HIGH",
        calibrated_score=0.999,
        operating_threshold=0.80,
        is_triggered=True,
    )

    decision = engine.evaluate([extreme_acoustic_signal], transfer_amount=100000.0)

    # Invariant: Must remain Level 1 observation only
    assert decision.action_level == 1
    assert decision.action_name == "SILENT_OBSERVE_LOG"
    assert decision.requires_out_of_band is False
    assert decision.is_reversible is True


def test_corroborated_signals_trigger_verification():
    """
    When >= 2 independent families fire (Acoustic + Payment Behavior) on a high-value transfer,
    the system safely escalates to Level 3 Out-of-Band verification.
    """
    engine = ZeroTrustPolicyEngine(high_impact_amount_threshold=5000.0)

    acoustic_signal = EvaluatedSignal(
        family=SignalFamily.ACOUSTIC,
        name="SYNTHETIC_SPEECH_DETECTED",
        calibrated_score=0.88,
        operating_threshold=0.80,
        is_triggered=True,
    )
    behavior_signal = EvaluatedSignal(
        family=SignalFamily.PAYMENT_BEHAVIOR,
        name="NOVEL_BENEFICIARY_SPIKE",
        calibrated_score=0.92,
        operating_threshold=0.75,
        is_triggered=True,
    )

    decision = engine.evaluate([acoustic_signal, behavior_signal], transfer_amount=25000.0)

    assert decision.action_level == 3
    assert decision.action_name == "OUT_OF_BAND_STEP_UP_CHALLENGE"
    assert decision.is_reversible is True
    assert decision.requires_out_of_band is True


def test_low_impact_corroborated_triggers_friction_nudge():
    """
    When >= 2 families fire on a low-impact transfer (< 5,000 INR),
    Level 2 Cognitive Friction Nudge is engaged without out-of-band blockage.
    """
    engine = ZeroTrustPolicyEngine(high_impact_amount_threshold=5000.0)

    acoustic_signal = EvaluatedSignal(
        family=SignalFamily.ACOUSTIC,
        name="SYNTHETIC_SPEECH_DETECTED",
        calibrated_score=0.85,
        operating_threshold=0.80,
        is_triggered=True,
    )
    caller_signal = EvaluatedSignal(
        family=SignalFamily.CALLER_CONTEXT,
        name="UNKNOWN_CALLER_NOT_IN_CONTACTS",
        calibrated_score=0.75,
        operating_threshold=0.50,
        is_triggered=True,
    )

    decision = engine.evaluate([acoustic_signal, caller_signal], transfer_amount=1500.0)

    assert decision.action_level == 2
    assert decision.action_name == "COGNITIVE_FRICTION_NUDGE"
    assert decision.requires_out_of_band is False
    assert decision.is_reversible is True
