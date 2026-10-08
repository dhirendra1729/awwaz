"""
Zero-Trust Policy Engine (STAGE 2: VERIFY -> STAGE 3: ACT).
Reference: Section 10.2 of AWWAZ Master Blueprint.

CRITICAL SAFETY INVARIANT:
No single signal family—at any score or confidence—is permitted to trigger
friction, holds, or blocks. Escalation requires >= 2 independent families
to fire simultaneously.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Set


class SignalFamily(str, Enum):
    ACOUSTIC = "acoustic"
    CALLER_CONTEXT = "caller_context"
    PAYMENT_BEHAVIOR = "payment_behavior"
    SESSION_TELEPHONY = "session_telephony"


@dataclass(frozen=True)
class EvaluatedSignal:
    family: SignalFamily
    name: str
    calibrated_score: float  # 0.0 to 1.0
    operating_threshold: float
    is_triggered: bool


@dataclass(frozen=True)
class PolicyDecision:
    action_level: int         # 0 (Pass) to 6 (Decline)
    action_name: str
    is_reversible: bool
    requires_out_of_band: bool
    reason_codes: List[str]


class ZeroTrustPolicyEngine:
    """
    CRITICAL SAFETY INVARIANT:
    No single signal family—at any score or confidence—is permitted
    to trigger friction, holds, or blocks. Escalation requires >= 2
    independent families to fire simultaneously.
    """
    def __init__(self, high_impact_amount_threshold: float = 5000.0):
        self.amount_threshold = high_impact_amount_threshold

    def evaluate(self, signals: List[EvaluatedSignal], transfer_amount: float) -> PolicyDecision:
        triggered_families: Set[SignalFamily] = {s.family for s in signals if s.is_triggered}
        reason_codes = [s.name for s in signals if s.is_triggered]

        # Invariant 1: If 0 or 1 family fired, NO user-visible friction allowed
        if len(triggered_families) == 0:
            return PolicyDecision(
                action_level=0,
                action_name="PASS_THROUGH",
                is_reversible=True,
                requires_out_of_band=False,
                reason_codes=["ALL_SIGNALS_NORMAL"],
            )

        if len(triggered_families) == 1:
            # Observation telemetry only; zero user disruption
            return PolicyDecision(
                action_level=1,
                action_name="SILENT_OBSERVE_LOG",
                is_reversible=True,
                requires_out_of_band=False,
                reason_codes=reason_codes,
            )

        # Invariant 2: Two or more independent families fired
        is_high_impact = transfer_amount >= self.amount_threshold

        if not is_high_impact:
            return PolicyDecision(
                action_level=2,
                action_name="COGNITIVE_FRICTION_NUDGE",
                is_reversible=True,
                requires_out_of_band=False,
                reason_codes=reason_codes,
            )

        # High impact + corroborated multi-family evidence -> Out-of-band verification required
        return PolicyDecision(
            action_level=3,
            action_name="OUT_OF_BAND_STEP_UP_CHALLENGE",
            is_reversible=True,
            requires_out_of_band=True,
            reason_codes=reason_codes,
        )
