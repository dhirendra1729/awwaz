"""
The 7-Level Graduated Action Ladder State Machine (STAGE 3: ACT).
Reference: Section 7 of Master Blueprint.

Levels:
0: PASS_THROUGH
1: SILENT_TELEMETRY_LOG
2: COGNITIVE_FRICTION_NUDGE
3: OUT_OF_BAND_STEP_UP
4: REVERSIBLE_COOLING_OFF
5: TIER_2_SOC_HUMAN_REVIEW
6: DECLINE_AND_FREEZE (HUMAN DECISION ONLY - never by AI alone)
"""

from dataclasses import dataclass
from enum import IntEnum
import time
from typing import Dict, List, Optional


class ActionLevel(IntEnum):
    PASS_THROUGH = 0
    SILENT_TELEMETRY_LOG = 1
    COGNITIVE_FRICTION_NUDGE = 2
    OUT_OF_BAND_STEP_UP = 3
    REVERSIBLE_COOLING_OFF = 4
    TIER_2_SOC_HUMAN_REVIEW = 5
    DECLINE_AND_FREEZE = 6


@dataclass(frozen=True)
class ActionStep:
    level: ActionLevel
    name: str
    is_reversible: bool
    requires_human_decision_only: bool
    description: str


class GraduatedActionLadder:
    """
    Manages state transitions across the 7 action levels.
    Enforces that Level 6 (Decline & Freeze) cannot be reached via automated AI scoring.
    """
    STEPS: Dict[ActionLevel, ActionStep] = {
        ActionLevel.PASS_THROUGH: ActionStep(
            ActionLevel.PASS_THROUGH,
            "PASS_THROUGH",
            is_reversible=True,
            requires_human_decision_only=False,
            description="All signals clean. Instant pass-through authorization.",
        ),
        ActionLevel.SILENT_TELEMETRY_LOG: ActionStep(
            ActionLevel.SILENT_TELEMETRY_LOG,
            "SILENT_TELEMETRY_LOG",
            is_reversible=True,
            requires_human_decision_only=False,
            description="Single anomaly observed. Observed and logged silently.",
        ),
        ActionLevel.COGNITIVE_FRICTION_NUDGE: ActionStep(
            ActionLevel.COGNITIVE_FRICTION_NUDGE,
            "COGNITIVE_FRICTION_NUDGE",
            is_reversible=True,
            requires_human_decision_only=False,
            description="2 families fired on low-value transfer. Non-blocking advisory notice.",
        ),
        ActionLevel.OUT_OF_BAND_STEP_UP: ActionStep(
            ActionLevel.OUT_OF_BAND_STEP_UP,
            "OUT_OF_BAND_STEP_UP",
            is_reversible=True,
            requires_human_decision_only=False,
            description="High impact + corroborated signals. Deterministic out-of-band verification required.",
        ),
        ActionLevel.REVERSIBLE_COOLING_OFF: ActionStep(
            ActionLevel.REVERSIBLE_COOLING_OFF,
            "REVERSIBLE_COOLING_OFF",
            is_reversible=True,
            requires_human_decision_only=False,
            description="Verification timed out. 30-minute cooling-off window with release option.",
        ),
        ActionLevel.TIER_2_SOC_HUMAN_REVIEW: ActionStep(
            ActionLevel.TIER_2_SOC_HUMAN_REVIEW,
            "TIER_2_SOC_HUMAN_REVIEW",
            is_reversible=True,
            requires_human_decision_only=False,
            description="Verification denied. Escalated to Tier-2 Fraud SOC human queue with SLA.",
        ),
        ActionLevel.DECLINE_AND_FREEZE: ActionStep(
            ActionLevel.DECLINE_AND_FREEZE,
            "DECLINE_AND_FREEZE",
            is_reversible=False,
            requires_human_decision_only=True,
            description="Permanent transaction decline. Exclusively authorized by human analyst.",
        ),
    }

    def __init__(self, cooling_off_ttl_seconds: int = 1800):
        self.cooling_off_ttl = cooling_off_ttl_seconds

    def handle_verification_outcome(
        self,
        current_level: ActionLevel,
        outcome: str,  # "VERIFIED", "DENIED", "TIMEOUT", "EMERGENCY_CONFIRMED"
        is_analyst_authorized: bool = False,
    ) -> ActionLevel:
        """
        Transition between levels based on deterministic verification or human review.
        """
        if outcome == "EMERGENCY_CONFIRMED":
            # Emergency Rescue Path: Immediate release
            return ActionLevel.PASS_THROUGH

        if outcome == "VERIFIED":
            return ActionLevel.PASS_THROUGH

        if outcome == "TIMEOUT":
            return ActionLevel.REVERSIBLE_COOLING_OFF

        if outcome == "DENIED":
            return ActionLevel.TIER_2_SOC_HUMAN_REVIEW

        if outcome == "ANALYST_DECLINE":
            if not is_analyst_authorized:
                raise PermissionError("Level 6 DECLINE_AND_FREEZE strictly requires human analyst authorization.")
            return ActionLevel.DECLINE_AND_FREEZE

        return current_level
