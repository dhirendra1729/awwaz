"""
Caller Intelligence and Telephony Signaling Context.
Analyzes CLI spoof likelihood, carrier lookup, and telecom signaling anomalies.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TelephonyContext:
    calling_number: str
    is_in_contacts: bool
    carrier_circle: str            # e.g., "UP_EAST", "DELHI_NCR", "MUMBAI"
    is_voip_origin: bool           # Virtual SIP / Twilio trunk injection
    signaling_cli_spoofed: bool
    call_duration_seconds: float
    is_call_active_during_upi: bool


@dataclass(frozen=True)
class CallerProfile:
    risk_score: float             # 0.0 to 1.0
    is_unknown_caller: bool
    is_suspicious_origin: bool
    anomaly_flags: list[str]


class CallerIntelEngine:
    """
    Evaluates telephony metadata to establish caller credibility and flag spoofing.
    """
    def __init__(self):
        # Known Indian telco circle prefixes and routing patterns
        self.valid_indian_mobile_prefixes = {"6", "7", "8", "9"}

    def analyze(self, context: TelephonyContext) -> CallerProfile:
        flags = []
        score = 0.0

        # 1. Contact list check
        is_unknown = not context.is_in_contacts
        if is_unknown:
            flags.append("CALLER_NOT_IN_SAVED_CONTACTS")
            score += 0.25

        # 2. Number normalization & spoofing checks
        clean_num = context.calling_number.replace("+91", "").replace("-", "").strip()
        if len(clean_num) == 10:
            if clean_num[0] not in self.valid_indian_mobile_prefixes:
                flags.append("INVALID_INDIAN_MOBILE_RANGE")
                score += 0.35
        elif len(clean_num) > 10:
            flags.append("INTERNATIONAL_ROUTED_INBOUND")
            score += 0.30

        # 3. Direct telecom signaling spoof flag
        if context.signaling_cli_spoofed:
            flags.append("TELECOM_CLI_SPOOF_HEADER_MISMATCH")
            score += 0.45

        # 4. VoIP / Virtual trunk injection
        if context.is_voip_origin:
            flags.append("VIRTUAL_SIP_TRUNK_INJECTION")
            score += 0.30

        # 5. In-session concurrency: call active while opening payment
        if context.is_call_active_during_upi:
            flags.append("CALL_ACTIVE_DURING_UPI_SESSION")
            score += 0.20

        final_score = min(1.0, score)
        return CallerProfile(
            risk_score=round(final_score, 3),
            is_unknown_caller=is_unknown,
            is_suspicious_origin=context.is_voip_origin or context.signaling_cli_spoofed,
            anomaly_flags=flags,
        )
