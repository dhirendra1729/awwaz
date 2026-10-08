"""
Method V2: Out-of-Band Family Life-Line Beacon (The Raksham Circle Protocol).
Implements the multi-device out-of-band verification protocol and Emergency Rescue State Machine.
"""

from dataclasses import dataclass
from enum import Enum
import hashlib
import os
import time
from typing import Dict, Optional


class FamilyResponseStatus(str, Enum):
    PENDING = "pending"
    I_AM_SAFE_SCAM = "i_am_safe_scam"                   # Son confirms safe -> Scammer exposed
    YES_REAL_EMERGENCY = "yes_real_emergency"           # Genuine emergency -> Emergency Rescue Path
    TIMEOUT = "timeout"                                 # Unanswered -> Reversible cooling-off hold


@dataclass(frozen=True)
class FamilyBeaconRequest:
    beacon_id: str
    target_user_id: str
    relative_phone_number: str
    alleged_caller_identity: str
    suspected_amount_inr: float
    created_at_epoch: float
    ttl_seconds: int = 45


@dataclass(frozen=True)
class FamilyBeaconResolution:
    beacon_id: str
    status: FamilyResponseStatus
    action_directive: str
    victim_screen_banner_text: str
    should_accelerate_funds: bool
    should_block_imposter: bool


class FamilyLifelineBeaconEngine:
    """
    Manages outbound out-of-band family pings and processes incoming lifeline acknowledgments.
    """
    def __init__(self, default_ttl_seconds: int = 45):
        self.default_ttl = default_ttl_seconds
        self._active_beacons: Dict[str, FamilyBeaconRequest] = {}

    def dispatch_beacon(
        self,
        target_user_id: str,
        relative_phone: str,
        caller_identity: str,
        suspected_amount: float,
    ) -> FamilyBeaconRequest:
        """Dispatches an out-of-band high-priority push/SMS beacon to the enrolled family member."""
        beacon_id = hashlib.sha256(f"{target_user_id}:{relative_phone}:{time.time()}".encode("utf-8")).hexdigest()[:12]
        request = FamilyBeaconRequest(
            beacon_id=beacon_id,
            target_user_id=target_user_id,
            relative_phone_number=relative_phone,
            alleged_caller_identity=caller_identity,
            suspected_amount_inr=suspected_amount,
            created_at_epoch=time.time(),
            ttl_seconds=self.default_ttl,
        )
        self._active_beacons[beacon_id] = request
        return request

    def resolve_beacon(
        self,
        beacon_id: str,
        response: FamilyResponseStatus,
    ) -> FamilyBeaconResolution:
        """
        Processes response from family member and yields actionable directive.
        """
        beacon = self._active_beacons.get(beacon_id)
        if not beacon:
            return FamilyBeaconResolution(
                beacon_id=beacon_id,
                status=FamilyResponseStatus.TIMEOUT,
                action_directive="EXPIRED_OR_NOT_FOUND",
                victim_screen_banner_text="Verification session expired.",
                should_accelerate_funds=False,
                should_block_imposter=False,
            )

        # Check TTL
        if (time.time() - beacon.created_at_epoch) > beacon.ttl_seconds and response == FamilyResponseStatus.PENDING:
            response = FamilyResponseStatus.TIMEOUT

        if response == FamilyResponseStatus.I_AM_SAFE_SCAM:
            resolution = FamilyBeaconResolution(
                beacon_id=beacon_id,
                status=response,
                action_directive="CONFIRMED_VOICE_CLONING_SCAM",
                victim_screen_banner_text=(
                    f"🚨 SAAKSHI VERIFIED: {beacon.alleged_caller_identity} CONFIRMED SAFE. "
                    "THIS IS A DEEPFAKE SCAM CALL. HANG UP IMMEDIATELY!"
                ),
                should_accelerate_funds=False,
                should_block_imposter=True,
            )
        elif response == FamilyResponseStatus.YES_REAL_EMERGENCY:
            # SECTION 7.2: THE EMERGENCY RESCUE PATH
            resolution = FamilyBeaconResolution(
                beacon_id=beacon_id,
                status=response,
                action_directive="FLIP_TO_ASSISTED_EMERGENCY_RELEASE",
                victim_screen_banner_text=(
                    f"🟢 EMERGENCY CONFIRMED BY {beacon.alleged_caller_identity}. "
                    "TRANSFER APPROVED & EXPEDITED. EMERGENCY SERVICES CONCIERGE ATTACHED."
                ),
                should_accelerate_funds=True,
                should_block_imposter=False,
            )
        else:
            resolution = FamilyBeaconResolution(
                beacon_id=beacon_id,
                status=FamilyResponseStatus.TIMEOUT,
                action_directive="ENGAGE_REVERSIBLE_COOLING_OFF",
                victim_screen_banner_text=(
                    "Family verification unconfirmed. Temporary 30-minute cooling-off interlock active for your protection."
                ),
                should_accelerate_funds=False,
                should_block_imposter=False,
            )

        if beacon_id in self._active_beacons:
            del self._active_beacons[beacon_id]
        return resolution
