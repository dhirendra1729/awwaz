"""
AWWAZ (SAAKSHI) FastAPI Application Entrypoint.
Provides REST and WebSocket interfaces for banking and consumer edge integration.
"""

from datetime import datetime, timezone
import hashlib
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..policy.decision_engine import (
    ZeroTrustPolicyEngine,
    EvaluatedSignal,
    SignalFamily,
    PolicyDecision,
)
from ..policy.action_ladder import GraduatedActionLadder, ActionLevel
from ..context.caller_intel import CallerIntelEngine, TelephonyContext
from ..context.payment_profiler import PaymentProfiler, PaymentIntent
from ..context.prior_lifter import BayesianPriorLifter
from ..verify.family_beacon import (
    FamilyLifelineBeaconEngine,
    FamilyResponseStatus,
    FamilyBeaconResolution,
)
from ..verify.fido_push import FidoStrongBoxPush, HardwareChallenge
from ..verify.tokens import Ed25519TokenSigner
from ..prove.hash_chain import HashChainLedger
from ..prove.i4c_dossier import I4CIncidentDossierGenerator, CybercrimeIncidentReport
from .websocket_stream import router as ws_router

app = FastAPI(
    title="AWWAZ (SAAKSHI) Voice Firewall API",
    description="The Zero-Trust Voice Firewall for Financial Systems & UPI Protection",
    version="0.1.0",
)

app.include_router(ws_router)

# Engine Singletons
policy_engine = ZeroTrustPolicyEngine(high_impact_amount_threshold=5000.0)
action_ladder = GraduatedActionLadder()
caller_intel = CallerIntelEngine()
payment_profiler = PaymentProfiler()
prior_lifter = BayesianPriorLifter()
beacon_engine = FamilyLifelineBeaconEngine()
fido_engine = FidoStrongBoxPush()
token_signer = Ed25519TokenSigner()
ledger = HashChainLedger()
dossier_gen = I4CIncidentDossierGenerator()


# Request / Response Schemas
class EvaluateRequest(BaseModel):
    session_id: str
    payer_vpa: str
    payee_vpa: str
    amount_inr: float
    historical_avg_amount_inr: float = 1000.0
    is_new_beneficiary: bool = False
    calling_number: str
    is_in_contacts: bool = True
    carrier_circle: str = "DELHI_NCR"
    is_voip: bool = False
    is_cli_spoofed: bool = False
    is_call_active_during_upi: bool = False
    calibrated_p_spoof: float = Field(0.0, ge=0.0, le=1.0)
    operating_threshold: float = 0.80


class DispatchBeaconRequest(BaseModel):
    target_user_id: str
    relative_phone: str
    caller_identity: str
    amount_inr: float


class ResolveBeaconRequest(BaseModel):
    beacon_id: str
    status: str  # "i_am_safe_scam", "yes_real_emergency", "timeout"


class ExportDossierRequest(BaseModel):
    session_id: str
    suspect_vpa: str
    calling_number: str
    amount_inr: float
    calibrated_p_spoof: float
    partial_spoof_intervals: List[str] = []


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "awwaz-saakshi",
        "zero_trust_invariants_active": True,
        "dpdp_ephemeral_enforced": True,
    }


@app.post("/evaluate")
def evaluate_transaction(req: EvaluateRequest):
    # 1. Evaluate context
    tel_ctx = TelephonyContext(
        calling_number=req.calling_number,
        is_in_contacts=req.is_in_contacts,
        carrier_circle=req.carrier_circle,
        is_voip_origin=req.is_voip,
        signaling_cli_spoofed=req.is_cli_spoofed,
        call_duration_seconds=45.0,
        is_call_active_during_upi=req.is_call_active_during_upi,
    )
    caller_profile = caller_intel.analyze(tel_ctx)

    pay_intent = PaymentIntent(
        payer_vpa=req.payer_vpa,
        payee_vpa=req.payee_vpa,
        amount_inr=req.amount_inr,
        historical_avg_amount_inr=req.historical_avg_amount_inr,
        is_new_beneficiary=req.is_new_beneficiary,
    )
    beneficiary_risk = payment_profiler.evaluate(pay_intent)

    # 2. Bayesian Prior-Lifting
    prior_ctx = prior_lifter.evaluate_gating(
        caller=caller_profile,
        payment=beneficiary_risk,
        is_call_active_during_upi=req.is_call_active_during_upi,
    )

    # 3. Build signals across families
    signals: List[EvaluatedSignal] = []

    # Acoustic family
    is_acoustic_triggered = req.calibrated_p_spoof >= req.operating_threshold
    signals.append(
        EvaluatedSignal(
            family=SignalFamily.ACOUSTIC,
            name="ACOUSTIC_VOCODER_ANOMALY",
            calibrated_score=req.calibrated_p_spoof,
            operating_threshold=req.operating_threshold,
            is_triggered=is_acoustic_triggered,
        )
    )

    # Caller Context family
    is_caller_triggered = caller_profile.risk_score >= 0.50
    signals.append(
        EvaluatedSignal(
            family=SignalFamily.CALLER_CONTEXT,
            name="CALLER_SUSPICIOUS_CLI_OR_UNKNOWN",
            calibrated_score=caller_profile.risk_score,
            operating_threshold=0.50,
            is_triggered=is_caller_triggered,
        )
    )

    # Payment Behavior family
    is_payment_triggered = beneficiary_risk.risk_score >= 0.50
    signals.append(
        EvaluatedSignal(
            family=SignalFamily.PAYMENT_BEHAVIOR,
            name="PAYMENT_ANOMALY_OR_NOVEL_VPA",
            calibrated_score=beneficiary_risk.risk_score,
            operating_threshold=0.50,
            is_triggered=is_payment_triggered,
        )
    )

    # Session Telephony family
    signals.append(
        EvaluatedSignal(
            family=SignalFamily.SESSION_TELEPHONY,
            name="CALL_CONCURRENT_WITH_PAYMENT",
            calibrated_score=1.0 if req.is_call_active_during_upi else 0.0,
            operating_threshold=0.50,
            is_triggered=req.is_call_active_during_upi,
        )
    )

    # 4. Zero-Trust Policy Decision
    decision = policy_engine.evaluate(signals, transfer_amount=req.amount_inr)

    # 5. Append to hash chain audit ledger
    acoustic_bytes = f"spoof:{req.calibrated_p_spoof}".encode("utf-8")
    block = ledger.append_event(
        session_id=req.session_id,
        caller_number=req.calling_number,
        vpa=req.payee_vpa,
        acoustic_vector_bytes=acoustic_bytes,
        action_level=decision.action_level,
        reason_codes=decision.reason_codes,
    )

    return {
        "decision": {
            "action_level": decision.action_level,
            "action_name": decision.action_name,
            "is_reversible": decision.is_reversible,
            "requires_out_of_band": decision.requires_out_of_band,
            "reason_codes": decision.reason_codes,
        },
        "bayesian_prior_context": {
            "is_gated": prior_ctx.is_gated,
            "lifted_precision": prior_ctx.lifted_precision,
            "precision_multiplier": prior_ctx.precision_multiplier,
            "criteria_met": prior_ctx.gating_criteria_met,
        },
        "ledger_block_index": block.index,
        "ledger_block_hash": block.block_hash,
    }


@app.post("/verify/beacon/dispatch")
def dispatch_family_beacon(req: DispatchBeaconRequest):
    beacon = beacon_engine.dispatch_beacon(
        target_user_id=req.target_user_id,
        relative_phone=req.relative_phone,
        caller_identity=req.caller_identity,
        suspected_amount=req.amount_inr,
    )
    return {
        "beacon_id": beacon.beacon_id,
        "ttl_seconds": beacon.ttl_seconds,
        "status": "DISPATCHED_TO_TRUSTED_CONTACT",
    }


@app.post("/verify/beacon/resolve")
def resolve_family_beacon(req: ResolveBeaconRequest):
    status_enum = FamilyResponseStatus(req.status)
    resolution = beacon_engine.resolve_beacon(req.beacon_id, status_enum)
    return {
        "beacon_id": resolution.beacon_id,
        "status": resolution.status.value,
        "action_directive": resolution.action_directive,
        "screen_banner_text": resolution.victim_screen_banner_text,
        "should_accelerate_funds": resolution.should_accelerate_funds,
        "should_block_imposter": resolution.should_block_imposter,
    }


@app.post("/prove/dossier")
def export_dossier(req: ExportDossierRequest):
    report = CybercrimeIncidentReport(
        incident_id=f"INC-{req.session_id[:8].upper()}",
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        victim_vpa_hash=hashlib.sha256("victim@upi".encode("utf-8")).hexdigest(),
        suspect_vpa=req.suspect_vpa,
        attempted_amount_inr=req.amount_inr,
        calling_number=req.calling_number,
        telecom_circle="UP_EAST",
        cli_spoofed=True,
        ai_model_version="AWWAZ-AcousticNet-v1.0",
        calibrated_p_spoof=req.calibrated_p_spoof,
        partial_spoof_timestamps=req.partial_spoof_intervals,
        acoustic_evidence_hash=hashlib.sha256(f"p:{req.calibrated_p_spoof}".encode("utf-8")).hexdigest(),
        ledger_block_hash=ledger.chain[-1].block_hash if ledger.chain else HashChainLedger.GENESIS_HASH,
    )
    xml_content = dossier_gen.generate_xml_dossier(report)
    return {
        "incident_id": report.incident_id,
        "xml_dossier": xml_content,
        "dpdp_compliant": True,
    }
