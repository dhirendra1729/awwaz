"""
Payment Intent Profiler and Beneficiary Risk Engine.
Evaluates UPI VPA novelty and amount anomalies against historical user behavior.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class PaymentIntent:
    payer_vpa: str
    payee_vpa: str
    amount_inr: float
    historical_avg_amount_inr: float
    is_new_beneficiary: bool
    payee_account_age_days: Optional[int] = None
    transaction_note: Optional[str] = None


@dataclass(frozen=True)
class BeneficiaryRisk:
    risk_score: float             # 0.0 to 1.0
    is_amount_anomalous: bool     # Amount > 3x historical average
    is_new_payee: bool
    amount_ratio: float
    risk_flags: List[str]


class PaymentProfiler:
    """
    Profiles payment transactions to detect coercive distress transfers.
    APP fraud typically exhibits sudden out-of-pattern transfers to newly created VPAs.
    """
    def __init__(self, anomaly_ratio_threshold: float = 3.0):
        self.anomaly_ratio_threshold = anomaly_ratio_threshold

    def evaluate(self, intent: PaymentIntent) -> BeneficiaryRisk:
        flags = []
        score = 0.0

        avg = max(100.0, intent.historical_avg_amount_inr)
        ratio = intent.amount_inr / avg
        is_amount_anomalous = ratio >= self.anomaly_ratio_threshold

        # 1. New Beneficiary VPA
        if intent.is_new_beneficiary:
            flags.append("NOVEL_UNSEEN_BENEFICIARY_VPA")
            score += 0.35

        # 2. Significant amount spike
        if is_amount_anomalous:
            flags.append(f"TRANSACTION_AMOUNT_SPIKE_{round(ratio, 1)}X_PROFILE")
            score += 0.40

        # 3. Freshly minted payee VPA (mule accounts are often < 7 days old)
        if intent.payee_account_age_days is not None and intent.payee_account_age_days <= 7:
            flags.append("FRESH_BENEFICIARY_ACCOUNT_AGE_LE_7D")
            score += 0.25

        final_score = min(1.0, score)
        return BeneficiaryRisk(
            risk_score=round(final_score, 3),
            is_amount_anomalous=is_amount_anomalous,
            is_new_payee=intent.is_new_beneficiary,
            amount_ratio=round(ratio, 2),
            risk_flags=flags,
        )
