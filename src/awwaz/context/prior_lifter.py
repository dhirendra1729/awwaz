"""
Contextual Bayesian Prior-Lifter.
Solves the Base-Rate Paradox in Voice Cloning Fraud Detection (Section 2 of Master Blueprint).

In unconditional call streams:
    Base prevalence pi = 10^-5 -> Precision = 0.08% (1,250 false alarms per true attack).

When gated on transaction & telephony context C:
    Gated prevalence pi' = 10^-2 -> Precision = 44.69% (>560x precision gain without model retraining).
"""

from dataclasses import dataclass
from typing import Tuple
from .caller_intel import CallerProfile
from .payment_profiler import BeneficiaryRisk


@dataclass(frozen=True)
class GatedPriorContext:
    is_gated: bool
    effective_prior_pi: float
    unconditioned_precision: float
    lifted_precision: float
    precision_multiplier: float
    gating_criteria_met: list[str]


class BayesianPriorLifter:
    """
    Computes Bayesian prior-lifting and calibrated posterior probabilities.
    """
    def __init__(
        self,
        base_prevalence_pi: float = 1e-5,    # 1 in 100,000 calls
        gated_prevalence_pi: float = 0.01,   # 1% under multi-family gating
        default_tpr: float = 0.80,           # 80% True Positive Rate
        default_fpr: float = 0.01,           # 1.0% False Positive Rate
    ):
        self.base_prevalence_pi = base_prevalence_pi
        self.gated_prevalence_pi = gated_prevalence_pi
        self.default_tpr = default_tpr
        self.default_fpr = default_fpr

    def compute_precision(self, prevalence: float, tpr: float, fpr: float) -> float:
        """
        Bayes theorem:
        Precision = (pi * TPR) / (pi * TPR + (1 - pi) * FPR)
        """
        numerator = prevalence * tpr
        denominator = numerator + (1.0 - prevalence) * fpr
        if denominator <= 0:
            return 0.0
        return float(numerator / denominator)

    def evaluate_gating(
        self,
        caller: CallerProfile,
        payment: BeneficiaryRisk,
        is_call_active_during_upi: bool,
        tpr: float = 0.80,
        fpr: float = 0.01,
    ) -> GatedPriorContext:
        """
        Evaluates the Gating Condition C:
        C = {Unknown Caller} ∩ {New Beneficiary VPA} ∩ {Amount > 3x Profile} ∩ {Call Active During UPI}
        """
        criteria = []
        if caller.is_unknown_caller:
            criteria.append("UNKNOWN_CALLER")
        if payment.is_new_payee:
            criteria.append("NEW_BENEFICIARY_VPA")
        if payment.is_amount_anomalous:
            criteria.append("AMOUNT_GT_3X_PROFILE")
        if is_call_active_during_upi:
            criteria.append("CALL_ACTIVE_DURING_UPI")

        # Prior is lifted if at least 3 out of 4 gating conditions fire, or all 4 for full lift
        is_fully_gated = len(criteria) >= 3

        effective_prior = self.gated_prevalence_pi if is_fully_gated else self.base_prevalence_pi

        unconditioned_prec = self.compute_precision(self.base_prevalence_pi, tpr, fpr)
        lifted_prec = self.compute_precision(effective_prior, tpr, fpr)
        multiplier = lifted_prec / max(1e-9, unconditioned_prec)

        return GatedPriorContext(
            is_gated=is_fully_gated,
            effective_prior_pi=effective_prior,
            unconditioned_precision=round(unconditioned_prec, 6),
            lifted_precision=round(lifted_prec, 4),
            precision_multiplier=round(multiplier, 1),
            gating_criteria_met=criteria,
        )
