"""
AdaptiveWeightOptimizer — Adaptive Weights Module
===================================================
Dynamically adjusts α, β, γ weights based on the signal reliability
observed in the current inference pass.

This is your "Adaptive Hybrid" contribution — stronger than fixed weights.

Two strategies are implemented:
  1. signal_variance : Down-weights signals that are at extremes (near 0 or 1)
                       since they may be unreliable for that specific claim.
  2. calibrated      : Uses pre-calibrated reliability profiles per signal type.
"""


class AdaptiveWeightOptimizer:
    """
    Adjusts α, β, γ adaptively based on per-sample signal behavior.

    Instead of fixed weights, this optimizer observes the three scores
    and redistributes weight toward the more reliable signals for that
    particular claim.

    Parameters
    ----------
    base_alpha : float
        Starting weight for retrieval similarity.
    base_beta : float
        Starting weight for semantic similarity.
    base_gamma : float
        Starting weight for confidence.
    strategy : str
        'signal_variance' | 'calibrated' | 'entropy_reweight'
    min_weight : float
        Minimum weight any signal can receive (prevents zeroing out).
    """

    def __init__(
        self,
        base_alpha: float = 0.4,
        base_beta: float = 0.4,
        base_gamma: float = 0.2,
        strategy: str = "signal_variance",
        min_weight: float = 0.05,
    ):
        self.base_alpha = base_alpha
        self.base_beta = base_beta
        self.base_gamma = base_gamma
        self.strategy = strategy
        self.min_weight = min_weight

    def adjust(
        self,
        retrieval_score: float,
        semantic_score: float,
        confidence_score: float
    ) -> tuple[float, float, float]:
        """
        Return adjusted (alpha, beta, gamma) weights summing to 1.0.

        Parameters
        ----------
        retrieval_score, semantic_score, confidence_score : float
            The three individual scores for the current claim.

        Returns
        -------
        tuple[float, float, float]
            Adjusted (alpha, beta, gamma).
        """
        if self.strategy == "signal_variance":
            return self._signal_variance_adjust(
                retrieval_score, semantic_score, confidence_score
            )
        elif self.strategy == "calibrated":
            return self._calibrated_adjust(
                retrieval_score, semantic_score, confidence_score
            )
        elif self.strategy == "entropy_reweight":
            return self._entropy_reweight(
                retrieval_score, semantic_score, confidence_score
            )
        else:
            return self.base_alpha, self.base_beta, self.base_gamma

    def _signal_variance_adjust(self, r, s, c) -> tuple[float, float, float]:
        """
        A signal near 0.5 is most informative (uncertain, could go either way).
        A signal near 0 or 1 is less discriminative — down-weight it.

        Reliability = 1 - |score - 0.5| * 2 (peaked at 0.5, falls to 0 at extremes)
        """
        def reliability(score):
            # Peak at 0.5 → 1.0; at 0 or 1 → 0.0
            return 1.0 - abs(score - 0.5) * 2.0

        r_rel = reliability(r) * self.base_alpha
        s_rel = reliability(s) * self.base_beta
        c_rel = reliability(c) * self.base_gamma

        total = r_rel + s_rel + c_rel
        if total < 1e-6:
            return self.base_alpha, self.base_beta, self.base_gamma

        alpha = max(self.min_weight, r_rel / total)
        beta  = max(self.min_weight, s_rel / total)
        gamma = max(self.min_weight, c_rel / total)

        # Re-normalize after min_weight clamping
        total2 = alpha + beta + gamma
        return alpha / total2, beta / total2, gamma / total2

    def _calibrated_adjust(self, r, s, c) -> tuple[float, float, float]:
        """
        Rule-based adjustment based on signal agreement/disagreement.

        If retrieval score is near 0 (Wikipedia found nothing useful),
        up-weight semantic + confidence. If confidence is very high,
        up-weight it further.
        """
        alpha = self.base_alpha
        beta = self.base_beta
        gamma = self.base_gamma

        # If retrieval is near 0 (no Wikipedia support), penalize it
        if r < 0.15:
            alpha *= 0.4
            beta  *= 1.3
            gamma *= 1.3

        # If confidence is very high, reward it
        if c > 0.75:
            gamma *= 1.5
            alpha *= 0.85

        # If semantic is very low despite retrieval finding passages, trust retrieval
        if s < 0.15 and r > 0.4:
            alpha *= 1.3
            beta  *= 0.6

        # Normalize
        total = alpha + beta + gamma
        alpha, beta, gamma = alpha/total, beta/total, gamma/total

        # Enforce min weights
        alpha = max(self.min_weight, alpha)
        beta  = max(self.min_weight, beta)
        gamma = max(self.min_weight, gamma)

        total = alpha + beta + gamma
        return alpha/total, beta/total, gamma/total

    def _entropy_reweight(self, r, s, c) -> tuple[float, float, float]:
        """
        Treat the three scores as a distribution and use cross-signal
        entropy to determine which is most informative.

        Scores that are close to each other → high agreement → use base weights.
        Scores that diverge → up-weight the majority signal.
        """
        scores = [r, s, c]
        base_weights = [self.base_alpha, self.base_beta, self.base_gamma]

        mean = sum(scores) / 3.0
        deviations = [abs(sc - mean) for sc in scores]

        # Signals close to mean → reliable (well-aligned), up-weight slightly
        # Signals far from mean → outlier, down-weight
        max_dev = max(deviations) if max(deviations) > 0 else 1.0
        alignment_weights = [1.0 - (d / max_dev) * 0.5 for d in deviations]

        adjusted = [bw * aw for bw, aw in zip(base_weights, alignment_weights)]
        total = sum(adjusted)

        alpha = max(self.min_weight, adjusted[0] / total)
        beta  = max(self.min_weight, adjusted[1] / total)
        gamma = max(self.min_weight, adjusted[2] / total)

        total2 = alpha + beta + gamma
        return alpha/total2, beta/total2, gamma/total2

    def calibrate_on_validation(
        self,
        validation_data: list[dict],
        detector,
        n_trials: int = 50
    ) -> tuple[float, float, float]:
        """
        Grid search over (α, β, γ) on a validation set to find best weights.

        Parameters
        ----------
        validation_data : list of dict
            Each dict: {"claim": ..., "question": ..., "label": "FACTUAL"|"HALLUCINATION"}
        detector : HybridHallucinationDetector
            Detector instance to use for scoring.
        n_trials : int
            Number of weight combinations to try.

        Returns
        -------
        tuple[float, float, float]
            Best (alpha, beta, gamma).
        """
        import random

        best_acc = 0.0
        best_weights = (self.base_alpha, self.base_beta, self.base_gamma)

        label_map = {"FACTUAL": 1, "UNCERTAIN": 0, "HALLUCINATION": -1}

        for _ in range(n_trials):
            # Random weights summing to 1
            a = random.uniform(0.1, 0.8)
            b = random.uniform(0.1, 0.8 - a)
            g = 1.0 - a - b

            if g < 0.05:
                continue

            # Temporarily override weights
            old = (detector.alpha, detector.beta, detector.gamma)
            detector.alpha, detector.beta, detector.gamma = a, b, g

            correct = 0
            for item in validation_data:
                result = detector.detect(
                    claim=item["claim"],
                    question=item.get("question")
                )
                pred = 1 if result.verdict == "FACTUAL" else -1
                true = 1 if item["label"] == "FACTUAL" else -1
                if pred == true:
                    correct += 1

            acc = correct / len(validation_data)
            if acc > best_acc:
                best_acc = acc
                best_weights = (a, b, g)

            detector.alpha, detector.beta, detector.gamma = old

        print(f"   🎯 Best validation accuracy: {best_acc:.1%} with α={best_weights[0]:.2f}, β={best_weights[1]:.2f}, γ={best_weights[2]:.2f}")
        return best_weights
