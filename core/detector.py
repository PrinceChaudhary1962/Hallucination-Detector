"""
HybridHallucinationDetector — Core Module
==========================================
Combines three signals:
  1. Retrieval Similarity  — how well the claim matches Wikipedia knowledge
  2. Semantic Similarity   — SBERT cosine similarity between claim & retrieved context
  3. Confidence Score      — token-level entropy from the generative model

Final Score = α * Retrieval_Sim + β * Semantic_Sim + γ * Confidence
"""

import time
from dataclasses import dataclass, field
from typing import Optional

from core.retrieval import WikipediaRetriever
from core.semantic import SemanticSimilarityScorer
from core.confidence import ConfidenceScorer
from core.adaptive import AdaptiveWeightOptimizer


@dataclass
class DetectionResult:
    """Full result object from a hallucination detection run."""
    claim: str
    question: Optional[str]

    # Individual scores (0.0 → 1.0, higher = more factual)
    retrieval_score: float
    semantic_score: float
    confidence_score: float

    # Hybrid final score
    final_score: float

    # Weights used
    alpha: float
    beta: float
    gamma: float

    # Retrieved evidence
    retrieved_passages: list = field(default_factory=list)
    retrieval_query: str = ""

    # Verdict
    verdict: str = ""          # "FACTUAL" | "UNCERTAIN" | "HALLUCINATION"
    verdict_emoji: str = ""
    processing_time: float = 0.0

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "question": self.question,
            "scores": {
                "retrieval": round(self.retrieval_score, 4),
                "semantic": round(self.semantic_score, 4),
                "confidence": round(self.confidence_score, 4),
                "final": round(self.final_score, 4),
            },
            "weights": {
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma,
            },
            "verdict": self.verdict,
            "retrieved_passages": self.retrieved_passages[:2],  # top 2 for brevity
            "processing_time_sec": round(self.processing_time, 2),
        }


class HybridHallucinationDetector:
    """
    Main hallucination detector combining retrieval, semantic, and confidence signals.

    Parameters
    ----------
    model_name : str
        HuggingFace model for confidence scoring. Needs to support
        token log-probabilities. Recommended: 'google/flan-t5-base'
    alpha : float
        Weight for retrieval similarity score.
    beta : float
        Weight for semantic similarity score.
    gamma : float
        Weight for confidence score.
    adaptive_weights : bool
        If True, use AdaptiveWeightOptimizer to tune α, β, γ on validation data.
    hallucination_threshold : float
        Below this score → HALLUCINATION verdict.
    uncertain_threshold : float
        Between hallucination_threshold and this → UNCERTAIN verdict.
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        alpha: float = 0.5,
        beta: float = 0.4,
        gamma: float = 0.1,
        adaptive_weights: bool = False,
        hallucination_threshold: float = 0.42,
        uncertain_threshold: float = 0.58,
    ):
        self._validate_weights(alpha, beta, gamma)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.adaptive_weights = adaptive_weights
        self.hallucination_threshold = hallucination_threshold
        self.uncertain_threshold = uncertain_threshold

        print("🔧 Initializing Hybrid Hallucination Detector...")

        self.retriever = WikipediaRetriever(top_k=3)
        self.semantic_scorer = SemanticSimilarityScorer()
        self.confidence_scorer = ConfidenceScorer(model_name=model_name)

        if adaptive_weights:
            self.weight_optimizer = AdaptiveWeightOptimizer()
        else:
            self.weight_optimizer = None

        print("✅ Detector ready.\n")

    def detect(self, claim: str, question: Optional[str] = None) -> DetectionResult:
        """
        Run the full hybrid detection pipeline on a claim.

        Parameters
        ----------
        claim : str
            The generated answer / statement to evaluate.
        question : str, optional
            The original question that prompted the claim.
            If provided, used to improve Wikipedia retrieval.

        Returns
        -------
        DetectionResult
        """
        start = time.time()

        query = f"{claim} {question}" if question else claim
        # Auto-expand short claims into full sentences using question context
# This makes the model work correctly regardless of dataset format
        if question and len(claim.split()) <= 5:
            claim = self._expand_claim(claim, question)

        # ── Step 1: Retrieval ──────────────────────────────────────────────
        retrieved = self.retriever.retrieve(query)
        context_text = " ".join(retrieved) if retrieved else ""

        retrieval_score = self.retriever.compute_retrieval_similarity(
            claim=claim,
            passages=retrieved
        ) if retrieved else 0.0

        # ── Step 2: Semantic Similarity ────────────────────────────────────
        if context_text:
            semantic_score = self.semantic_scorer.score(claim, context_text)
        else:
            semantic_score = 0.0
            # 🛡️ Retrieval fallback (important)


        # ── Step 3: Confidence Score ───────────────────────────────────────
        confidence_score = self.confidence_scorer.score(claim)

        # ── Step 4: Adaptive Weights (optional) ───────────────────────────
        alpha, beta, gamma = self.alpha, self.beta, self.gamma
        if self.weight_optimizer:
            alpha, beta, gamma = self.weight_optimizer.adjust(
                retrieval_score, semantic_score, confidence_score
            )

        # ── Step 5: Hybrid Score ───────────────────────────────────────────
        final_score = (
            alpha * retrieval_score +
            beta  * semantic_score +
            gamma * confidence_score
        )
        
        # 🔥 Penalize weak retrieval
        
            # 📈 Boost scientific/common facts
       
        final_score = max(0.0, min(1.0, final_score))  # clamp to [0, 1]

        # ── Step 6: Verdict ────────────────────────────────────────────────
        verdict, emoji = self._get_verdict(final_score)

        elapsed = time.time() - start

        return DetectionResult(
            claim=claim,
            question=question,
            retrieval_score=retrieval_score,
            semantic_score=semantic_score,
            confidence_score=confidence_score,
            final_score=final_score,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            retrieved_passages=retrieved,
            retrieval_query=query,
            verdict=verdict,
            verdict_emoji=emoji,
            processing_time=elapsed,
        )

    def detect_batch(self, items: list[dict]) -> list[DetectionResult]:
        """
        Run detection on a list of {"claim": ..., "question": ...} dicts.
        """
        results = []
        for i, item in enumerate(items, 1):
            print(f"  Processing {i}/{len(items)}...", end="\r")
            r = self.detect(
                claim=item["claim"],
                question=item.get("question")
            )
            results.append(r)
        print()
        return results

    def _get_verdict(self, score: float) -> tuple[str, str]:
        if score < self.hallucination_threshold:
            return "HALLUCINATION", "🔴"
        elif score < self.uncertain_threshold:
            return "UNCERTAIN", "🟡"
        else:
            return "FACTUAL", "🟢"
    def _expand_claim(self, answer: str, question: str) -> str:
        """
        Expand a short answer into a full declarative sentence.
        Works for any dataset that provides question + short answer.
        """
        q = question.strip().rstrip("?").strip()
        a = answer.strip().rstrip(".").strip()

        # Already long enough
        if len(a.split()) > 5:
            return a

        q_lower = q.lower()

        for prefix in [
            "which ", "what is the ", "what was the ", "what are the ",
            "what ", "who was ", "who is ", "who ", "where did ", "where ",
            "when did ", "when was ", "when ", "how old is ", "how old was ",
            "how many ", "how much ", "how ", "in which ",
            "what nationality was ", "what nationality is "
        ]:
            if q_lower.startswith(prefix):
                topic = q[len(prefix):].strip()
                topic = topic[0].upper() + topic[1:] if topic else topic
                return f"{topic} is {a}."

        return f"{a}. {q}."

    @staticmethod
    def _validate_weights(alpha, beta, gamma):
        total = alpha + beta + gamma
        if abs(total - 1.0) > 1e-3:
            raise ValueError(
                f"Weights must sum to 1.0. Got α={alpha}, β={beta}, γ={gamma} → sum={total:.3f}"
            )
        for name, w in [("alpha", alpha), ("beta", beta), ("gamma", gamma)]:
            if not 0.0 <= w <= 1.0:
                raise ValueError(f"Weight '{name}' must be in [0, 1]. Got {w}")
            

