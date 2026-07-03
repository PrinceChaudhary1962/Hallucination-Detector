"""
ConfidenceScorer — Confidence Module
======================================
Estimates model confidence for a claim using token-level log-probabilities
and entropy from a HuggingFace seq2seq or causal LM.

Higher confidence → higher score (model is more certain about the claim).
Lower confidence / higher entropy → lower score (model is uncertain).
"""

import math
from typing import Optional


class ConfidenceScorer:
    """
    Scores the linguistic confidence of a claim using a language model.

    Supported modes:
      - 'logprob'  : Average log-probability of tokens in the claim
      - 'entropy'  : Entropy-based uncertainty (inverted for scoring)
      - 'perplexity': Perplexity-based scoring (inverted)

    Falls back to a heuristic lexical confidence scorer if no model available.

    Parameters
    ----------
    model_name : str
        HuggingFace model name. For seq2seq: 'google/flan-t5-base'.
        For causal LM: 'gpt2', 'microsoft/phi-2', etc.
    scoring_mode : str
        'entropy' | 'logprob' | 'perplexity'
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        scoring_mode: str = "entropy"
    ):
        self.model_name = model_name
        self.scoring_mode = scoring_mode
        self.model = None
        self.tokenizer = None
        self._model_type = None
        self._mode = "heuristic"  # fallback

        self._try_load_model()

    def _try_load_model(self):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

            print(f"   📦 Loading confidence model: {self.model_name} ...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Try seq2seq first (T5, BART, etc.)
            try:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
                self._model_type = "seq2seq"
            except Exception:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
                self._model_type = "causal"

            self.model.eval()
            self._mode = "model"
            self._torch = torch
            print(f"   ✅ Confidence model loaded ({self._model_type})")

        except ImportError:
            print("   ⚠️  transformers/torch not found. Using heuristic confidence scorer.")
            print("       Install with: pip install transformers torch")
            self._mode = "heuristic"
        except Exception as e:
            print(f"   ⚠️  Model load failed ({e}). Using heuristic confidence scorer.")
            self._mode = "heuristic"

    def score(self, claim: str) -> float:
        """
        Compute a confidence score for the claim.

        Parameters
        ----------
        claim : str
            The text to evaluate.

        Returns
        -------
        float
            Score in [0.0, 1.0]. Higher = more confident / less hallucination-prone.
        """
        if self._mode == "model":
            if self.scoring_mode == "entropy":
                return self._entropy_score(claim)
            elif self.scoring_mode == "logprob":
                return self._logprob_score(claim)
            else:
                return self._perplexity_score(claim)
        else:
            return self._heuristic_score(claim)

    # ──────────────────────────────────────────────────────────────────────
    # Model-based scoring
    # ──────────────────────────────────────────────────────────────────────

    def _entropy_score(self, claim: str) -> float:
        """
        Low entropy = model is confident about each token = higher score.
        Entropy is averaged across all token positions in the claim.
        """
        try:
            import torch
            import torch.nn.functional as F

            tokens = self.tokenizer(
                claim,
                return_tensors="pt",
                truncation=True,
                max_length=256
            )
            input_ids = tokens["input_ids"]

            with torch.no_grad():
                if self._model_type == "seq2seq":
                    # For seq2seq: use the claim as both encoder input and decoder target
                    outputs = self.model(
                        input_ids=input_ids,
                        labels=input_ids
                    )
                    logits = outputs.logits
                else:
                    outputs = self.model(input_ids=input_ids)
                    logits = outputs.logits

            # Compute token-level entropy
            probs = F.softmax(logits, dim=-1)  # [1, seq_len, vocab_size]
            log_probs = torch.log(probs + 1e-10)
            entropy = -(probs * log_probs).sum(dim=-1)  # [1, seq_len]
            mean_entropy = entropy.mean().item()

            # Max theoretical entropy = log(vocab_size)
            vocab_size = logits.shape[-1]
            max_entropy = math.log(vocab_size)

            # Normalize: low entropy → high confidence score
            normalized_entropy = mean_entropy / max_entropy
            confidence = 1.0 - normalized_entropy

            return max(0.0, min(1.0, confidence))

        except Exception as e:
            print(f"   ⚠️  Entropy scoring error: {e}")
            return self._heuristic_score(claim)

    def _logprob_score(self, claim: str) -> float:
        """
        Average log-probability of claim tokens.
        Higher avg log-prob → model assigns high probability to the text.
        """
        try:
            import torch
            import torch.nn.functional as F

            tokens = self.tokenizer(
                claim,
                return_tensors="pt",
                truncation=True,
                max_length=256
            )
            input_ids = tokens["input_ids"]

            with torch.no_grad():
                if self._model_type == "seq2seq":
                    outputs = self.model(input_ids=input_ids, labels=input_ids)
                    logits = outputs.logits
                else:
                    outputs = self.model(input_ids=input_ids)
                    logits = outputs.logits[:, :-1, :]
                    input_ids = input_ids[:, 1:]

            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(
                2, input_ids.unsqueeze(-1)
            ).squeeze(-1)

            avg_log_prob = token_log_probs.mean().item()  # typically < 0

            # Map avg log-prob to [0, 1]
            # Typical range: -15 (very uncertain) to 0 (very certain)
            score = 1.0 / (1.0 + math.exp(-avg_log_prob - 2.0))  # sigmoid shift
            return max(0.0, min(1.0, score))

        except Exception as e:
            print(f"   ⚠️  Log-prob scoring error: {e}")
            return self._heuristic_score(claim)

    def _perplexity_score(self, claim: str) -> float:
        """
        Inverted perplexity score. Low PPL → high confidence.
        """
        try:
            import torch

            tokens = self.tokenizer(
                claim,
                return_tensors="pt",
                truncation=True,
                max_length=256
            )
            input_ids = tokens["input_ids"]

            with torch.no_grad():
                if self._model_type == "seq2seq":
                    outputs = self.model(input_ids=input_ids, labels=input_ids)
                    loss = outputs.loss.item()
                else:
                    outputs = self.model(input_ids=input_ids, labels=input_ids)
                    loss = outputs.loss.item()

            ppl = math.exp(loss)

            # Map PPL to [0, 1]: PPL=1 → 1.0, PPL=100+ → ~0
            score = 1.0 / (1.0 + math.log(max(ppl, 1.0)))
            return max(0.0, min(1.0, score))

        except Exception as e:
            print(f"   ⚠️  Perplexity scoring error: {e}")
            return self._heuristic_score(claim)

    # ──────────────────────────────────────────────────────────────────────
    # Heuristic fallback (no model required)
    # ──────────────────────────────────────────────────────────────────────

    def _heuristic_score(self, claim: str) -> float:
        """
        Training-free heuristic confidence proxy based on:
          - Presence of uncertainty hedge words (lowers score)
          - Sentence specificity: numbers, named entities (raises score)
          - Length penalty for very short/long claims

        This is a rough approximation — use a real model for best results.
        """
        score = 0.68 # baseline

        # Uncertainty markers lower confidence
        uncertainty_phrases = [
            "i think", "i believe", "maybe", "perhaps", "possibly",
            "not sure", "might be", "could be", "it seems", "apparently",
            "i'm not certain", "approximately", "around", "roughly",
            "it's possible", "it may", "unclear"
        ]
        lower_claim = claim.lower()
        hedge_count = sum(1 for phrase in uncertainty_phrases if phrase in lower_claim)
        score -= hedge_count * 0.08

        # Specific numerical claims suggest factual grounding
        import re
        numbers = re.findall(r'\b\d{3,}\b', claim)       # only 3+ digit numbers
        years   = re.findall(r'\b(1[0-9]{3}|20[0-9]{2})\b', claim)
        score += min(len(numbers) * 0.06, 0.12)
        score += min(len(years)   * 0.04, 0.08)

        # Named entities (capitalized words not at sentence start)
        words = claim.split()
        mid_caps = sum(1 for w in words[1:] if w and w[0].isupper() and w.isalpha())
        score += min(mid_caps * 0.04, 0.12)

        # Length: very short or very long claims are often less reliable
        word_count = len(words)
        if word_count < 5:
            score -= 0.10
        elif word_count > 80:
            score -= 0.08

        # Definitive language
        definitive_words = ["is", "are", "was", "were", "the", "in", "on", "at"]
        definitive_count = sum(1 for w in lower_claim.split() if w in definitive_words)
        score += min(definitive_count * 0.01, 0.05)

        return max(0.0, min(1.0, score))
