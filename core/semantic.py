"""
SemanticSimilarityScorer — Semantic Module
==========================================
Uses Sentence-BERT (SBERT) to compute cosine similarity between
the claim and retrieved Wikipedia context.

Falls back to a TF-IDF cosine if sentence-transformers isn't installed.
"""

import math
import re
from collections import Counter


class SemanticSimilarityScorer:
    """
    Scores semantic similarity between a claim and a reference context.

    Tries to use SentenceTransformers (SBERT) for dense embeddings.
    Falls back to sparse TF-IDF cosine similarity if unavailable.

    Parameters
    ----------
    model_name : str
        SBERT model to use. 'all-MiniLM-L6-v2' is fast and accurate.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._mode = "tfidf"  # default fallback

        self._try_load_sbert()

    def _try_load_sbert(self):
        try:
            from sentence_transformers import SentenceTransformer
            print(f"   📦 Loading SBERT model: {self.model_name} ...")
            self.model = SentenceTransformer(self.model_name)
            self._mode = "sbert"
            print(f"   ✅ SBERT loaded ({self.model_name})")
        except ImportError:
            print("   ⚠️  sentence-transformers not found. Using TF-IDF cosine fallback.")
            print("       Install with: pip install sentence-transformers")
            self._mode = "tfidf"
        except Exception as e:
            print(f"   ⚠️  SBERT load failed ({e}). Using TF-IDF fallback.")
            self._mode = "tfidf"

    def score(self, claim: str, context: str) -> float:
        """
        Compute similarity between claim and context.

        Parameters
        ----------
        claim : str
            The generated answer to evaluate.
        context : str
            Reference text (Wikipedia passages concatenated).

        Returns
        -------
        float
            Similarity score in [0.0, 1.0].
        """
        if self._mode == "sbert":
            return self._sbert_score(claim, context)
        else:
            return self._tfidf_cosine(claim, context)

    def _sbert_score(self, claim: str, context: str) -> float:
        """
        Dense cosine similarity via SBERT embeddings.
        Handles long contexts by chunking and taking max similarity.
        """
        try:
            # Chunk context into ~500 char segments for better matching
            chunks = self._chunk_text(context, max_chars=500)
            if not chunks:
                chunks = [context]

            claim_emb = self.model.encode([claim], convert_to_tensor=True)
            ctx_embs = self.model.encode(chunks, convert_to_tensor=True)

            from sentence_transformers import util
            scores = util.cos_sim(claim_emb, ctx_embs)[0]

            # Use max similarity across chunks (best-matching chunk)
            best_score = float(scores.max().item())

            # Cosine can be slightly negative; clamp to [0, 1]
            return max(0.0, min(1.0, best_score))

        except Exception as e:
            print(f"   ⚠️  SBERT scoring error: {e}. Falling back to TF-IDF.")
            return self._tfidf_cosine(claim, context)

    def _tfidf_cosine(self, text1: str, text2: str) -> float:
        """
        Sparse TF-IDF cosine similarity — no models required.
        Computes cosine between TF-IDF vectors of the two texts.
        """
        tokens1 = self._tokenize(text1)
        tokens2 = self._tokenize(text2)

        if not tokens1 or not tokens2:
            return 0.0

        # Build a mini corpus for IDF
        vocab = set(tokens1) | set(tokens2)
        N = 2  # two documents

        def df(tok):
            return (tok in set(tokens1)) + (tok in set(tokens2))

        idf = {tok: math.log((N + 1) / (df(tok) + 1)) + 1.0 for tok in vocab}

        def tfidf_vec(tokens):
            tf = Counter(tokens)
            total = len(tokens)
            return {tok: (tf[tok] / total) * idf[tok] for tok in tokens}

        vec1 = tfidf_vec(tokens1)
        vec2 = tfidf_vec(tokens2)

        # Cosine similarity
        dot = sum(vec1.get(tok, 0) * vec2.get(tok, 0) for tok in vocab)
        norm1 = math.sqrt(sum(v**2 for v in vec1.values()))
        norm2 = math.sqrt(sum(v**2 for v in vec2.values()))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return max(0.0, min(1.0, dot / (norm1 * norm2)))

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        STOPWORDS = {
            "the", "a", "an", "is", "was", "are", "were", "to", "of",
            "in", "for", "on", "with", "at", "by", "from", "that", "this",
            "and", "or", "but", "not", "it", "its", "as", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "which", "who", "what",
            "when", "where", "how", "he", "she", "they", "we", "i", "you"
        }
        tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
        return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

    @staticmethod
    def _chunk_text(text: str, max_chars: int = 500) -> list[str]:
        """Split text into sentence-aware chunks of roughly max_chars."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks, current = [], ""
        for sent in sentences:
            if len(current) + len(sent) <= max_chars:
                current += " " + sent
            else:
                if current:
                    chunks.append(current.strip())
                current = sent
        if current:
            chunks.append(current.strip())
        return chunks if chunks else [text[:max_chars]]
