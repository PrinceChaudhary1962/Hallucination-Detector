"""
Benchmark Evaluation Utility
==============================
Runs the detector on a built-in test set and reports:
  - Accuracy, Precision, Recall, F1
  - Per-class breakdown
  - Comparison with baselines (Retrieval-only, Semantic-only, Confidence-only)

Run with: python main.py --benchmark
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.detector import HybridHallucinationDetector


# ── Built-in test set ─────────────────────────────────────────────────────────
# 20 examples: 10 factual, 10 hallucinated
# Sources: TruthfulQA-inspired + common misconceptions

TEST_SET = [
    # ── FACTUAL ──────────────────────────────────────────────────────────
    {
        "question": "Who developed the theory of general relativity?",
        "claim": "Albert Einstein developed the general theory of relativity, published in 1915.",
        "label": "FACTUAL"
    },
    {
        "question": "What is the chemical formula for water?",
        "claim": "The chemical formula for water is H2O, consisting of two hydrogen atoms and one oxygen atom.",
        "label": "FACTUAL"
    },
    {
        "question": "What is the capital of France?",
        "claim": "Paris is the capital city of France.",
        "label": "FACTUAL"
    },
    {
        "question": "When did World War II end?",
        "claim": "World War II ended in 1945 with the surrender of Germany in May and Japan in September.",
        "label": "FACTUAL"
    },
    {
        "question": "What is the speed of light?",
        "claim": "The speed of light in a vacuum is approximately 299,792 kilometers per second.",
        "label": "FACTUAL"
    },
    {
        "question": "Who painted the Mona Lisa?",
        "claim": "The Mona Lisa was painted by Leonardo da Vinci, completed around 1503-1519.",
        "label": "FACTUAL"
    },
    {
        "question": "What is the largest planet in our solar system?",
        "claim": "Jupiter is the largest planet in our solar system.",
        "label": "FACTUAL"
    },
    {
        "question": "What year did the Berlin Wall fall?",
        "claim": "The Berlin Wall fell in 1989, marking a significant moment in the end of the Cold War.",
        "label": "FACTUAL"
    },
    {
        "question": "What is DNA?",
        "claim": "DNA (deoxyribonucleic acid) is a molecule that carries genetic information in living organisms.",
        "label": "FACTUAL"
    },
    {
        "question": "Who was the first person to walk on the moon?",
        "claim": "Neil Armstrong was the first person to walk on the moon during the Apollo 11 mission in 1969.",
        "label": "FACTUAL"
    },

    # ── HALLUCINATIONS ────────────────────────────────────────────────────
    {
        "question": "What is the capital of Australia?",
        "claim": "Sydney is the capital city of Australia.",
        "label": "HALLUCINATION"  # Canberra is the capital
    },
    {
        "question": "Who invented the telephone?",
        "claim": "The telephone was invented by Nikola Tesla in 1876.",
        "label": "HALLUCINATION"  # Alexander Graham Bell
    },
    {
        "question": "How many bones does an adult human have?",
        "claim": "An adult human body has 215 bones.",
        "label": "HALLUCINATION"  # 206 bones
    },
    {
        "question": "Who wrote Hamlet?",
        "claim": "Hamlet was written by Christopher Marlowe in the 16th century.",
        "label": "HALLUCINATION"  # William Shakespeare
    },
    {
        "question": "What is the longest river in the world?",
        "claim": "The Amazon River is the longest river in the world.",
        "label": "HALLUCINATION"  # The Nile (or Amazon, disputed — but commonly Nile)
    },
    {
        "question": "What planet is closest to the sun?",
        "claim": "Venus is the planet closest to the sun in our solar system.",
        "label": "HALLUCINATION"  # Mercury
    },
    {
        "question": "Who was the first President of the United States?",
        "claim": "Benjamin Franklin was the first President of the United States.",
        "label": "HALLUCINATION"  # George Washington
    },
    {
        "question": "What is the chemical symbol for gold?",
        "claim": "The chemical symbol for gold is Gd.",
        "label": "HALLUCINATION"  # Au
    },
    {
        "question": "How many continents are there on Earth?",
        "claim": "There are 8 continents on Earth.",
        "label": "HALLUCINATION"  # 7
    },
    {
        "question": "What is the hardest natural substance?",
        "claim": "The hardest natural substance on Earth is quartz.",
        "label": "HALLUCINATION"  # Diamond
    },
]


def run_benchmark(detector: "HybridHallucinationDetector"):
    """Run benchmark on the built-in test set and print results."""

    print("\n" + "="*70)
    print("  BENCHMARK MODE")
    print("="*70)
    print(f"\n  Running on {len(TEST_SET)} test cases...")
    print("  (10 factual + 10 hallucinated)\n")

    import time
    results = []
    for i, item in enumerate(TEST_SET, 1):
        print(f"  [{i:2d}/{len(TEST_SET)}] {item['claim'][:60]}...")
        result = detector.detect(claim=item["claim"], question=item["question"])
        results.append({
            "result": result,
            "true_label": item["label"]
        })
        time.sleep(2.0)

    # ── Compute metrics ───────────────────────────────────────────────────
    hybrid_metrics = _compute_metrics(results, "HYBRID")

    # ── Baselines ─────────────────────────────────────────────────────────
    # Retrieval only
    retrieval_only = [
        {"result": _mock_result(r["result"], "retrieval"), "true_label": r["true_label"]}
        for r in results
    ]
    ret_metrics = _compute_metrics(retrieval_only, "RETRIEVAL ONLY")

    # Semantic only
    semantic_only = [
        {"result": _mock_result(r["result"], "semantic"), "true_label": r["true_label"]}
        for r in results
    ]
    sem_metrics = _compute_metrics(semantic_only, "SEMANTIC ONLY")

    # Confidence only
    conf_only = [
        {"result": _mock_result(r["result"], "confidence"), "true_label": r["true_label"]}
        for r in results
    ]
    conf_metrics = _compute_metrics(conf_only, "CONFIDENCE ONLY")

    # ── Print comparison table ─────────────────────────────────────────────
    print("\n" + "="*70)
    print("  RESULTS COMPARISON")
    print("="*70)
    print(f"\n  {'Method':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("  " + "-"*65)

    for label, metrics in [
        ("Retrieval Only", ret_metrics),
        ("Semantic Only", sem_metrics),
        ("Confidence Only", conf_metrics),
        ("YOUR HYBRID MODEL", hybrid_metrics),
    ]:
        highlight = " ◄" if label == "YOUR HYBRID MODEL" else ""
        print(
            f"  {label:<25} "
            f"{metrics['accuracy']:>9.1%} "
            f"{metrics['precision']:>10.1%} "
            f"{metrics['recall']:>10.1%} "
            f"{metrics['f1']:>9.3f}"
            f"{highlight}"
        )

    print("\n" + "="*70)
    print("  DETAILED RESULTS")
    print("="*70)

    for i, item in enumerate(results, 1):
        r = item["result"]
        true = item["true_label"]
        pred = r.verdict
        pred_factual = pred == "FACTUAL"
        true_factual = true == "FACTUAL"
        if pred == "UNCERTAIN":
            match = "~"  # uncertain = partial credit
        elif pred_factual == true_factual:
            match = "✓"
        else:
            match = "✗"
        print(
            f"  [{i:2d}] {match} "
            f"True: {true:<15} "
            f"Pred: {pred:<15} "
            f"Score: {r.final_score:.3f} "
            f"({r.verdict_emoji})"
        )

    print("\n" + "="*70)
    print("  Benchmark complete.")
    print("="*70 + "\n")


def _compute_metrics(items: list, name: str) -> dict:
    """Compute accuracy, precision, recall, F1 for binary (FACTUAL vs HALLUCINATION)."""
    tp = fp = tn = fn = 0

    for item in items:
        pred_factual = item["result"].verdict in ("FACTUAL", "UNCERTAIN")
        true_factual = item["true_label"] == "FACTUAL"

        if pred_factual and true_factual:
            tp += 1
        elif pred_factual and not true_factual:
            fp += 1
        elif not pred_factual and true_factual:
            fn += 1
        else:
            tn += 1

    n = len(items)
    accuracy  = (tp + tn) / n if n else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def _mock_result(result, signal: str):
    """Create a mock result using only one signal for baseline comparison."""
    from dataclasses import replace

    score_map = {
        "retrieval":  result.retrieval_score,
        "semantic":   result.semantic_score,
        "confidence": result.confidence_score,
    }
    score = score_map[signal]

    # Use same thresholds as detector
    if score < 0.42:
        verdict, emoji = "HALLUCINATION", "🔴"
    elif score < 0.58:
        verdict, emoji = "UNCERTAIN", "🟡"
    else:
        verdict, emoji = "FACTUAL", "🟢"

    from core.detector import DetectionResult
    return DetectionResult(
        claim=result.claim,
        question=result.question,
        retrieval_score=result.retrieval_score,
        semantic_score=result.semantic_score,
        confidence_score=result.confidence_score,
        final_score=score,
        alpha=1.0 if signal == "retrieval" else 0.0,
        beta=1.0 if signal == "semantic" else 0.0,
        gamma=1.0 if signal == "confidence" else 0.0,
        verdict=verdict,
        verdict_emoji=emoji,
    )
