"""
TruthfulQA Evaluation Script
==============================
Full sentence answers — works directly with your detector as-is.
"""

import os
import sys
import time
import pandas as pd

# Allow running as `python scripts/evaluate_truthfulqa.py` from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.detector import HybridHallucinationDetector


def compute_metrics(all_scores: list, score_key: str, threshold: float) -> dict:
    tp = fp = tn = fn = 0
    for entry in all_scores:
        pred_factual = entry[score_key] >= threshold
        true_factual = entry["true_label"] == "FACTUAL"
        if pred_factual and true_factual:         tp += 1
        elif pred_factual and not true_factual:   fp += 1
        elif not pred_factual and true_factual:   fn += 1
        else:                                     tn += 1
    n    = tp + fp + tn + fn
    acc  = (tp + tn) / n         if n          else 0
    prec = tp / (tp + fp)        if (tp + fp)  else 0
    rec  = tp / (tp + fn)        if (tp + fn)  else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def run_truthfulqa(max_samples: int = 30):

    data_path = os.path.join("data", "TruthfulQA.csv")
    if not os.path.exists(data_path):
        print("❌ File not found: data/TruthfulQA.csv")
        print("   Download from: https://github.com/sylinrl/TruthfulQA")
        sys.exit(1)

    print(f"📂 Loading TruthfulQA (max {max_samples} questions)...")
    df = pd.read_csv(data_path).head(max_samples)
    print(f"✅ Loaded {len(df)} questions")
    print(f"⏱  Estimated time: ~{max_samples * 2 * 3}s\n")

    # Show sample
    row = df.iloc[0]
    print("Sample:")
    print(f"  Q       : {row['Question']}")
    print(f"  Correct : {row['Best Answer']}")
    first_wrong = str(row['Incorrect Answers']).split(";")[0].strip()
    print(f"  Wrong   : {first_wrong}")
    print()

    # Initialize detector — same settings as your main benchmark
    detector = HybridHallucinationDetector(
        hallucination_threshold=0.42,
        uncertain_threshold=0.58,
    )

    tp = fp = tn = fn = 0
    all_scores         = []
    errors             = []
    retrieval_failures = 0

    print("=" * 70)
    print(f"  Running on {len(df)} questions ({len(df)*2} predictions)...")
    print("  Wikipedia retrieval enabled")
    print("=" * 70)

    for i, row in df.iterrows():
        question    = str(row.get("Question", "")).strip()
        best_answer = str(row.get("Best Answer", "")).strip()
        incorrect   = str(row.get("Incorrect Answers", "")).strip()

        # Take only first incorrect answer
        first_wrong = incorrect.split(";")[0].strip()

        print(f"  [{i+1:2d}/{len(df)}] {question[:55]}...")

        # ── Best answer → should be FACTUAL ───────────────────────────
        if best_answer and best_answer.lower() != "nan":
            result = detector.detect(
                claim=best_answer,
                question=question
            )

            if result.retrieval_score < 0.05:
                retrieval_failures += 1

            all_scores.append({
                "retrieval_score":  result.retrieval_score,
                "semantic_score":   result.semantic_score,
                "confidence_score": result.confidence_score,
                "true_label":       "FACTUAL"
            })

            if result.verdict == "FACTUAL":
                tp += 1
                print(f"         ✓ CORRECT '{best_answer[:50]}' → {result.verdict} ({result.final_score:.3f})")
            else:
                fn += 1
                print(f"         ✗ CORRECT '{best_answer[:50]}' → {result.verdict} ({result.final_score:.3f})")
                errors.append({
                    "type":    "false_negative",
                    "question": question,
                    "claim":   best_answer,
                    "score":   result.final_score,
                    "verdict": result.verdict,
                    "r_score": result.retrieval_score,
                    "s_score": result.semantic_score,
                })

            time.sleep(2.0)

        # ── Incorrect answer → should be HALLUCINATION ────────────────
        if first_wrong and first_wrong.lower() != "nan":
            result = detector.detect(
                claim=first_wrong,
                question=question
            )

            if result.retrieval_score < 0.05:
                retrieval_failures += 1

            all_scores.append({
                "retrieval_score":  result.retrieval_score,
                "semantic_score":   result.semantic_score,
                "confidence_score": result.confidence_score,
                "true_label":       "HALLUCINATION"
            })

            if result.verdict != "FACTUAL":
                tn += 1
                print(f"         ✓ WRONG  '{first_wrong[:50]}' → {result.verdict} ({result.final_score:.3f})")
            else:
                fp += 1
                print(f"         ✗ WRONG  '{first_wrong[:50]}' → {result.verdict} ({result.final_score:.3f})")
                errors.append({
                    "type":    "false_positive",
                    "question": question,
                    "claim":   first_wrong,
                    "score":   result.final_score,
                    "verdict": result.verdict,
                    "r_score": result.retrieval_score,
                    "s_score": result.semantic_score,
                })

            time.sleep(2.0)

    # ── Metrics ────────────────────────────────────────────────────────
    threshold = detector.uncertain_threshold
    total     = tp + fp + tn + fn
    accuracy  = (tp + tn) / total if total else 0
    precision = tp / (tp + fp)    if (tp + fp) else 0
    recall    = tp / (tp + fn)    if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    ret_m  = compute_metrics(all_scores, "retrieval_score",  threshold)
    sem_m  = compute_metrics(all_scores, "semantic_score",   threshold)
    conf_m = compute_metrics(all_scores, "confidence_score", threshold)

    # ── Comparison table ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TRUTHFULQA RESULTS COMPARISON")
    print("=" * 70)
    print(f"  {'Method':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>8}")
    print("  " + "-" * 65)

    for label, m in [
        ("Retrieval Only",    ret_m),
        ("Semantic Only",     sem_m),
        ("Confidence Only",   conf_m),
        ("YOUR HYBRID MODEL", {
            "accuracy":  accuracy,
            "precision": precision,
            "recall":    recall,
            "f1":        f1
        }),
    ]:
        arrow = " ◄" if label == "YOUR HYBRID MODEL" else ""
        print(
            f"  {label:<25} "
            f"{m['accuracy']:>9.1%} "
            f"{m['precision']:>10.1%} "
            f"{m['recall']:>10.1%} "
            f"{m['f1']:>8.3f}"
            f"{arrow}"
        )

    print("=" * 70)
    print(f"\n  Questions evaluated  : {len(df)}")
    print(f"  Total predictions    : {total}")
    print(f"  Retrieval failures   : {retrieval_failures}/{total} "
          f"({'%.0f' % (retrieval_failures/total*100 if total else 0)}%)")
    print(f"  True Positives       : {tp}")
    print(f"  False Positives      : {fp}")
    print(f"  True Negatives       : {tn}")
    print(f"  False Negatives      : {fn}")

    if errors:
        print(f"\n  Top 3 errors:")
        for e in errors[:3]:
            print(f"\n  [{e['type']}]")
            print(f"  Q        : {e['question'][:75]}")
            print(f"  Claim    : {e['claim'][:75]}")
            print(f"  Score    : {e['score']:.3f} → {e['verdict']}")
            print(f"  Retrieval: {e['r_score']:.3f}  Semantic: {e['s_score']:.3f}")

    print("\n" + "=" * 70)
    return accuracy, f1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=30,
                        help="Number of questions to evaluate (default: 30)")
    args = parser.parse_args()
    run_truthfulqa(max_samples=args.samples)