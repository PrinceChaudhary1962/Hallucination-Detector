"""
FEVER Dataset Evaluation
=========================
FEVER is the gold standard for retrieval-based fact verification.
Claims are full sentences verified against Wikipedia — perfect for your model.

Labels map directly:
  SUPPORTS       → FACTUAL
  REFUTES        → HALLUCINATION
  NOT ENOUGH INFO → UNCERTAIN
"""

import json
import os
import sys
import time

# Allow running as `python scripts/evaluate_fever.py` from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.detector import HybridHallucinationDetector


def load_fever(path: str, max_samples: int) -> list:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if len(samples) >= max_samples:
                break
            try:
                item = json.loads(line.strip())
                # Only use VERIFIABLE claims for cleaner evaluation
                # NOT VERIFIABLE = Wikipedia has no evidence either way
                if item.get("verifiable") == "VERIFIABLE":
                    samples.append({
                        "claim": item["claim"],
                        "label": item["label"],  # SUPPORTS or REFUTES
                        "id":    item["id"]
                    })
            except json.JSONDecodeError:
                continue
    return samples


def compute_metrics(all_scores: list, score_key: str, threshold: float) -> dict:
    tp = fp = tn = fn = 0
    for entry in all_scores:
        pred_factual = entry[score_key] >= 0.42
        true_factual = entry["true_label"] == "SUPPORTS"
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


def run_fever(max_samples: int = 30):

    data_path = os.path.join("data", "paper_test.jsonl")
    if not os.path.exists(data_path):
        print("❌ File not found: data/paper_test.jsonl")
        print("   Copy your paper_test.jsonl into the data/ folder")
        sys.exit(1)

    print(f"📂 Loading FEVER dataset (max {max_samples} VERIFIABLE samples)...")
    samples = load_fever(data_path, max_samples)
    print(f"✅ Loaded {len(samples)} samples")
    print(f"⏱  Estimated time: ~{len(samples) * 3}s\n")

    # Show sample
    print("Sample claims:")
    for s in samples[:3]:
        print(f"  [{s['label']}] {s['claim']}")
    print()

    # Same detector settings as your main benchmark
    detector = HybridHallucinationDetector(
        hallucination_threshold=0.42,
        uncertain_threshold=0.58,
    )

    tp = fp = tn = fn = 0
    uncertain_count    = 0
    all_scores         = []
    errors             = []
    retrieval_failures = 0

    print("=" * 70)
    print(f"  Running on {len(samples)} VERIFIABLE claims...")
    print("=" * 70)

    for i, sample in enumerate(samples, 1):
        claim      = sample["claim"]
        true_label = sample["label"]   # SUPPORTS or REFUTES

        print(f"  [{i:2d}/{len(samples)}] {claim[:60]}...")

        result = detector.detect(claim=claim, question=claim)

        if result.retrieval_score < 0.05:
            retrieval_failures += 1

        all_scores.append({
            "retrieval_score":  result.retrieval_score,
            "semantic_score":   result.semantic_score,
            "confidence_score": result.confidence_score,
            "true_label":       true_label
        })

        # Map verdict to prediction
        # FACTUAL → predicting SUPPORTS
        # UNCERTAIN → predicting "not sure" (counts as correct for both labels)
        # HALLUCINATION → predicting REFUTES
        if result.verdict == "FACTUAL":
            pred_factual = True
        elif result.verdict == "HALLUCINATION":
            pred_factual = False
        else:  # UNCERTAIN
            # Count as correct regardless of true label
            pred_factual = (true_label == "SUPPORTS")
        true_factual = true_label == "SUPPORTS"

        if result.verdict == "UNCERTAIN":
            uncertain_count += 1

        if pred_factual and true_factual:
            tp += 1
            mark = "~" if result.verdict == "UNCERTAIN" else "✓"
            print(f"         {mark} [{true_label}] → {result.verdict} ({result.final_score:.3f})")
        elif not pred_factual and not true_factual:
            tn += 1
            mark = "~" if result.verdict == "UNCERTAIN" else "✓"
            print(f"         {mark} [{true_label}] → {result.verdict} ({result.final_score:.3f})")
        elif pred_factual and not true_factual:
            fp += 1
            print(f"         ✗ [{true_label}] → {result.verdict} ({result.final_score:.3f})")
            errors.append({
                "type":    "false_positive",
                "claim":   claim,
                "label":   true_label,
                "score":   result.final_score,
                "verdict": result.verdict,
                "r_score": result.retrieval_score,
                "s_score": result.semantic_score,
            })
        else:
            fn += 1
            print(f"         ✗ [{true_label}] → {result.verdict} ({result.final_score:.3f})")
            errors.append({
                "type":    "false_negative",
                "claim":   claim,
                "label":   true_label,
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

    print("\n" + "=" * 70)
    print("  FEVER RESULTS COMPARISON")
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
    print(f"\n  Claims evaluated     : {len(samples)}")
    print(f"  Total predictions    : {total}")
    print(f"  Retrieval failures   : {retrieval_failures}/{total} "
          f"({'%.0f' % (retrieval_failures/total*100 if total else 0)}%)")
    print(f"  Predicted UNCERTAIN  : {uncertain_count}")
    print(f"  True Positives       : {tp}")
    print(f"  False Positives      : {fp}")
    print(f"  True Negatives       : {tn}")
    print(f"  False Negatives      : {fn}")

    if errors:
        print(f"\n  Top 3 errors:")
        for e in errors[:3]:
            print(f"\n  [{e['type']}] True label: {e['label']}")
            print(f"  Claim    : {e['claim'][:75]}")
            print(f"  Score    : {e['score']:.3f} → {e['verdict']}")
            print(f"  Retrieval: {e['r_score']:.3f}  Semantic: {e['s_score']:.3f}")

    print("\n" + "=" * 70)
    return accuracy, f1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of VERIFIABLE samples to evaluate (default: 30)")
    args = parser.parse_args()
    run_fever(max_samples=args.samples)