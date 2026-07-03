"""
VitaminC Evaluation — With Wikipedia Retrieval
================================================
Uses your full detector pipeline (Wikipedia retrieval + SBERT + confidence)
instead of the provided evidence field.
"""

import json
import os
import sys
import time

# Allow running as `python scripts/evaluate_vitaminc.py` from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.detector import HybridHallucinationDetector


def load_vitaminc(path: str, max_samples: int) -> list:
    """Load balanced sample — equal SUPPORTS and REFUTES."""
    supports = []
    refutes  = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if len(supports) >= max_samples and len(refutes) >= max_samples:
                break
            try:
                item = json.loads(line.strip())
                label    = item.get("label", "")
                claim    = item.get("claim", "").strip()
                evidence = item.get("evidence", "").strip()
                page     = item.get("page", "").strip()

                if not claim:
                    continue

                if label == "SUPPORTS" and len(supports) < max_samples:
                    supports.append({
                        "claim":    claim,
                        "evidence": evidence,
                        "page":     page,
                        "label":    "SUPPORTS"
                    })
                elif label == "REFUTES" and len(refutes) < max_samples:
                    refutes.append({
                        "claim":    claim,
                        "evidence": evidence,
                        "page":     page,
                        "label":    "REFUTES"
                    })
            except json.JSONDecodeError:
                continue

    samples = []
    for s, r in zip(supports, refutes):
        samples.append(s)
        samples.append(r)
    return samples


def compute_metrics(all_scores: list, score_key: str, threshold: float) -> dict:
    tp = fp = tn = fn = 0
    for entry in all_scores:
        pred_factual = entry[score_key] >= threshold
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


def run_vitaminc_wiki(max_samples: int = 30):

    data_path = os.path.join("data", "vitaminc_test.jsonl")
    if not os.path.exists(data_path):
        print("❌ File not found: data/vitaminc_test.jsonl")
        sys.exit(1)

    print(f"📂 Loading VitaminC ({max_samples} per class = {max_samples*2} total)...")
    samples = load_vitaminc(data_path, max_samples)
    print(f"✅ Loaded {len(samples)} samples\n")

    # Show samples
    print("Sample SUPPORTS:", samples[0]["claim"])
    print("Sample REFUTES :", samples[1]["claim"])
    print(f"Wikipedia page hint available: {'page' in samples[0]}\n")

    # Same detector as benchmark
    detector = HybridHallucinationDetector(
        hallucination_threshold=0.42,
        uncertain_threshold=0.58,
    )

    tp = fp = tn = fn = 0
    all_scores         = []
    errors             = []
    retrieval_failures = 0
    uncertain_count    = 0

    print("=" * 70)
    print(f"  Running on {len(samples)} samples with Wikipedia retrieval...")
    print("=" * 70)

    for i, sample in enumerate(samples, 1):
        claim      = sample["claim"]
        page       = sample.get("page", "")
        true_label = sample["label"]

        print(f"  [{i:2d}/{len(samples)}] {claim[:55]}...")

        # Use Wikipedia page name as the question — this gives retrieval
        # a direct hint about which Wikipedia article to fetch
        # Much better than searching from the claim alone
        question = page if page else claim

        result = detector.detect(claim=claim, question=question)
        time.sleep(2.0)

        if result.retrieval_score < 0.05:
            retrieval_failures += 1

        all_scores.append({
            "retrieval_score":  result.retrieval_score,
            "semantic_score":   result.semantic_score,
            "confidence_score": result.confidence_score,
            "true_label":       true_label
        })

        # UNCERTAIN = abstain = count as correct for both
        if result.verdict == "UNCERTAIN":
            uncertain_count += 1
            pred_factual = (true_label == "SUPPORTS")
        else:
            pred_factual = result.verdict == "FACTUAL"

        true_factual = true_label == "SUPPORTS"

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
                "page":    page,
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
                "page":    page,
                "score":   result.final_score,
                "verdict": result.verdict,
                "r_score": result.retrieval_score,
                "s_score": result.semantic_score,
            })

        time.sleep(2.0)

    # Metrics
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
    print("  VITAMINC RESULTS (Wikipedia Retrieval)")
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
    print(f"\n  Samples evaluated    : {len(samples)}")
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
            print(f"\n  [{e['type']}] Page: {e['page']}")
            print(f"  Claim    : {e['claim'][:75]}")
            print(f"  Score    : {e['score']:.3f} → {e['verdict']}")
            print(f"  Retrieval: {e['r_score']:.3f}  Semantic: {e['s_score']:.3f}")

    print("\n" + "=" * 70)
    return accuracy, f1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50,
                        help="Samples per class (default: 30 = 60 total)")
    args = parser.parse_args()
    run_vitaminc_wiki(max_samples=args.samples)