"""
HaluEval Evaluation Script
===========================
Reconstructs full declarative sentences from Q+A pairs
so the detector works on proper claims as it was designed for.
"""

import json
import os
import sys
import time

# Allow running as `python scripts/evaluate_halueval.py` from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.detector import HybridHallucinationDetector


def load_halueval(path: str, max_samples: int) -> list:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            try:
                item = json.loads(line.strip())
                samples.append(item)
            except json.JSONDecodeError:
                continue
    return samples


def build_full_claim(question: str, answer: str) -> str:
    """
    Convert Q+A into a full declarative sentence your model understands.

    Examples:
      Q: "Which magazine was started first?"  A: "Arthur's Magazine"
      → "Arthur's Magazine was started first."

      Q: "What nationality was James Henry Miller's wife?"  A: "American"
      → "James Henry Miller's wife was American."

      Q: "Where did the music originate?"  A: "United States"
      → "The music originated in the United States."

      Q: "How old is the protagonist?"  A: "16-year-old"
      → "The protagonist is 16-year-old."
    """
    q = question.strip().rstrip("?").strip()
    a = answer.strip().rstrip(".").strip()

    q_lower = q.lower()

    # Already a full sentence (hallucinated answers are usually full sentences)
    if len(a.split()) > 6:
        return a

    # "Which X" → "X was/is [answer]"
    if q_lower.startswith("which"):
        rest = q[6:].strip()  # remove "which"
        return f"{a} is the {rest}."

    # "Who" → subject + answer
    elif q_lower.startswith("who was") or q_lower.startswith("who is"):
        rest = q[7:].strip()
        return f"{a} {rest}."

    elif q_lower.startswith("who"):
        rest = q[4:].strip()
        return f"{a} {rest}."

    # "What nationality" → "X was [nationality]"
    elif "nationality" in q_lower:
        subject = q_lower.replace("what nationality was", "").replace(
                  "what nationality is", "").strip()
        return f"{subject.title()} was {a}."

    # "What is the length/size/age" → direct statement
    elif q_lower.startswith("what is") or q_lower.startswith("what was"):
        rest = q[8:].strip()
        return f"The {rest} is {a}."

    # "What" general
    elif q_lower.startswith("what"):
        rest = q[5:].strip()
        return f"The {rest} is {a}."

    # "Where" → location statement
    elif q_lower.startswith("where"):
        rest = q[6:].strip()
        return f"{rest.capitalize()} in {a}."

    # "When" → time statement
    elif q_lower.startswith("when"):
        rest = q[5:].strip()
        return f"{rest.capitalize()} in {a}."

    # "How old" → age statement
    elif q_lower.startswith("how old"):
        rest = q[8:].strip()
        return f"{rest.capitalize()} is {a}."

    # "How many / how much"
    elif q_lower.startswith("how"):
        rest = q[4:].strip()
        return f"The {rest} is {a}."

    # "Are both / Is X" → yes/no answer
    elif q_lower.startswith("are") or q_lower.startswith("is ") or q_lower.startswith("were"):
        return f"The answer is {a}. {q}."

    # "In which" → location
    elif q_lower.startswith("in which"):
        rest = q[9:].strip()
        return f"{rest.capitalize()} in {a}."

    # Default — append answer to question context
    else:
        return f"{a}. {q}."


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


def run_halueval(max_samples: int = 30):

    data_path = os.path.join("data", "qa_data.json")
    if not os.path.exists(data_path):
        print("❌ File not found: data/qa_data.json")
        sys.exit(1)

    print(f"📂 Loading HaluEval dataset (max {max_samples} samples)...")
    samples = load_halueval(data_path, max_samples)
    print(f"✅ Loaded {len(samples)} samples")
    print(f"⏱  Estimated time: ~{max_samples * 2 * 3}s\n")

    # Show how sentence reconstruction works
    s = samples[0]
    print("Sentence reconstruction example:")
    print(f"  Q      : {s['question']}")
    print(f"  Right  : {s['right_answer']}")
    print(f"  Built  : {build_full_claim(s['question'], s['right_answer'])}")
    print(f"  Hal    : {s['hallucinated_answer']}")
    print(f"  Built  : {build_full_claim(s['question'], s['hallucinated_answer'])}")
    print()

    detector = HybridHallucinationDetector(
        hallucination_threshold=0.42,
        uncertain_threshold=0.58,
    )

    tp = fp = tn = fn = 0
    all_scores         = []
    errors             = []
    retrieval_failures = 0

    print("=" * 70)
    print(f"  Running on {len(samples)} samples ({len(samples)*2} predictions)...")
    print("=" * 70)

    for i, item in enumerate(samples, 1):
        question     = item.get("question", "")
        right_answer = item.get("right_answer", "")
        hal_answer   = item.get("hallucinated_answer", "")

        print(f"  [{i:2d}/{len(samples)}] {question[:55]}...")

        # ── Right answer → should be FACTUAL ──────────────────────────
        if right_answer:
            # Build full sentence claim
            full_claim = build_full_claim(question, right_answer)

            result = detector.detect(
                claim=full_claim,
                question=question   # question drives Wikipedia retrieval
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
                print(f"         ✓ RIGHT  '{full_claim[:50]}' → {result.verdict} ({result.final_score:.3f})")
            else:
                fn += 1
                print(f"         ✗ RIGHT  '{full_claim[:50]}' → {result.verdict} ({result.final_score:.3f})")
                errors.append({
                    "type":     "false_negative",
                    "question": question,
                    "claim":    full_claim,
                    "score":    result.final_score,
                    "verdict":  result.verdict,
                    "r_score":  result.retrieval_score,
                    "s_score":  result.semantic_score,
                })

            time.sleep(2.0)

        # ── Hallucinated answer → should be HALLUCINATION ─────────────
        if hal_answer:
            full_claim = build_full_claim(question, hal_answer)

            result = detector.detect(
                claim=full_claim,
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
                print(f"         ✓ HALL   '{full_claim[:50]}' → {result.verdict} ({result.final_score:.3f})")
            else:
                fp += 1
                print(f"         ✗ HALL   '{full_claim[:50]}' → {result.verdict} ({result.final_score:.3f})")
                errors.append({
                    "type":     "false_positive",
                    "question": question,
                    "claim":    full_claim,
                    "score":    result.final_score,
                    "verdict":  result.verdict,
                    "r_score":  result.retrieval_score,
                    "s_score":  result.semantic_score,
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
    print("  HALUEVAL RESULTS COMPARISON")
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
                        help="Number of samples (default: 30)")
    args = parser.parse_args()
    run_halueval(max_samples=args.samples)