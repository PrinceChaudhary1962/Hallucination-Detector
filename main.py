"""
Hybrid Hallucination Detection System
======================================
Combines Retrieval Similarity + Semantic Similarity + Confidence Score
into a single weighted hybrid score for detecting LLM hallucinations.

Usage:
    python main.py --text "Your claim here"
    python main.py --demo
    python main.py --benchmark
"""

import argparse
from core.detector import HybridHallucinationDetector
from utils.benchmark import run_benchmark
from utils.display import print_result, print_banner


def main():
    print_banner()

    parser = argparse.ArgumentParser(description="Hybrid Hallucination Detector")
    parser.add_argument("--text", type=str, help="Claim/answer text to evaluate")
    parser.add_argument("--question", type=str, help="Original question (optional, improves retrieval)")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for retrieval similarity (default: 0.4)")
    parser.add_argument("--beta", type=float, default=0.4, help="Weight for semantic similarity (default: 0.4)")
    parser.add_argument("--gamma", type=float, default=0.1, help="Weight for confidence score (default: 0.2)")
    parser.add_argument("--adaptive", action="store_true", help="Use adaptive weight adjustment")
    parser.add_argument("--demo", action="store_true", help="Run demo with example claims")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark evaluation")
    parser.add_argument("--model", type=str, default="google/flan-t5-base",
                        help="HuggingFace model for answer generation")

    args = parser.parse_args()

    # Initialize detector
    detector = HybridHallucinationDetector(
        model_name=args.model,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        adaptive_weights=args.adaptive
    )

    if args.demo:
        run_demo(detector)
    elif args.benchmark:
        run_benchmark(detector)
    elif args.text:
        result = detector.detect(claim=args.text, question=args.question)
        print_result(result)
    else:
        parser.print_help()
        print("\n💡 Quick start: python main.py --demo")


def run_demo(detector):
    """Run demo with a set of example claims — some true, some hallucinated."""

    demo_cases = [
        {
            "question": "Who wrote the theory of relativity?",
            "claim": "Albert Einstein developed the theory of relativity in the early 20th century.",
            "label": "FACTUAL"
        },
        {
            "question": "What is the capital of Australia?",
            "claim": "The capital of Australia is Sydney.",
            "label": "HALLUCINATION"  # Common misconception — it's Canberra
        },
        {
            "question": "When was the Eiffel Tower built?",
            "claim": "The Eiffel Tower was constructed between 1887 and 1889 for the World's Fair.",
            "label": "FACTUAL"
        },
        {
            "question": "Who invented the telephone?",
            "claim": "The telephone was invented by Nikola Tesla in 1876.",
            "label": "HALLUCINATION"  # Alexander Graham Bell
        },
        {
            "question": "What is the boiling point of water?",
            "claim": "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
            "label": "FACTUAL"
        },
    ]

    print("\n" + "="*70)
    print("  DEMO MODE — Running 5 example claims")
    print("="*70)

    for i, case in enumerate(demo_cases, 1):
        print(f"\n[{i}/5] Question: {case['question']}")
        print(f"      Claim   : {case['claim']}")
        print(f"      Label   : {case['label']}")
        print("-" * 50)

        result = detector.detect(claim=case["claim"], question=case["question"])
        from utils.display import print_result
        print_result(result, compact=True)

    print("\n" + "="*70)
    print("  Demo complete.")
    print("="*70)


if __name__ == "__main__":
    main()
