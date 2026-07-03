"""
Display Utilities — pretty-print detection results.
"""

from core.detector import DetectionResult


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║       Hybrid Hallucination Detection System                      ║
║  Retrieval Similarity + Semantic Similarity + Confidence Score   ║
╚══════════════════════════════════════════════════════════════════╝
""")


def print_result(result: DetectionResult, compact: bool = False):
    """Pretty-print a DetectionResult."""

    if compact:
        bar = _score_bar(result.final_score)
        print(
            f"  {result.verdict_emoji} {result.verdict:<14} "
            f"Score: {result.final_score:.3f}  {bar}\n"
            f"     R={result.retrieval_score:.3f}  "
            f"S={result.semantic_score:.3f}  "
            f"C={result.confidence_score:.3f}  "
            f"[α={result.alpha:.2f} β={result.beta:.2f} γ={result.gamma:.2f}]"
        )
        return

    print("\n" + "─"*60)
    print(f"  Claim   : {result.claim}")
    if result.question:
        print(f"  Question: {result.question}")

    print("\n  ── Scores ──────────────────────────────────────────")
    print(f"  Retrieval Similarity  (α={result.alpha:.2f}): {result.retrieval_score:.4f}  {_score_bar(result.retrieval_score)}")
    print(f"  Semantic Similarity   (β={result.beta:.2f}): {result.semantic_score:.4f}  {_score_bar(result.semantic_score)}")
    print(f"  Confidence Score      (γ={result.gamma:.2f}): {result.confidence_score:.4f}  {_score_bar(result.confidence_score)}")
    print(f"\n  Hybrid Final Score          : {result.final_score:.4f}  {_score_bar(result.final_score)}")

    print(f"\n  ── Verdict ─────────────────────────────────────────")
    print(f"  {result.verdict_emoji}  {result.verdict}")

    if result.retrieved_passages:
        print(f"\n  ── Top Retrieved Passage (truncated) ────────────────")
        snippet = result.retrieved_passages[0][:250].replace("\n", " ")
        print(f"  \"{snippet}...\"")

    print(f"\n  ⏱  Processing time: {result.processing_time:.2f}s")
    print("─"*60 + "\n")


def _score_bar(score: float, width: int = 20) -> str:
    """Visual bar: ████░░░░░░ 0.65"""
    filled = int(score * width)
    bar = "█" * filled + "░" * (width - filled)
    color_code = _score_color(score)
    reset = "\033[0m"
    return f"{color_code}[{bar}]{reset} {score:.3f}"


def _score_color(score: float) -> str:
    if score >= 0.60:
        return "\033[92m"   # green
    elif score >= 0.35:
        return "\033[93m"   # yellow
    else:
        return "\033[91m"   # red
