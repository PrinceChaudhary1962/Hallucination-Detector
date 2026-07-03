# Hybrid Hallucination Detection System

**Final Score = α × Retrieval_Similarity + β × Semantic_Similarity + γ × Confidence**

Combines three independent signals to detect hallucinations in LLM-generated text — no training required.

---

## Architecture

```
┌────────────────────────────────────────────────────────┐
│                     INPUT CLAIM                         │
└───────────────────────┬────────────────────────────────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
   ┌──────────┐  ┌──────────┐  ┌──────────┐
   │Retrieval │  │Semantic  │  │Confidence│
   │Wikipedia │  │  SBERT   │  │  Entropy │
   │TF-IDF    │  │Cosine Sim│  │ Log-Prob │
   └────┬─────┘  └────┬─────┘  └────┬─────┘
        │              │              │
        ▼              ▼              ▼
       r_score        s_score       c_score
        │              │              │
        └──────────────┴──────────────┘
                       │
               ┌───────▼────────┐
               │ Adaptive Weight│  (optional)
               │  Optimizer     │
               └───────┬────────┘
                       │
              α·r + β·s + γ·c
                       │
                ┌──────▼──────┐
                │  VERDICT    │
                │ FACTUAL /   │
                │ UNCERTAIN / │
                │HALLUCINATION│
                └─────────────┘
```

## Project Structure

```
hallucination_detector/
│
├── main.py                       # Entry point
│
├── core/
│   ├── __init__.py
│   ├── detector.py               # HybridHallucinationDetector (main class)
│   ├── retrieval.py              # Wikipedia retrieval + TF-IDF similarity
│   ├── semantic.py               # SBERT semantic similarity scorer
│   ├── confidence.py             # Token entropy / log-prob confidence scorer
│   └── adaptive.py               # AdaptiveWeightOptimizer
│
├── utils/
│   ├── __init__.py
│   ├── benchmark.py              # Built-in benchmark evaluation + baselines
│   └── display.py                # Pretty-print results
│
├── scripts/                      # External dataset evaluation (run from project root)
│   ├── download_vitaminc.py      # Fetches VitaminC test split into data/
│   ├── evaluate_fever.py         # FEVER benchmark
│   ├── evaluate_halueval.py      # HaluEval benchmark
│   ├── evaluate_truthfulqa.py    # TruthfulQA benchmark
│   ├── evaluate_vitaminc.py      # VitaminC w/ live Wikipedia retrieval
│   └── evaluate_vitaminc_gold.py # VitaminC w/ gold evidence (matches prior work)
│
├── data/                         # Datasets go here (gitignored — see data/README.md)
│   └── README.md
│
├── requirements.txt
└── .gitignore
```

### Running the evaluation scripts

All scripts in `scripts/` expect to be run **from the project root** (they
read from and expect the `core` package to be importable):

```bash
python scripts/download_vitaminc.py
python scripts/evaluate_vitaminc.py --samples 50
python scripts/evaluate_fever.py --samples 30
python scripts/evaluate_halueval.py --samples 30
python scripts/evaluate_truthfulqa.py --samples 30
```

See `data/README.md` for where to get each dataset (`paper_test.jsonl`
for FEVER, `qa_data.json` for HaluEval, `TruthfulQA.csv` for TruthfulQA).

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Single claim
```bash
python main.py --text "Albert Einstein invented the telephone in 1876."
```

### With original question (improves retrieval)
```bash
python main.py \
  --question "Who invented the telephone?" \
  --text "Alexander Graham Bell invented the telephone in 1876."
```

### Run demo (5 examples)
```bash
python main.py --demo
```

### Run benchmark (20 test cases + baseline comparison)
```bash
python main.py --benchmark
```

### Custom weights
```bash
python main.py --alpha 0.5 --beta 0.3 --gamma 0.2 --text "Your claim here"
```

### Adaptive weights (your paper's main contribution)
```bash
python main.py --adaptive --text "Your claim here"
```

## Python API

```python
from core.detector import HybridHallucinationDetector

detector = HybridHallucinationDetector(
    model_name="google/flan-t5-base",   # for confidence scoring
    alpha=0.4,                           # retrieval weight
    beta=0.4,                            # semantic weight
    gamma=0.2,                           # confidence weight
    adaptive_weights=True,               # use AdaptiveWeightOptimizer
)

result = detector.detect(
    claim="The capital of Australia is Sydney.",
    question="What is the capital of Australia?"
)

print(result.verdict)        # HALLUCINATION
print(result.final_score)    # e.g. 0.21
print(result.retrieval_score)
print(result.semantic_score)
print(result.confidence_score)

# Full JSON-serializable dict
import json
print(json.dumps(result.to_dict(), indent=2))
```

### Batch detection

```python
items = [
    {"question": "Who wrote Hamlet?", "claim": "Hamlet was written by Shakespeare."},
    {"question": "Who wrote Hamlet?", "claim": "Hamlet was written by Dickens."},
]
results = detector.detect_batch(items)
for r in results:
    print(r.verdict, r.final_score)
```

## Scoring Details

| Signal | Method | Score Range |
|--------|--------|-------------|
| Retrieval Similarity | TF-IDF weighted token overlap vs Wikipedia | [0, 1] |
| Semantic Similarity | SBERT cosine similarity (fallback: TF-IDF cosine) | [0, 1] |
| Confidence Score | Token entropy from HuggingFace model (fallback: heuristic) | [0, 1] |

## Verdict Thresholds

| Score | Verdict |
|-------|---------|
| ≥ 0.60 | 🟢 FACTUAL |
| 0.35 – 0.59 | 🟡 UNCERTAIN |
| < 0.35 | 🔴 HALLUCINATION |

Thresholds can be customized:
```python
detector = HybridHallucinationDetector(
    hallucination_threshold=0.40,
    uncertain_threshold=0.65
)
```

## Adaptive Weights

Three strategies available in `AdaptiveWeightOptimizer`:
- `signal_variance` — down-weights extreme signals, up-weights informative ones
- `calibrated` — rule-based adjustment based on signal agreement patterns
- `entropy_reweight` — redistributes weight to signals closest to consensus

## Paper Contribution Summary

> **"Adaptive Hybrid Hallucination Detection Framework"**
> 
> Combines retrieval-augmented similarity, dense semantic matching, and
> model confidence into a unified score. Adaptive weight optimization
> dynamically adjusts signal contributions per sample, outperforming
> single-signal baselines without requiring additional training.
