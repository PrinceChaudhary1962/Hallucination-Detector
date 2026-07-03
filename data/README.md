# data/

This folder holds evaluation datasets used by the scripts in `scripts/`.
Datasets are **not** committed to the repo (see `.gitignore`) — they're
either large or redistributable from their original source, so fetch them
yourself:

| File                    | Used by                              | How to get it |
|-------------------------|---------------------------------------|----------------|
| `vitaminc_test.jsonl`   | `evaluate_vitaminc.py`, `evaluate_vitaminc_gold.py` | `python scripts/download_vitaminc.py` |
| `paper_test.jsonl`      | `evaluate_fever.py`                  | FEVER dataset — see https://fever.ai/dataset/fever.html |
| `qa_data.json`          | `evaluate_halueval.py`               | HaluEval dataset — see https://github.com/RUCAIBox/HaluEval |
| `TruthfulQA.csv`        | `evaluate_truthfulqa.py`             | https://github.com/sylinrl/TruthfulQA |

Place downloaded files directly in this folder before running the
corresponding evaluation script from the project root, e.g.:

```bash
python scripts/evaluate_vitaminc.py --samples 50
```
