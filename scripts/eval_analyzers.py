"""CLI: python scripts/eval_analyzers.py

Runs data/test.jsonl through heuristic, ollama (base qwen2.5:7b), and tinker
(fine-tuned). Writes tag-level precision/recall and the no_metrics false-negative
rate to eval_results.json.

If a model is not configured, that analyzer falls back to the heuristic.
Those rows are marked used_fallback so the table does not pretend a fine-tune ran.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.analyzers import get_analyzer
from app.analyzers.prompts import extract_json
from app.analyzers.schemas import TagList
from app.models import QA

TEST_PATH = Path("data/test.jsonl")
PREFS = {
    "target_role": "backend engineer",
    "domain": "product infrastructure",
    "weak_areas": "no_metrics, rambling",
}
NAMES = ("heuristic", "ollama", "tinker")


def qa_from_user(content: str) -> QA:
    question, _, answer = content.partition("\nAnswer:")
    question = question.removeprefix("Question:").strip()
    answer = answer.strip()
    if answer.lower().startswith("duration seconds:"):
        answer = answer.split("\n", 1)[-1].strip()
    return QA(session_id=0, question_text=question, her_answer=answer)


def gold_tags(messages: list[dict]) -> set[str]:
    assistant = next(message["content"] for message in messages if message["role"] == "assistant")
    parsed = TagList.model_validate_json(extract_json(assistant))
    return {item.tag for item in parsed.tags}


def score(pairs: list[tuple[set[str], set[str]]]) -> dict:
    tp = fp = fn = 0
    no_metrics_positives = 0
    no_metrics_misses = 0
    for gold, predicted in pairs:
        tp += len(gold & predicted)
        fp += len(predicted - gold)
        fn += len(gold - predicted)
        if "no_metrics" in gold:
            no_metrics_positives += 1
            if "no_metrics" not in predicted:
                no_metrics_misses += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fnr = (no_metrics_misses / no_metrics_positives) if no_metrics_positives else 0.0
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "no_metrics_false_negative_rate": round(fnr, 3),
        "no_metrics_positives": no_metrics_positives,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }


def main() -> None:
    if not TEST_PATH.exists():
        print("data/test.jsonl is missing. Run python data/make_jsonl.py first.")
        return

    rows = []
    with TEST_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))

    results = {}
    for name in NAMES:
        analyzer = get_analyzer(name)
        pairs = []
        fallbacks = 0
        for row in rows:
            messages = row["messages"]
            user = next(message["content"] for message in messages if message["role"] == "user")
            qa = qa_from_user(user)
            predicted = {tag.tag for tag in analyzer.analyze(qa, PREFS)}
            if analyzer.used_fallback:
                fallbacks += 1
            pairs.append((gold_tags(messages), predicted))
        metrics = score(pairs)
        metrics["fallback_count"] = fallbacks
        metrics["examples"] = len(rows)
        metrics["analyzer"] = analyzer.name
        if name != "heuristic" and fallbacks == len(rows):
            metrics["note"] = (
                "Every call fell back to the heuristic. These numbers are not a "
                "measurement of the base or fine-tuned model."
            )
        results[name] = metrics
        print(
            f"{name}: precision={metrics['precision']} recall={metrics['recall']} "
            f"no_metrics_fnr={metrics['no_metrics_false_negative_rate']} "
            f"fallbacks={fallbacks}/{len(rows)}"
        )

    with open("eval_results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print("Wrote eval_results.json")


if __name__ == "__main__":
    main()
