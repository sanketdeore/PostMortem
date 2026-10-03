"""CLI: python -m app.compare_models

Runs the same 10 sample answers through three open-weight models on Backboard
and writes tag agreement, latency, and cost to compare_results.json.
"""

import json
from itertools import combinations

import httpx

from .analyzers.prompts import extract_json, system_prompt, user_prompt
from .analyzers.schemas import TagList
from .backboard_client import post_message, token_usage
from .config import settings
from .models import QA
from .open_models import resolve_open_model
from .preferences import get_prefs
from .sample_qa import SAMPLE_QA


def _prices() -> dict[str, tuple[float | None, float | None]]:
    response = httpx.get(
        f"{settings.backboard_base_url.rstrip('/')}/models",
        headers={"X-API-Key": settings.backboard_api_key},
        params={"model_type": "llm", "limit": 500},
        timeout=30.0,
    )
    response.raise_for_status()
    prices: dict[str, tuple[float | None, float | None]] = {}
    for model in response.json().get("models", []):
        pair = (
            model.get("input_cost_per_1m_tokens"),
            model.get("output_cost_per_1m_tokens"),
        )
        prices[model.get("name", "")] = pair
    return prices


def _cost(prices: dict, model: str, input_tokens: int, output_tokens: int) -> float | None:
    slug = resolve_open_model(model)
    pair = prices.get(slug) or prices.get(model)
    if not pair:
        for name, value in prices.items():
            if slug.endswith(name) or name.endswith(model):
                pair = value
                break
    if not pair or pair[0] is None or pair[1] is None:
        return None
    return input_tokens * float(pair[0]) / 1e6 + output_tokens * float(pair[1]) / 1e6


def main() -> None:
    if not settings.backboard_api_key:
        print("BACKBOARD_API_KEY is not set — add it to .env first.")
        return

    prefs = get_prefs()
    prompt = system_prompt(prefs)
    models = [item.strip() for item in settings.compare_models.split(",") if item.strip()]
    try:
        prices = _prices()
    except httpx.HTTPError as exc:
        print(f"Could not load model prices ({exc}). Cost will be null.")
        prices = {}

    per_model: dict[str, dict] = {}
    tag_sets: dict[str, list[list[str]]] = {}

    for model in models:
        rows = []
        sets: list[list[str]] = []
        for question, answer in SAMPLE_QA:
            qa = QA(session_id=0, question_text=question, her_answer=answer)
            try:
                payload, latency_ms = post_message(
                    user_prompt(qa),
                    model=model,
                    system_prompt=prompt,
                    memory="off",
                    json_output=True,
                )
                content = payload.get("content") or ""
                if content.startswith("LLM Error") or content.startswith("Error"):
                    raise ValueError(content[:300])
                parsed = TagList.model_validate_json(extract_json(content))
                tags = sorted({item.tag for item in parsed.tags})
                incoming, outgoing = token_usage(payload)
                rows.append(
                    {
                        "question": question,
                        "tags": tags,
                        "latency_ms": round(latency_ms, 1),
                        "input_tokens": incoming,
                        "output_tokens": outgoing,
                        "reported_cost": payload.get("cost"),
                    }
                )
                sets.append(tags)
            except Exception as exc:
                rows.append({"question": question, "error": str(exc)[:300]})
                sets.append(None)
        ok = [row for row in rows if "error" not in row]
        input_tokens = sum(row["input_tokens"] for row in ok)
        output_tokens = sum(row["output_tokens"] for row in ok)
        reported = [row["reported_cost"] for row in ok if isinstance(row["reported_cost"], (int, float))]
        total = sum(reported) if reported else _cost(prices, model, input_tokens, output_tokens)
        per_model[model] = {
            "slug": resolve_open_model(model),
            "rows": rows,
            "ok": len(ok),
            "avg_latency_ms": round(sum(row["latency_ms"] for row in ok) / len(ok), 1) if ok else None,
            "total_cost_usd": round(total, 6) if total is not None else None,
        }
        tag_sets[model] = sets
        print(
            f"{model}: ok={len(ok)}/{len(SAMPLE_QA)} "
            f"avg_latency={per_model[model]['avg_latency_ms']}ms "
            f"cost={per_model[model]['total_cost_usd']}"
        )

    agreement = {}
    for left, right in combinations(models, 2):
        matches = 0
        compared = 0
        for a, b in zip(tag_sets[left], tag_sets[right]):
            if a is None or b is None:
                continue
            compared += 1
            if a == b:
                matches += 1
        rate = round(matches / compared, 3) if compared else None
        agreement[f"{left} vs {right}"] = {"rate": rate, "compared": compared}
        print(f"agreement {left} vs {right}: {rate} over {compared} answers")

    document = {
        "sample_size": len(SAMPLE_QA),
        "provider": settings.backboard_llm_provider,
        "per_model": per_model,
        "tag_agreement": agreement,
    }
    with open("compare_results.json", "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)
    print("Wrote compare_results.json")


if __name__ == "__main__":
    main()
