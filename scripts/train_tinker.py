"""CLI: python scripts/train_tinker.py

LoRA-fine-tunes Qwen/Qwen2.5-7B-Instruct on data/train.jsonl and writes the
sampler path to .env as TINKER_MODEL_NAME.

Requires: pip install -r requirements-train.txt and TINKER_API_KEY in .env.
The training rows use the same system prompt as app/analyzers/prompts.py.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
# Tinker's catalog no longer serves Qwen2.5-7B-Instruct (the API returns 400).
# Qwen3-8B is the closest supported open instruct model in the same size class.
FALLBACK_MODEL = "Qwen/Qwen3-8B"
DATA_PATH = Path("data/train.jsonl")
ENV_PATH = Path(".env")


def load_conversations(path: Path) -> list[list[dict[str, str]]]:
    conversations = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                conversations.append(json.loads(line)["messages"])
    return conversations


def update_env(key: str, value: str) -> None:
    lines = ENV_PATH.read_text(encoding="utf-8").splitlines() if ENV_PATH.exists() else []
    written = []
    found = False
    for line in lines:
        if line.startswith(f"{key}="):
            written.append(f"{key}={value}")
            found = True
        else:
            written.append(line)
    if not found:
        written.append(f"{key}={value}")
    ENV_PATH.write_text("\n".join(written) + "\n", encoding="utf-8")


async def train(rank: int, steps: int, max_length: int, batch_size: int) -> None:
    import tinker
    from tinker_cookbook.hyperparam_utils import get_lr
    from tinker_cookbook.model_info import get_recommended_renderer_name
    from tinker_cookbook.renderers import TrainOnWhat, get_renderer
    from tinker_cookbook.supervised.data import conversation_to_datum

    service_client = tinker.ServiceClient()
    base_model = BASE_MODEL
    try:
        training_client = await service_client.create_lora_training_client_async(
            base_model=base_model, rank=rank
        )
    except Exception as exc:
        if "not supported" not in str(exc):
            raise
        print(f"{base_model} is not supported by this Tinker account.")
        print(f"Falling back to {FALLBACK_MODEL}.")
        base_model = FALLBACK_MODEL
        training_client = await service_client.create_lora_training_client_async(
            base_model=base_model, rank=rank
        )
    tokenizer = training_client.get_tokenizer()
    try:
        renderer_name = get_recommended_renderer_name(base_model)
    except Exception:
        renderer_name = "qwen3"
    renderer = get_renderer(renderer_name, tokenizer)
    try:
        lr = get_lr(base_model)
    except Exception:
        lr = 1e-4

    conversations = load_conversations(DATA_PATH)
    print(f"Loaded {len(conversations)} training conversations")
    data = [
        conversation_to_datum(
            conv,
            renderer,
            max_length=max_length,
            train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        )
        for conv in conversations
    ]

    print(f"Base: {base_model} | renderer: {renderer_name} | LoRA rank: {rank} | lr: {lr} | steps: {steps}")
    for step in range(steps):
        batch = [
            data[(step * batch_size + i) % len(data)]
            for i in range(min(batch_size, len(data)))
        ]
        started = time.time()
        forward = await training_client.forward_backward_async(batch, "cross_entropy")
        optim = await training_client.optim_step_async(tinker.AdamParams(learning_rate=lr))
        result = await forward.result_async()
        await optim.result_async()
        metrics = result.metrics or {}
        loss = metrics.get("loss", next(iter(metrics.values()), float("nan")))
        print(
            f"Step {step + 1}/{steps}: loss = {loss:.4f} ({time.time() - started:.1f}s)"
        )

    saved = await training_client.save_weights_for_sampler_async(name="duster")
    weights_path = None
    if saved is not None:
        weights_path = getattr(saved, "path", None) or getattr(saved, "weights_path", None)
    # Some SDK builds return an empty save result even after the checkpoint exists.
    if not weights_path:
        listed = await service_client.create_rest_client().list_user_checkpoints_async(limit=20)
        for checkpoint in listed.checkpoints:
            if checkpoint.checkpoint_id.endswith("duster"):
                weights_path = checkpoint.tinker_path
                break
    if weights_path:
        update_env("TINKER_MODEL_NAME", weights_path)
        print(f"Saved. Wrote TINKER_MODEL_NAME={weights_path} to .env")
        print("Set ANALYZER=tinker to serve the fine-tune.")
    else:
        print("Training finished, but the sampler path was not on the save result.")
        print("Copy it from the Tinker console into TINKER_MODEL_NAME.")


def main() -> None:
    if settings.tinker_api_key:
        os.environ.setdefault("TINKER_API_KEY", settings.tinker_api_key)
    if not os.environ.get("TINKER_API_KEY"):
        print("TINKER_API_KEY is not set. Add it to .env before training.")
        return
    if not DATA_PATH.exists():
        print(f"{DATA_PATH} is missing. Run python data/make_jsonl.py first.")
        return
    asyncio.run(train(rank=32, steps=100, max_length=1024, batch_size=4))


if __name__ == "__main__":
    main()
