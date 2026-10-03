# PostMortem

An interview post-mortem coach for one person.

She does the interview, gets rejected or ghosted, and never learns why. Rejections never come with feedback. PostMortem turns the interview the way she remembers it — a messy `Q:` / `A:` dump, or a voice note — into the weakness that keeps showing up, then gives her a drill aimed at that weakness.

Open-source models only. Nothing in this repo calls OpenAI or Anthropic. Every model call goes through Ollama on her machine, Backboard's open-weight models, or a Tinker fine-tune. With no keys set, the heuristic analyzer still tags answers.

## What it does

- **Debrief.** Company, role, outcome, and a dump of what they asked and what she said. Optional `.m4a` / `.mp3` voice note, transcribed locally with faster-whisper and deleted. The audio never leaves the machine.
- **Patterns.** Weakness tags counted across interviews (`no_metrics — 7x in 4 sessions`), with an arrow for whether that tag is showing up more or less lately. Outcome badges. A drill streak.
- **Drills.** Five practice questions for her target role. She answers, gets the tags, concrete feedback, and a rewritten strong version of *her* answer. Focus mode biases the questions toward her top two tags.
- **Memory.** Each drill is appended to one Backboard assistant thread, and the home page reads that thread back.
- **Comparison and fine-tune.** The same ten answers can be run through three open models on Backboard. A LoRA fine-tune of Qwen2.5-7B-Instruct can be trained on `data/train.jsonl` and scored against the heuristic and the base model.

## Architecture

```
  voice note (.m4a/.mp3)                typed dump
           |                                |
           v                                v
   faster-whisper (local CPU) ----->  parse Q: / A:
   file deleted after                      |
                                           v
                                    +--------------+
                                    |   SQLite     |
                                    | session, qa  |
                                    | weakness_tag |
                                    | drill        |
                                    +------+-------+
                                           |
                                           v
                              get_analyzer()  ANALYZER=
                              +--------+--------+--------+
                              |        |        |        |
                         heuristic  ollama  backboard  tinker
                         (rules)   qwen2.5   open      LoRA
                                   local     weights   Qwen
                                           |
                                           v
                              home: counts, arrows, streak
                              drill: 5 questions, score, rewrite
                                           |
                                           v
                              one Backboard assistant thread
                              (drill history, read back on the home page)
```

## Setup

Python 3.11.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
copy .env.example .env          # then edit .env
python -m app.seed
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000. The seed loads two interviews (Northwind rejected, Lumen Health ghosted) so the pattern page is not empty.

### Analyzers

| `ANALYZER` | What runs |
| --- | --- |
| `heuristic` | Length, digits, filler density, structure words. No network. Default. |
| `ollama` | Local `qwen2.5:7b`. `ollama pull qwen2.5:7b` first. |
| `backboard` | Open-weight model via `https://app.backboard.io/api` and `X-API-Key`. Provider is `openrouter`. A closed provider is refused. |
| `tinker` | Fine-tuned sampler from `TINKER_MODEL_NAME`. Falls back to the heuristic if the key or the model path is missing. |

Every model response is checked with Pydantic. If the JSON is missing or invalid, that call uses the heuristic and the page says so. A model failure does not crash the request.

### Voice notes

The first transcription downloads the whisper model (default `base`) onto this machine. ffmpeg has to be on the PATH for `.m4a`. The Docker image installs it. The temp file is removed when transcription finishes.

### Backboard model comparison

```bash
python -m app.compare_models
```

Runs the 10 answers in `app/sample_qa.py` through `llama-3.1-8b-instruct`, `qwen2.5-7b-instruct`, and `mistral-7b-instruct` (mapped to OpenRouter slugs) and writes `compare_results.json` with per-model tags, latency, cost, and pairwise tag agreement.

### Tinker fine-tune

```bash
pip install -r requirements-train.txt
python data/make_jsonl.py          # refreshes data/train.jsonl and data/test.jsonl
python scripts/train_tinker.py     # LoRA on Qwen/Qwen2.5-7B-Instruct, writes TINKER_MODEL_NAME
python scripts/eval_analyzers.py   # heuristic vs ollama vs tinker -> eval_results.json
```

The training script asks for `Qwen/Qwen2.5-7B-Instruct`. As of this writing Tinker rejects that base model (`400 not supported`) and the script continues on `Qwen/Qwen3-8B`, the closest supported open instruct model, with the same prompt and the same jsonl.

`eval_results.json` is tag-level precision, recall, and the false-negative rate on `no_metrics`. If Ollama or Tinker is not configured, that row is the heuristic fallback and the file says so. Do not treat a fallback row as a fine-tune result.

Held-out run after the Qwen3-8B LoRA (10 examples, 0 Tinker fallbacks):

| analyzer | precision | recall | no_metrics false-negative rate |
| --- | --- | --- | --- |
| heuristic | 0.35 | 0.50 | 0.50 |
| ollama qwen2.5:7b | 0.35 | 0.50 | 0.50 (fallback on all 10; Ollama was not running) |
| tinker LoRA | 0.50 | 0.429 | 0.50 |

The fine-tune cut false tags (13 → 6) and raised precision. Recall dipped (7 → 8 false negatives), and it still missed half of the `no_metrics` answers. Training loss fell from 10.23 to 0.00 over 100 steps on 18 conversations, so the weights fit that small set tightly.

The training target is the same strict JSON the app asks for at inference time, including evidence quotes copied from her answer. A few labels disagree with the keyword rules on purpose: "2 years of experience" contains a digit and is still `no_metrics`, because it is not an outcome.

## Deploy

`render.yaml` is an always-on web service (`plan: starter`) running `uvicorn app.main:app`. Set `BACKBOARD_API_KEY` in the Render dashboard. The committed default analyzer is `heuristic`, so the site works before any key is added.

```bash
docker build -t duster .
docker run --env-file .env -p 8000:8000 duster
```

SQLite on Render's disk is fine for one person. It does not survive a redeploy that replaces the disk. For a demo, re-run is not required; seed from the shell if the disk is empty.

## Tags

`rambling`, `no_metrics`, `no_structure`, `weak_closing`, `too_short`, `off_topic`, `unclear`. Severity is `low`, `med`, or `high`.

## Tests

```bash
python -m unittest tests.test_core
```

## License

MIT. See `LICENSE`.
