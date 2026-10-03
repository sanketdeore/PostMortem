"""Labeled interview answers for the LoRA set and the held-out eval.

Run this file to regenerate data/train.jsonl and data/test.jsonl.
The assistant turn is the strict JSON the analyzer is trained to emit.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.analyzers.prompts import system_prompt, user_prompt
from app.models import QA

ROOT = Path(__file__).resolve().parent

# Shared preference block so train-time prompts match inference.
PREFS = {
    "target_role": "backend engineer",
    "domain": "product infrastructure",
    "weak_areas": "no_metrics, rambling",
}


def _row(question: str, answer: str, tags: list[dict]) -> dict:
    qa = QA(session_id=0, question_text=question, her_answer=answer)
    return {
        "messages": [
            {"role": "system", "content": system_prompt(PREFS)},
            {"role": "user", "content": user_prompt(qa)},
            {"role": "assistant", "content": json.dumps({"tags": tags}, ensure_ascii=False)},
        ]
    }


def _tag(tag: str, severity: str, quote: str) -> dict:
    return {"tag": tag, "severity": severity, "evidence_quote": quote}


# Human labels. A few disagree with the keyword heuristic on purpose:
# a digit that is not an outcome ("2 years") is still no_metrics, and
# the word "first" used as filler is still no_structure.
TRAIN = [
    _row(
        "Tell me about a time you sped up a service.",
        "Um, it was slow, you know, and I kind of looked around and basically made it better I think.",
        [
            _tag("rambling", "high", "Um, it was slow, you know"),
            _tag("no_metrics", "high", "made it better I think"),
            _tag("weak_closing", "med", "made it better I think"),
        ],
    ),
    _row(
        "How would you design a rate limiter?",
        "I would design the limiter as a token bucket in Redis: 100 requests per minute per API key, and return 429 when the bucket is empty.",
        [],
    ),
    _row(
        "Describe a production incident you owned.",
        "The checkout error rate hit 8 percent. I rolled back the last deploy in 12 minutes and the error rate went back to 0.4 percent.",
        [],
    ),
    _row(
        "Why do you want this role?",
        "I have 2 years of experience and I am a hard worker who is passionate about technology and growth.",
        [
            _tag("no_metrics", "high", "I have 2 years of experience"),
            _tag("off_topic", "med", "passionate about technology and growth"),
        ],
    ),
    _row(
        "Tell me about a disagreement with a teammate.",
        "We disagreed about the schema. I wrote a one-page comparison, we picked the smaller table, and the migration finished in two days instead of the two weeks the wider design needed.",
        [],
    ),
    _row(
        "What is your greatest weakness?",
        "I work too hard.",
        [_tag("too_short", "high", "I work too hard.")],
    ),
    _row(
        "Walk me through a project you are proud of.",
        "So like the project was, um, a dashboard and I did the backend and also some of the frontend and there were meetings and then we shipped it and people said it was nice and I learned a lot which was good.",
        [
            _tag("rambling", "high", "So like the project was, um"),
            _tag("no_metrics", "high", "people said it was nice"),
            _tag("weak_closing", "med", "I learned a lot which was good"),
        ],
    ),
    _row(
        "How do you test a payments change?",
        "First I write a failing test for the rounding bug. Then I fix the integer-cents path. The result was zero mismatched settlements across 10,000 replayed orders.",
        [],
    ),
    _row(
        "Tell me about a time you mentored someone.",
        "I helped them.",
        [_tag("too_short", "high", "I helped them.")],
    ),
    _row(
        "How did you cut page-load time?",
        "The first thing that comes to mind is, um, we kind of looked at the waterfall and you know it got better after some changes I made to the bundle.",
        [
            _tag("rambling", "high", "um, we kind of looked"),
            _tag("no_metrics", "high", "it got better"),
            _tag("no_structure", "med", "The first thing that comes to mind"),
        ],
    ),
    _row(
        "Tell me about a time you missed a deadline.",
        "The migration slipped because the backfill was single-threaded. I split it into 8 workers, and the job that had been at 30 hours finished in 4.",
        [],
    ),
    _row(
        "Describe your ideal team.",
        "Nice people, good snacks, not too many meetings, and a manager who is chill.",
        [
            _tag("off_topic", "med", "good snacks, not too many meetings"),
            _tag("no_structure", "low", "Nice people, good snacks"),
        ],
    ),
    _row(
        "How would you investigate a memory leak?",
        "I would take a heap snapshot, compare it to one from an hour earlier, find the object that grew, and confirm the fix by watching RSS stay flat for 24 hours.",
        [],
    ),
    _row(
        "Tell me about a time you influenced without authority.",
        "I talked to people and eventually they agreed with me because my idea was better and that was the outcome.",
        [
            _tag("no_metrics", "high", "they agreed with me"),
            _tag("no_structure", "med", "I talked to people and eventually they agreed"),
            _tag("weak_closing", "med", "that was the outcome"),
        ],
    ),
    _row(
        "What would you do in the first 30 days?",
        "Listen.",
        [_tag("too_short", "high", "Listen.")],
    ),
    _row(
        "Tell me about a time data changed your mind.",
        "I thought the cache was the bottleneck. The trace showed 70 percent of the time was in a single SQL join, so I rewrote that query and p95 dropped from 900 ms to 120 ms.",
        [],
    ),
    _row(
        "How do you handle on-call?",
        "It is really really really important to be available and to look at the alerts and to think about the system and to be a team player and to stay calm and to communicate and to write notes and to follow up and to be kind and to be fast and to be thorough and to be humble and to be present and to be ready and to be careful and to be curious and to be steady.",
        [
            _tag("rambling", "high", "really really really important"),
            _tag("unclear", "med", "be available and to look at the alerts"),
            _tag("no_metrics", "med", "be a team player"),
        ],
    ),
    _row(
        "Tell me about a tradeoff you made.",
        "We could store the audit log forever or for 90 days. I picked 90 days because legal only needed a quarter, and storage dropped by 60 percent.",
        [],
    ),
]

TEST = [
    _row(
        "Tell me about a time you reduced errors.",
        "There were errors in the form and I changed the validation so fewer people hit the error and the support channel got quieter.",
        [
            _tag("no_metrics", "high", "the support channel got quieter"),
            _tag("no_structure", "med", "There were errors in the form"),
        ],
    ),
    _row(
        "Why clinical software?",
        "I like building nice interfaces and I have used a lot of consumer apps so I would enjoy the design reviews.",
        [_tag("off_topic", "high", "I like building nice interfaces")],
    ),
    _row(
        "How do you debug an incident?",
        "I check the logs and then I fix it.",
        [
            _tag("too_short", "high", "I check the logs and then I fix it."),
            _tag("no_metrics", "high", "I check the logs and then I fix it."),
        ],
    ),
    _row(
        "Design a job queue.",
        "First, producers publish JSON to a durable topic. Then workers pull in batches of 50. The result was a 3x throughput increase on the nightly import, from 20 minutes to 7.",
        [],
    ),
    _row(
        "Tell me about a time you were wrong.",
        "I have been wrong a few times in my 3 years here but I always learn and grow and stay positive about the experience.",
        [
            _tag("no_metrics", "high", "I have been wrong a few times"),
            _tag("weak_closing", "med", "stay positive about the experience"),
        ],
    ),
    _row(
        "Describe a system you scaled.",
        "We were at 200 requests per second and the database was the wall. I added a read replica and p99 went from 1.2 seconds to 180 milliseconds.",
        [],
    ),
    _row(
        "What do you do when requirements change mid-sprint?",
        "Um, I kind of, you know, just go with the flow and basically redo the ticket and like hope the estimate still works.",
        [
            _tag("rambling", "high", "Um, I kind of, you know"),
            _tag("no_structure", "med", "just go with the flow"),
            _tag("weak_closing", "med", "hope the estimate still works"),
        ],
    ),
    _row(
        "Tell me about a time you documented a decision.",
        "I wrote it down.",
        [_tag("too_short", "high", "I wrote it down.")],
    ),
    _row(
        "How would you roll out a breaking API change?",
        "I would ship the new field behind a version header, dual-write for two weeks, watch the old-client count hit zero, and then delete the old field.",
        [],
    ),
    _row(
        "Tell me about a time you prioritized.",
        "The first idea was to rebuild the service, which would have been cool, but I kind of just patched the hot path and it seemed fine.",
        [
            _tag("no_metrics", "high", "it seemed fine"),
            _tag("no_structure", "med", "The first idea was to rebuild"),
            _tag("weak_closing", "med", "it seemed fine"),
        ],
    ),
]


def write() -> None:
    for name, rows in (("train.jsonl", TRAIN), ("test.jsonl", TEST)):
        path = ROOT / name
        path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    write()
