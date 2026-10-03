"""CLI: python -m app.seed

Inserts two fake interviews with questions, answers, and weakness tags.
Safe to run twice: it does nothing if a session already exists.
"""

from datetime import date, timedelta

from sqlmodel import Session, select

from .db import create_db_and_tables, engine
from .models import InterviewSession, QA, WeaknessTag
from .preferences import get_prefs, set_pref


def _add(db: Session, interview: InterviewSession, pairs: list[dict]) -> None:
    db.add(interview)
    db.commit()
    db.refresh(interview)
    for pair in pairs:
        qa = QA(
            session_id=interview.id,
            question_text=pair["question"],
            her_answer=pair["answer"],
            duration_sec=pair.get("duration_sec"),
        )
        db.add(qa)
        db.commit()
        db.refresh(qa)
        for tag in pair["tags"]:
            db.add(
                WeaknessTag(
                    qa_id=qa.id,
                    tag=tag["tag"],
                    severity=tag["severity"],
                    evidence_quote=tag["evidence"],
                )
            )
    db.commit()


def seed() -> None:
    create_db_and_tables()
    prefs = get_prefs()
    if not prefs.get("target_role"):
        set_pref("target_role", "backend engineer")

    with Session(engine) as db:
        existing = db.exec(select(InterviewSession)).first()
        if existing is not None:
            print("Sessions already exist — nothing to seed.")
            return

        today = date.today()
        _add(
            db,
            InterviewSession(
                company="Northwind",
                role="Backend engineer",
                date=today - timedelta(days=18),
                outcome="rejected",
                notes="Onsite. No recruiter notes afterward.",
            ),
            [
                {
                    "question": "Tell me about a time you improved a slow endpoint.",
                    "answer": (
                        "Um, so, like, we had this endpoint that was kind of slow and I, "
                        "you know, looked at it and basically I think I cached something "
                        "and it felt faster after that. I mean it was a good change."
                    ),
                    "duration_sec": 95,
                    "tags": [
                        {
                            "tag": "no_metrics",
                            "severity": "high",
                            "evidence": "it felt faster after that",
                        },
                        {
                            "tag": "rambling",
                            "severity": "high",
                            "evidence": "Um, so, like, we had this endpoint",
                        },
                        {
                            "tag": "weak_closing",
                            "severity": "med",
                            "evidence": "I mean it was a good change.",
                        },
                    ],
                },
                {
                    "question": "How do you debug a production incident?",
                    "answer": "I check the logs and then I fix it.",
                    "duration_sec": 12,
                    "tags": [
                        {
                            "tag": "too_short",
                            "severity": "high",
                            "evidence": "I check the logs and then I fix it.",
                        },
                        {
                            "tag": "no_metrics",
                            "severity": "high",
                            "evidence": "I check the logs and then I fix it.",
                        },
                    ],
                },
                {
                    "question": "Describe a design you led.",
                    "answer": (
                        "We needed a queue because the API was getting traffic and I "
                        "picked a tool the team already knew and we rolled it out to "
                        "the checkout path over a couple of weeks and people used it."
                    ),
                    "duration_sec": 70,
                    "tags": [
                        {
                            "tag": "no_structure",
                            "severity": "med",
                            "evidence": "We needed a queue because the API was getting traffic",
                        },
                        {
                            "tag": "no_metrics",
                            "severity": "high",
                            "evidence": "people used it.",
                        },
                    ],
                },
            ],
        )
        _add(
            db,
            InterviewSession(
                company="Lumen Health",
                role="Full-stack engineer",
                date=today - timedelta(days=4),
                outcome="ghosted",
                notes="Recruiter screen. No reply after the thank-you note.",
            ),
            [
                {
                    "question": "Why do you want to work on clinical software?",
                    "answer": (
                        "I like building nice interfaces and I have used a lot of "
                        "consumer apps so I think I would enjoy the design reviews "
                        "and the front-end polish on a product team."
                    ),
                    "duration_sec": 40,
                    "tags": [
                        {
                            "tag": "off_topic",
                            "severity": "high",
                            "evidence": "I like building nice interfaces",
                        },
                        {
                            "tag": "unclear",
                            "severity": "med",
                            "evidence": "I think I would enjoy the design reviews",
                        },
                    ],
                },
                {
                    "question": "Tell me about a time you reduced errors in a workflow.",
                    "answer": (
                        "There were errors in the form and I changed the validation "
                        "so fewer people hit the error and the support channel got quieter."
                    ),
                    "duration_sec": 55,
                    "tags": [
                        {
                            "tag": "no_metrics",
                            "severity": "high",
                            "evidence": "the support channel got quieter",
                        },
                        {
                            "tag": "no_structure",
                            "severity": "med",
                            "evidence": "There were errors in the form and I changed the validation",
                        },
                    ],
                },
            ],
        )
    print("Seeded 2 sessions (Northwind rejected, Lumen Health ghosted).")


if __name__ == "__main__":
    seed()
