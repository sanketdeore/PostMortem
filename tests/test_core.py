"""Parser, heuristic, and streak checks. No network."""

import os
import unittest
from datetime import date, datetime, timezone

os.environ["DATABASE_URL"] = "sqlite:///./test_duster.db"

from sqlmodel import Session, select

from app.analyzers.coaching import score_from_tags
from app.analyzers.heuristic import HeuristicAnalyzer
from app.analyzers.prompts import extract_json
from app.analyzers.schemas import TagDraft, snap_quote
from app.db import create_db_and_tables, engine
from app.models import Drill
from app.open_models import assert_open_provider
from app.parse import parse_debrief
from app.patterns import drill_streak


class ParseTests(unittest.TestCase):
    def test_messy_blocks(self):
        text = """
- Q: Tell me about a failure
that took too long
A: um we fixed it later (45 sec)

Question: Design a limiter.
Answer: I would use Redis.
"""
        pairs = parse_debrief(text)
        self.assertEqual(len(pairs), 2)
        self.assertIn("too long", pairs[0].question_text)
        self.assertEqual(pairs[0].duration_sec, 45)
        self.assertNotIn("45", pairs[0].her_answer)
        self.assertEqual(pairs[1].question_text, "Design a limiter.")

    def test_no_markers(self):
        self.assertEqual(parse_debrief("we just talked for a while"), [])


class HeuristicTests(unittest.TestCase):
    def test_no_metrics_and_rambling(self):
        from app.models import QA

        qa = QA(
            session_id=0,
            question_text="Tell me about a time you sped up a service.",
            her_answer="Um, it was slow, you know, and I kind of looked around and basically made it better I think.",
        )
        tags = {tag.tag for tag in HeuristicAnalyzer().analyze(qa, {})}
        self.assertIn("no_metrics", tags)
        self.assertIn("rambling", tags)

    def test_strong_answer_is_quiet(self):
        from app.models import QA

        qa = QA(
            session_id=0,
            question_text="How would you design a rate limiter?",
            her_answer=(
                "I would design the limiter as a token bucket in Redis: "
                "100 requests per minute per API key, and return 429 when the bucket is empty."
            ),
        )
        self.assertEqual(HeuristicAnalyzer().analyze(qa, {}), [])

    def test_years_of_experience_is_not_an_outcome(self):
        """A digit that isn't a result still trips the keyword rule.

        The held-out set labels this no_metrics. The heuristic misses it
        because a digit is present. That gap is what the fine-tune is for.
        """
        from app.models import QA

        qa = QA(
            session_id=0,
            question_text="Why do you want this role?",
            her_answer="I have 2 years of experience and I am a hard worker.",
        )
        tags = {tag.tag for tag in HeuristicAnalyzer().analyze(qa, {})}
        self.assertNotIn("no_metrics", tags)

    def test_score_bottoms_at_zero(self):
        tags = [
            TagDraft(tag="no_metrics", severity="high", evidence_quote="x"),
            TagDraft(tag="rambling", severity="high", evidence_quote="x"),
            TagDraft(tag="too_short", severity="high", evidence_quote="x"),
            TagDraft(tag="unclear", severity="high", evidence_quote="x"),
        ]
        self.assertEqual(score_from_tags(tags), 0)


class QuoteTests(unittest.TestCase):
    def test_snap_is_verbatim(self):
        answer = "I cached the query and it felt faster."
        self.assertEqual(snap_quote(answer, "it felt faster"), "it felt faster")
        self.assertIn("cached", snap_quote(answer, "not in the answer"))

    def test_fenced_json(self):
        raw = 'Sure\n```json\n{"tags": []}\n```'
        self.assertEqual(extract_json(raw), '{"tags": []}')


class ProviderTests(unittest.TestCase):
    def test_closed_provider_is_refused(self):
        with self.assertRaises(RuntimeError):
            assert_open_provider("openai")
        self.assertEqual(assert_open_provider("openrouter"), "openrouter")


class StreakTests(unittest.TestCase):
    def setUp(self):
        create_db_and_tables()
        with Session(engine) as db:
            for row in db.exec(select(Drill)).all():
                db.delete(row)
            db.commit()

    def test_streak_counts_through_yesterday(self):
        with Session(engine) as db:
            for day in (1, 2):
                db.add(
                    Drill(
                        question_text="q",
                        her_answer="a",
                        score=5,
                        feedback="f",
                        practiced_at=datetime(2026, 10, day, tzinfo=timezone.utc),
                    )
                )
            db.commit()
            # She has not practiced on the 3rd yet. The streak still holds.
            self.assertEqual(drill_streak(db, today=date(2026, 10, 3)), 2)
            self.assertEqual(drill_streak(db, today=date(2026, 10, 4)), 0)


if __name__ == "__main__":
    unittest.main()
