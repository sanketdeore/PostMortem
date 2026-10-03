"""Counts, trends, and the drill streak for the home page."""

from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

from sqlmodel import Session, select

from .models import Drill, InterviewSession, QA, WeaknessTag


def _as_date(value: date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    return value


def pattern_rows(db: Session) -> list[dict]:
    """Tag frequency plus a trend arrow against the older half of her interviews."""
    tags = db.exec(select(WeaknessTag)).all()
    qas = {qa.id: qa for qa in db.exec(select(QA)).all()}
    sessions = db.exec(select(InterviewSession).order_by(InterviewSession.date)).all()
    if not tags or not sessions:
        return []

    midpoint = len(sessions) // 2
    older_ids = {s.id for s in sessions[:midpoint]} if midpoint else set()
    newer_ids = {s.id for s in sessions[midpoint:]}

    counts: dict[str, int] = defaultdict(int)
    session_hits: dict[str, set[int]] = defaultdict(set)
    older_hits: dict[str, int] = defaultdict(int)
    newer_hits: dict[str, int] = defaultdict(int)

    for tag in tags:
        qa = qas.get(tag.qa_id)
        if qa is None:
            continue
        counts[tag.tag] += 1
        session_hits[tag.tag].add(qa.session_id)
        if qa.session_id in older_ids:
            older_hits[tag.tag] += 1
        elif qa.session_id in newer_ids:
            newer_hits[tag.tag] += 1

    rows = []
    for tag, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        older = older_hits[tag]
        newer = newer_hits[tag]
        if not older_ids:
            arrow, trend = "·", "only one stretch of interviews so far"
        elif newer > older:
            arrow, trend = "↑", "showing up more in recent interviews"
        elif newer < older:
            arrow, trend = "↓", "showing up less in recent interviews"
        else:
            arrow, trend = "→", "holding steady"
        n_sessions = len(session_hits[tag])
        rows.append(
            {
                "tag": tag,
                "count": count,
                "sessions": n_sessions,
                "label": f"{tag} — {count}x in {n_sessions} session{'s' if n_sessions != 1 else ''}",
                "arrow": arrow,
                "trend": trend,
            }
        )
    return rows


def top_tags(db: Session, n: int = 2) -> list[str]:
    rows = pattern_rows(db)
    return [row["tag"] for row in rows[:n]]


def drill_streak(db: Session, today: date | None = None) -> int:
    """Consecutive practice days ending today, or yesterday if she hasn't gone yet today."""
    drills = db.exec(select(Drill)).all()
    days = {_as_date(drill.practiced_at) for drill in drills}
    if not days:
        return 0
    today = today or datetime.now(timezone.utc).date()
    cursor = today if today in days else today - timedelta(days=1)
    if cursor not in days:
        return 0
    streak = 0
    while cursor in days:
        streak += 1
        cursor -= timedelta(days=1)
    return streak


def session_cards(db: Session) -> list[dict]:
    sessions = db.exec(
        select(InterviewSession).order_by(InterviewSession.date.desc())
    ).all()
    qas = db.exec(select(QA)).all()
    tags = db.exec(select(WeaknessTag)).all()
    tags_by_qa: dict[int, list[str]] = defaultdict(list)
    for tag in tags:
        tags_by_qa[tag.qa_id].append(tag.tag)
    by_session: dict[int, list[QA]] = defaultdict(list)
    for qa in qas:
        by_session[qa.session_id].append(qa)

    cards = []
    for interview in sessions:
        pairs = by_session.get(interview.id or -1, [])
        tag_names = []
        for qa in pairs:
            tag_names.extend(tags_by_qa.get(qa.id or -1, []))
        cards.append(
            {
                "id": interview.id,
                "company": interview.company,
                "role": interview.role,
                "date": interview.date,
                "outcome": interview.outcome,
                "qa_count": len(pairs),
                "tags": sorted(set(tag_names)),
            }
        )
    return cards
