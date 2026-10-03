"""SQLite tables for one person's interviews, tags, and drills."""

from datetime import date, datetime, timezone

from sqlmodel import Field, SQLModel


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class InterviewSession(SQLModel, table=True):
    """One interview. Stored as the `session` table from the brief."""

    __tablename__ = "session"

    id: int | None = Field(default=None, primary_key=True)
    company: str
    role: str
    date: date
    outcome: str  # rejected | ghosted | offer | unknown
    notes: str = ""
    created_at: datetime = Field(default_factory=utcnow)


class QA(SQLModel, table=True):
    """One question and the answer she actually gave."""

    __tablename__ = "qa"

    id: int | None = Field(default=None, primary_key=True)
    session_id: int = Field(foreign_key="session.id", index=True)
    question_text: str
    her_answer: str
    duration_sec: int | None = None


class WeaknessTag(SQLModel, table=True):
    __tablename__ = "weakness_tag"

    id: int | None = Field(default=None, primary_key=True)
    qa_id: int = Field(foreign_key="qa.id", index=True)
    tag: str  # rambling | no_metrics | no_structure | weak_closing | too_short | off_topic | unclear
    severity: str  # low | med | high
    evidence_quote: str
    created_at: datetime = Field(default_factory=utcnow)


class Drill(SQLModel, table=True):
    __tablename__ = "drill"

    id: int | None = Field(default=None, primary_key=True)
    question_text: str
    her_answer: str
    score: int = Field(ge=0, le=10)
    feedback: str
    practiced_at: datetime = Field(default_factory=utcnow)
    created_at: datetime = Field(default_factory=utcnow)


class Preference(SQLModel, table=True):
    """Key-value store: target role, domain, self-reported weak areas, and ids."""

    __tablename__ = "preference"

    id: int | None = Field(default=None, primary_key=True)
    key: str = Field(unique=True, index=True)
    value: str
    updated_at: datetime = Field(default_factory=utcnow)
