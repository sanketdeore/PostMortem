"""Read and write the single user's preference rows."""

from datetime import datetime, timezone

from sqlmodel import Session, select

from .db import engine
from .models import Preference

DEFAULTS: dict[str, str] = {
    "target_role": "backend engineer",
    "domain": "product infrastructure",
    "weak_areas": "no_metrics, rambling",
    "backboard_assistant_id": "",
    "backboard_thread_id": "",
    "backboard_summary": "",
    "drill_queue": "[]",
    "drill_index": "0",
}


def get_prefs() -> dict[str, str]:
    with Session(engine) as session:
        rows = session.exec(select(Preference)).all()
    prefs = dict(DEFAULTS)
    prefs.update({row.key: row.value for row in rows})
    return prefs


def set_pref(key: str, value: str) -> None:
    with Session(engine) as session:
        pref = session.exec(select(Preference).where(Preference.key == key)).first()
        if pref is None:
            session.add(Preference(key=key, value=value))
        else:
            pref.value = value
            pref.updated_at = datetime.now(timezone.utc)
            session.add(pref)
        session.commit()
