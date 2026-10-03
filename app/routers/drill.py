"""Practice questions, scored with the active analyzer."""

import json

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlmodel import Session

from ..analyzers import get_analyzer
from ..analyzers.coaching import format_feedback, score_from_tags
from ..db import get_session
from ..memory import log_drill
from ..models import Drill, QA
from ..patterns import top_tags
from ..preferences import get_prefs, set_pref

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


def _queue(prefs: dict[str, str]) -> tuple[list[str], int]:
    try:
        questions = json.loads(prefs.get("drill_queue") or "[]")
    except json.JSONDecodeError:
        questions = []
    if not isinstance(questions, list):
        questions = []
    try:
        index = int(prefs.get("drill_index") or 0)
    except ValueError:
        index = 0
    return [str(item) for item in questions], index


def _panel(request: Request, db: Session, **extra):
    prefs = get_prefs()
    questions, index = _queue(prefs)
    current = questions[index] if 0 <= index < len(questions) else ""
    context = {
        "question": current,
        "index": index,
        "total": len(questions),
        "focus_tags": top_tags(db, 2),
        "role": prefs.get("target_role") or "backend engineer",
        "result": None,
        "done": bool(questions) and index >= len(questions),
    }
    context.update(extra)
    template = "partials/drill_panel.html" if request.headers.get("hx-request") else "drill.html"
    return templates.TemplateResponse(request, template, context)


@router.get("/drill", response_class=HTMLResponse)
def drill_page(request: Request, db: Session = Depends(get_session)):
    return _panel(request, db)


@router.post("/drill/start", response_class=HTMLResponse)
def drill_start(
    request: Request,
    focus: str = Form(""),
    db: Session = Depends(get_session),
):
    prefs = get_prefs()
    role = prefs.get("target_role") or "backend engineer"
    focus_tags = top_tags(db, 2) if focus == "on" else []
    analyzer = get_analyzer()
    questions = analyzer.generate_questions(role, focus_tags, 5)
    set_pref("drill_queue", json.dumps(questions))
    set_pref("drill_index", "0")
    return _panel(request, db)


@router.post("/drill/answer", response_class=HTMLResponse)
def drill_answer(
    request: Request,
    question_text: str = Form(""),
    her_answer: str = Form(""),
    db: Session = Depends(get_session),
):
    question_text = question_text.strip()
    her_answer = her_answer.strip()
    if not question_text or not her_answer:
        return _panel(request, db, error="Write an answer before saving.")

    prefs = get_prefs()
    qa = QA(session_id=0, question_text=question_text, her_answer=her_answer)
    analyzer = get_analyzer()
    tags = analyzer.analyze(qa, prefs)
    note = analyzer.coach(qa, tags, prefs)
    note.score = score_from_tags(tags)
    drill = Drill(
        question_text=question_text,
        her_answer=her_answer,
        score=note.score,
        feedback=format_feedback(note),
    )
    db.add(drill)
    db.commit()

    questions, index = _queue(prefs)
    set_pref("drill_index", str(index + 1))
    log_drill(question_text, note.score, [tag.tag for tag in tags])

    return _panel(
        request,
        db,
        result={
            "question": question_text,
            "answer": her_answer,
            "tags": tags,
            "score": note.score,
            "feedback": note.feedback,
            "strong_version": note.strong_version,
            "analyzer": analyzer.name,
            "fallback": analyzer.used_fallback,
        },
    )
