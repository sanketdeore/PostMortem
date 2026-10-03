"""Post-interview dump: form, optional voice note, parse, analyze."""

import os
import tempfile
from datetime import date
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlmodel import Session

from ..analyzers import get_analyzer
from ..db import get_session
from ..models import InterviewSession, QA, WeaknessTag
from ..parse import parse_debrief
from ..preferences import get_prefs

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

_OUTCOMES = {"rejected", "ghosted", "offer", "unknown"}
_AUDIO = {".m4a", ".mp3"}
_MAX_AUDIO = 25 * 1024 * 1024


def _page(request: Request, **context):
    context.setdefault("error", "")
    context.setdefault("result", None)
    context.setdefault("today", date.today().isoformat())
    template = "partials/debrief_root.html" if request.headers.get("hx-request") else "debrief.html"
    return templates.TemplateResponse(request, template, context)


@router.get("/debrief", response_class=HTMLResponse)
def debrief_form(request: Request):
    return _page(request)


@router.post("/debrief", response_class=HTMLResponse)
def debrief_submit(
    request: Request,
    company: str = Form(""),
    role: str = Form(""),
    outcome: str = Form("unknown"),
    dump: str = Form(""),
    notes: str = Form(""),
    interview_date: str = Form(""),
    db: Session = Depends(get_session),
):
    company = company.strip()
    role = role.strip()
    outcome = outcome.strip().lower()
    if outcome not in _OUTCOMES:
        outcome = "unknown"
    form = {
        "company": company,
        "role": role,
        "outcome": outcome,
        "dump": dump,
        "notes": notes,
        "interview_date": interview_date,
    }
    if not company or not role:
        return _page(request, error="Company and role are both required.", **form)
    pairs = parse_debrief(dump)
    if not pairs:
        return _page(
            request,
            error='No question blocks found. Start each one with "Q:" and "A:".',
            **form,
        )

    try:
        when = date.fromisoformat(interview_date) if interview_date else date.today()
    except ValueError:
        when = date.today()

    interview = InterviewSession(
        company=company,
        role=role,
        date=when,
        outcome=outcome,
        notes=notes.strip(),
    )
    db.add(interview)
    db.commit()
    db.refresh(interview)

    analyzer = get_analyzer()
    prefs = get_prefs()
    saved = []
    for pair in pairs:
        qa = QA(
            session_id=interview.id,
            question_text=pair.question_text,
            her_answer=pair.her_answer,
            duration_sec=pair.duration_sec,
        )
        db.add(qa)
        db.commit()
        db.refresh(qa)
        tags = analyzer.analyze(qa, prefs)
        for tag in tags:
            db.add(
                WeaknessTag(
                    qa_id=qa.id,
                    tag=tag.tag,
                    severity=tag.severity,
                    evidence_quote=tag.evidence_quote,
                )
            )
        db.commit()
        saved.append({"question": qa.question_text, "answer": qa.her_answer, "tags": tags})

    return _page(
        request,
        result={
            "company": company,
            "role": role,
            "outcome": outcome,
            "pairs": saved,
            "analyzer": analyzer.name,
            "fallback": analyzer.used_fallback,
        },
    )


@router.post("/debrief/transcribe", response_class=HTMLResponse)
async def transcribe(request: Request, audio: UploadFile = File(...)):
    """Transcribe on this machine and swap the textarea. The file is deleted."""
    suffix = Path(audio.filename or "").suffix.lower()
    if suffix not in _AUDIO:
        return templates.TemplateResponse(
            request,
            "partials/dump_field.html",
            {"dump": "", "error": "Upload an .m4a or .mp3 file."},
        )
    payload = await audio.read()
    if not payload:
        return templates.TemplateResponse(
            request,
            "partials/dump_field.html",
            {"dump": "", "error": "That file was empty."},
        )
    if len(payload) > _MAX_AUDIO:
        return templates.TemplateResponse(
            request,
            "partials/dump_field.html",
            {"dump": "", "error": "Audio must be under 25 MB."},
        )

    path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(payload)
            path = handle.name
        from ..transcribe import transcribe_path

        text = transcribe_path(path)
    except Exception:
        text = ""
        return templates.TemplateResponse(
            request,
            "partials/dump_field.html",
            {
                "dump": "",
                "error": "Transcription failed on this machine. You can still type the dump.",
            },
        )
    finally:
        if path and os.path.exists(path):
            os.remove(path)

    return templates.TemplateResponse(
        request,
        "partials/dump_field.html",
        {"dump": text, "error": "" if text else "No speech detected in that file."},
    )
