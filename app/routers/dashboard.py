"""Home page: weakness patterns, sessions, streak, Backboard memory."""

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlmodel import Session

from ..config import settings
from ..db import get_session
from ..memory import read_thread_summary
from ..patterns import drill_streak, pattern_rows, session_cards
from ..preferences import get_prefs, set_pref

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "landing.html", {})


@router.get("/patterns", response_class=HTMLResponse)
def patterns(request: Request, db: Session = Depends(get_session)):
    return templates.TemplateResponse(
        request,
        "home.html",
        {
            "patterns": pattern_rows(db),
            "sessions": session_cards(db),
            "streak": drill_streak(db),
            "memory": read_thread_summary(),
            "analyzer": settings.analyzer,
            "prefs": get_prefs(),
        },
    )


@router.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request):
    return templates.TemplateResponse(
        request,
        "settings.html",
        {"prefs": get_prefs(), "saved": False, "analyzer": settings.analyzer},
    )


@router.post("/settings", response_class=HTMLResponse)
def save_settings(
    request: Request,
    target_role: str = Form(""),
    domain: str = Form(""),
    weak_areas: str = Form(""),
):
    set_pref("target_role", target_role.strip() or "backend engineer")
    set_pref("domain", domain.strip())
    set_pref("weak_areas", weak_areas.strip())
    return templates.TemplateResponse(
        request,
        "settings.html",
        {"prefs": get_prefs(), "saved": True, "analyzer": settings.analyzer},
    )
