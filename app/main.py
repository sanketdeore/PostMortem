"""PostMortem web app."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .db import create_db_and_tables
from .routers import dashboard, debrief, drill


@asynccontextmanager
async def lifespan(_app: FastAPI):
    create_db_and_tables()
    yield


app = FastAPI(title="PostMortem", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.include_router(dashboard.router)
app.include_router(debrief.router)
app.include_router(drill.router)


@app.get("/health")
def health():
    return {"status": "ok"}
