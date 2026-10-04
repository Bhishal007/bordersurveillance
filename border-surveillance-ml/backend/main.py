"""IBVAP Backend -- main FastAPI application."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.database.session import init_db
from backend.api import cameras, geofences, events
from backend.websocket import routes as ws_routes
from backend.agent import chat_with_agent # NEW: Omni-Agent Import

app = FastAPI(
    title="IBVAP Backend",
    description="Intelligent Border Video Analytics Platform",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(cameras.router)
app.include_router(geofences.router)
app.include_router(events.router)
app.include_router(ws_routes.router)

# NEW: Chat Route for React Dashboard
class ChatRequest(BaseModel):
    message: str

@app.post("/api/agent/chat")
def agent_chat_endpoint(req: ChatRequest):
    reply = chat_with_agent(req.message)
    return {"reply": reply}

@app.on_event("startup")
def on_startup():
    init_db()

@app.get("/")
def root():
    return {"status": "IBVAP backend online"}