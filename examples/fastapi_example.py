"""
FastAPI Integration Example
=========================
Example of using AI Orchestrator with FastAPI.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
from ai_orchestrator import AIOrchestrator

app = FastAPI(title="AI Orchestrator API")
orchestrator = AIOrchestrator(api_key="your-api-key")

class ProcessRequest(BaseModel):
    text: str
    context: Optional[Dict] = None
    session_id: Optional[str] = None

@app.post("/process")
async def process_text(request: ProcessRequest):
    try:
        result = await orchestrator.process_input(
            session_id=request.session_id,
            user_input=request.text,
            context_updates=request.context
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/session/{session_id}")
async def get_session(session_id: str):
    info = orchestrator.get_session_info(session_id)
    if not info:
        raise HTTPException(status_code=404, detail="Session not found")
    return info

@app.delete("/session/{session_id}")
async def close_session(session_id: str):
    success = await orchestrator.close_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "success", "message": "Session closed"}