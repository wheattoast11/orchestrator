"""
Test Suite for AI Orchestrator
============================
Comprehensive tests for the AI Orchestrator package.
"""

import pytest
import asyncio
from ai_orchestrator import AIOrchestrator, AgentConfig, OrchestrationMode
from unittest.mock import Mock, patch
import json

@pytest.fixture
def orchestrator():
    return AIOrchestrator(api_key="test-key")

@pytest.fixture
def mock_llm_response():
    return json.dumps({
        "status": "success",
        "result": "test result",
        "confidence": 0.95
    })

@pytest.mark.asyncio
async def test_create_session(orchestrator):
    session = await orchestrator.create_session()
    assert session.session_id is not None
    assert session.mode == OrchestrationMode.ADAPTIVE

@pytest.mark.asyncio
async def test_process_input(orchestrator, mock_llm_response):
    with patch.object(orchestrator, '_call_llm', return_value=mock_llm_response):
        result = await orchestrator.process_input(
            session_id="test-session",
            user_input="test input"
        )
        assert result is not None
        assert isinstance(result, dict)

@pytest.mark.asyncio
async def test_parallel_execution(orchestrator, mock_llm_response):
    with patch.object(orchestrator, '_call_llm', return_value=mock_llm_response):
        strategy = {
            "mode": "parallel",
            "tasks": [{"id": 1}, {"id": 2}]
        }
        result = await orchestrator._parallel_execution(
            strategy,
            await orchestrator.create_session()
        )
        assert result is not None
        assert isinstance(result, dict)

@pytest.mark.asyncio
async def test_sequential_execution(orchestrator, mock_llm_response):
    with patch.object(orchestrator, '_call_llm', return_value=mock_llm_response):
        strategy = {
            "mode": "sequential",
            "tasks": [{"id": 1}, {"id": 2}]
        }
        result = await orchestrator._sequential_execution(
            strategy,
            await orchestrator.create_session()
        )
        assert result is not None
        assert isinstance(result, dict)

@pytest.mark.asyncio
async def test_error_handling(orchestrator):
    with patch.object(orchestrator, '_call_llm', side_effect=Exception("Test error")):
        result = await orchestrator.process_input(
            session_id="test-session",
            user_input="test input"
        )
        assert result["status"] == "error"
        assert "error" in result

@pytest.mark.asyncio
async def test_session_management(orchestrator):
    session_id = "test-session"
    await orchestrator.create_session(session_id)
    
    info = orchestrator.get_session_info(session_id)
    assert info is not None
    assert info["session_id"] == session_id
    
    success = await orchestrator.close_session(session_id)
    assert success is True
    
    info = orchestrator.get_session_info(session_id)
    assert info is None

@pytest.mark.asyncio
async def test_custom_agent_config(orchestrator):
    custom_agent = AgentConfig(
        role="test_agent",
        capabilities=["test"],
        temperature=0.5
    )
    
    orchestrator.custom_agents["test"] = custom_agent
    assert "test" in orchestrator.custom_agents
    assert orchestrator.custom_agents["test"].role == "test_agent"

@pytest.mark.asyncio
async def test_cleanup_old_sessions(orchestrator):
    await orchestrator.create_session("test-session-1")
    await orchestrator.create_session("test-session-2")
    
    # Mock session age
    for session in orchestrator.active_sessions.values():
        session.created_at = session.created_at.replace(
            year=session.created_at.year - 1
        )
    
    closed_count = await orchestrator.cleanup_old_sessions(max_age_hours=1)
    assert closed_count == 2
    assert len(orchestrator.active_sessions) == 0

@pytest.mark.parametrize("execution_mode", [
    OrchestrationMode.SEQUENTIAL,
    OrchestrationMode.PARALLEL,
    OrchestrationMode.ADAPTIVE
])
@pytest.mark.asyncio
async def test_execution_modes(orchestrator, mock_llm_response, execution_mode):
    with patch.object(orchestrator, '_call_llm', return_value=mock_llm_response):
        result = await orchestrator.process_input(
            session_id="test-session",
            user_input="test input",
            context_updates={"mode": execution_mode.value}
        )
        assert result is not None
        assert isinstance(result, dict)