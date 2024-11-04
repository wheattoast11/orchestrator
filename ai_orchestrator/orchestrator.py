"""
AI Orchestrator - Core Implementation
Main module containing all core functionality for AI task orchestration.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OrchestrationMode(Enum):
    """Available execution modes for the orchestrator"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"

@dataclass
class AgentConfig:
    """Configuration for an AI agent"""
    role: str
    capabilities: List[str]
    model: str = "claude-3-5-sonnet-20241022"
    temperature: float = 0.7
    context_window: int = 100000
    max_tokens: int = 4000

@dataclass
class ExecutionContext:
    """Maintains state for orchestration flow"""
    session_id: str
    user_input: Any
    mode: OrchestrationMode
    metadata: Dict
    state: Dict
    history: List[Dict]
    created_at: datetime = datetime.now()

class AIOrchestrator:
    """
    Universal AI Orchestration Module
    Provides intelligent orchestration of AI tasks across applications.
    """
    
    DEFAULT_CONFIG = {
        "default_model": "claude-3-5-sonnet-20241022",
        "default_temperature": 0.7,
        "max_retries": 3,
        "timeout": 30,
        "max_parallel_tasks": 5
    }
    
    def __init__(
        self,
        api_key: str,
        base_config: Optional[Dict[str, Any]] = None,
        custom_agents: Optional[Dict[str, AgentConfig]] = None
    ):
        """
        Initialize the orchestrator with configuration.
        
        Args:
            api_key: API key for LLM service
            base_config: Optional base configuration
            custom_agents: Optional custom agent configurations
        """
        self.api_key = api_key
        self.config = {**self.DEFAULT_CONFIG, **(base_config or {})}
        self.custom_agents = custom_agents or {}
        self.active_sessions: Dict[str, ExecutionContext] = {}
        
        # Initialize prompts and validate configuration
        self._init_system()
        
    def _init_system(self) -> None:
        """Initialize system components and validate configuration"""
        self._init_base_prompts()
        self._validate_config()
        logger.info("AI Orchestrator initialized successfully")
        
    def _init_base_prompts(self) -> None:
        """Initialize core system prompts"""
        self.system_prompts = {
            "analyzer": """
            You are an input analyzer for an AI orchestration system.
            Analyze user input to determine:
            1. Primary intent and required capabilities
            2. Optimal execution mode
            3. Required agent configurations
            4. Risk factors and constraints
            
            Provide structured analysis following this schema:
            {
                "intent": str,
                "capabilities": List[str],
                "execution_mode": str,
                "agent_requirements": List[Dict],
                "risk_level": int,
                "confidence": float
            }
            """,
            
            "orchestrator": """
            You are Claude 3.5 Sonnet (20241022), acting as the primary orchestrator.
            Your role is to:
            1. Coordinate agent activities
            2. Maintain execution context
            3. Make strategic decisions
            4. Handle errors and recovery
            5. Optimize resource usage
            
            Follow provided execution patterns while adapting to context.
            """,
            
            "executor": """
            You are an AI task executor.
            Execute assigned tasks according to:
            1. Given constraints and requirements
            2. Specified execution mode
            3. Defined success criteria
            
            Maintain continuous feedback loop with orchestrator.
            """
        }

    def _validate_config(self) -> None:
        """Validate system configuration"""
        required_keys = ["default_model", "default_temperature", "max_retries"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")

    async def create_session(
        self,
        session_id: Optional[str] = None,
        initial_context: Optional[Dict] = None
    ) -> ExecutionContext:
        """
        Create new orchestration session.
        
        Args:
            session_id: Optional session ID (generated if not provided)
            initial_context: Optional initial context
            
        Returns:
            ExecutionContext: New session context
        """
        session_id = session_id or str(uuid.uuid4())
        context = ExecutionContext(
            session_id=session_id,
            user_input=None,
            mode=OrchestrationMode.ADAPTIVE,
            metadata=initial_context or {},
            state={},
            history=[]
        )
        self.active_sessions[session_id] = context
        logger.info(f"Created new session: {session_id}")
        return context

    async def process_input(
        self,
        session_id: str,
        user_input: Any,
        context_updates: Optional[Dict] = None
    ) -> Dict:
        """
        Process user input and orchestrate AI responses.
        
        Args:
            session_id: Session identifier
            user_input: User input to process
            context_updates: Optional context updates
            
        Returns:
            Dict: Processing results
        """
        try:
            # Get or create session
            session = self.active_sessions.get(session_id)
            if not session:
                session = await self.create_session(session_id)
            
            # Update context
            if context_updates:
                session.metadata.update(context_updates)
            session.user_input = user_input
            
            # Core processing pipeline
            analysis = await self._analyze_input(user_input, session)
            strategy = await self._determine_strategy(analysis, session)
            result = await self._execute_strategy(strategy, session)
            
            # Update history
            session.history.append({
                "timestamp": datetime.now().isoformat(),
                "input": user_input,
                "analysis": analysis,
                "strategy": strategy,
                "result": result
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing input: {str(e)}", exc_info=True)
            return await self._handle_error(e, session)

    async def _analyze_input(
        self,
        user_input: Any,
        context: ExecutionContext
    ) -> Dict:
        """
        Analyze user input to determine execution requirements.
        
        Args:
            user_input: User input to analyze
            context: Current execution context
            
        Returns:
            Dict: Analysis results
        """
        prompt = self._construct_prompt(
            "analyzer",
            user_input=user_input,
            context=context.metadata,
            history=context.history[-5:] if context.history else []
        )
        
        response = await self._call_llm(
            prompt,
            temperature=0.3  # Lower temperature for analysis
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            raise ValueError("Invalid analysis response format")

    async def _determine_strategy(
        self,
        analysis: Dict,
        context: ExecutionContext
    ) -> Dict:
        """
        Determine optimal execution strategy based on analysis.
        
        Args:
            analysis: Input analysis results
            context: Current execution context
            
        Returns:
            Dict: Execution strategy
        """
        prompt = self._construct_prompt(
            "orchestrator",
            analysis=analysis,
            context=context.metadata,
            history=context.history[-5:] if context.history else []
        )
        
        response = await self._call_llm(prompt)
        return json.loads(response)

    async def _execute_strategy(
        self,
        strategy: Dict,
        context: ExecutionContext
    ) -> Dict:
        """
        Execute determined strategy.
        
        Args:
            strategy: Strategy to execute
            context: Current execution context
            
        Returns:
            Dict: Execution results
        """
        execution_start = datetime.now()
        
        try:
            if strategy["mode"] == OrchestrationMode.PARALLEL.value:
                result = await self._parallel_execution(strategy, context)
            elif strategy["mode"] == OrchestrationMode.SEQUENTIAL.value:
                result = await self._sequential_execution(strategy, context)
            else:
                result = await self._adaptive_execution(strategy, context)
                
            execution_time = (datetime.now() - execution_start).total_seconds()
            
            return {
                "status": "success",
                "result": result,
                "execution_time": execution_time,
                "mode": strategy["mode"]
            }
            
        except Exception as e:
            logger.error(f"Strategy execution error: {str(e)}", exc_info=True)
            raise

    async def _parallel_execution(
        self,
        strategy: Dict,
        context: ExecutionContext
    ) -> Dict:
        """Handle parallel execution of tasks"""
        max_tasks = min(
            len(strategy["tasks"]),
            self.config["max_parallel_tasks"]
        )
        
        tasks = []
        for task in strategy["tasks"][:max_tasks]:
            tasks.append(self._execute_task(task, context))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return await self._aggregate_results(results, strategy, context)

    async def _sequential_execution(
        self,
        strategy: Dict,
        context: ExecutionContext
    ) -> Dict:
        """Handle sequential execution of tasks"""
        results = []
        for task in strategy["tasks"]:
            result = await self._execute_task(task, context)
            results.append(result)
            
            # Update context with intermediate results
            context.state["last_result"] = result
            
            # Check for early termination conditions
            if await self._should_terminate(result, strategy, context):
                break
                
        return await self._aggregate_results(results, strategy, context)

    async def _adaptive_execution(
        self,
        strategy: Dict,
        context: ExecutionContext
    ) -> Dict:
        """Handle adaptive execution based on context"""
        results = []
        current_strategy = strategy.copy()
        
        for task in strategy["tasks"]:
            # Analyze current state
            state_analysis = await self._analyze_state(context)
            
            # Adapt strategy based on analysis
            current_strategy = await self._adapt_strategy(
                current_strategy,
                state_analysis,
                context
            )
            
            # Execute adapted task
            result = await self._execute_task(task, context, current_strategy)
            results.append(result)
            
            # Update context
            context.state["last_result"] = result
            
        return await self._aggregate_results(results, current_strategy, context)

    async def _execute_task(
        self,
        task: Dict,
        context: ExecutionContext,
        strategy: Optional[Dict] = None
    ) -> Dict:
        """Execute individual task with error handling and retries"""
        retries = self.config["max_retries"]
        last_error = None
        
        while retries > 0:
            try:
                prompt = self._construct_prompt(
                    "executor",
                    task=task,
                    context=context.metadata,
                    strategy=strategy
                )
                
                result = await self._call_llm(
                    prompt,
                    temperature=task.get("temperature", self.config["default_temperature"])
                )
                
                return json.loads(result)
                
            except Exception as e:
                last_error = e
                retries -= 1
                if retries > 0:
                    await asyncio.sleep(1)  # Basic backoff
                    
        raise last_error or Exception("Task execution failed")

    async def _analyze_state(
        self,
        context: ExecutionContext
    ) -> Dict:
        """Analyze current execution state"""
        prompt = self._construct_prompt(
            "analyzer",
            context=context.metadata,
            state=context.state,
            history=context.history[-5:]
        )
        
        result = await self._call_llm(prompt, temperature=0.3)
        return json.loads(result)

    async def _adapt_strategy(
        self,
        strategy: Dict,
        state_analysis: Dict,
        context: ExecutionContext
    ) -> Dict:
        """Adapt strategy based on state analysis"""
        prompt = self._construct_prompt(
            "orchestrator",
            strategy=strategy,
            analysis=state_analysis,
            context=context.metadata
        )
        
        result = await self._call_llm(prompt)
        return json.loads(result)

    async def _aggregate_results(
        self,
        results: List[Dict],
        strategy: Dict,
        context: ExecutionContext
    ) -> Dict:
        """Aggregate results from multiple tasks"""
        return {
            "results": results,
            "summary": await self._generate_summary(results, strategy, context),
            "metadata": {
                "task_count": len(results),
                "success_count": sum(1 for r in results if r.get("status") == "success"),
                "strategy": strategy["mode"]
            }
        }

    async def _generate_summary(
        self,
        results: List[Dict],
        strategy: Dict,
        context: ExecutionContext
    ) -> Dict:
        """Generate summary of execution results"""
        prompt = self._construct_prompt(
            "analyzer",
            results=results,
            strategy=strategy,
            context=context.metadata
        )
        
        result = await self._call_llm(prompt, temperature=0.3)
        return json.loads(result)

    async def _should_terminate(
        self,
        result: Dict,
        strategy: Dict,
        context: ExecutionContext
    ) -> bool:
        """Check if execution should terminate early"""
        if result.get("status") == "error" and not strategy.get("continue_on_error"):
            return True
            
        if result.get("confidence", 1.0) >= strategy.get("confidence_threshold", 0.9):
            return True
            
        return False

    async def _handle_error(
        self,
        error: Exception,
        context: ExecutionContext
    ) -> Dict:
        """Handle errors during execution"""
        logger.error(f"Error in session {context.session_id}: {str(error)}", exc_info=True)
        
        try:
            error_prompt = self._construct_prompt(
                "orchestrator",
                error=str(error),
                context=context.metadata,
                history=context.history[-5:]
            )
            
            recovery_strategy = await self._call_llm(error_prompt)
            recovery_plan = json.loads(recovery_strategy)
            
            if recovery_plan.get("can_recover"):
                return await self._execute_recovery(recovery_plan, context)
            
        except Exception as recovery_error:
            logger.error(
                f"Error recovery failed: {str(recovery_error)}",
                exc_info=True
            )
            
        return {
            "status": "error",
            "error": str(error),
            "context": context.session_id,
            "timestamp": datetime.now().isoformat()
        }

    async def _execute_recovery(
        self,
        recovery_plan: Dict,
        context: ExecutionContext
    ) -> Dict:
        """Execute error recovery plan"""
        try:
            # Attempt recovery actions
            recovery_result = await self._execute_task(
                recovery_plan["recovery_task"],
                context
            )
            
            return {
                "status": "recovered",
                "original_error": recovery_plan.get("error"),
                "recovery_result": recovery_result,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Recovery execution failed: {str(e)}", exc_info=True)
            raise

    def _construct_prompt(
        self,
        prompt_type: str,
        **kwargs
    ) -> str:
        """Construct prompt based on type and parameters"""
        base_prompt = self.system_prompts[prompt_type]
        
        # Add dynamic content
        prompt_parts = [base_prompt]
        for key, value in kwargs.items():
            prompt_parts.append(f"\n{key}: {json.dumps(value, default=str)}")
            
        return "\n".join(prompt_parts)

    async def _call_llm(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Make API call to LLM"""
        # Implement actual API call here
        # This is a placeholder - implement based on chosen LLM API
        return "{}"  # Replace with actual API call

    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """Get information about a session"""
        session = self.active_sessions.get(session_id)
        if not session:
            return None
            
        return {
            "session_id": session.session_id,
            "created_at": session.created_at.isoformat(),
            "task_count": len(session.history),
            "last_activity": session.history[-1]["timestamp"] if session.history else None,
            "mode": session.mode.value,
            "metadata": session.metadata
        }

    async def close_session(self, session_id: str) -> bool:
        """Close and cleanup a session"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info(f"Closed session: {session_id}")
            return True
        return False

    async def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """Cleanup sessions older than specified age"""
        now = datetime.now()
        closed_count = 0
        
        for session_id, session in list(self.active_sessions.items()):
            age = (now - session.created_at).total_seconds() / 3600
            if age > max_age_hours:
                await self.close_session(session_id)
                closed_count += 1
                
        return closed_count