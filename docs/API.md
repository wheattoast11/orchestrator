# AI Orchestrator API Reference

## Core Classes

### AIOrchestrator

The main class for orchestrating AI tasks.

```python
class AIOrchestrator:
    def __init__(
        self,
        api_key: str,
        base_config: Optional[Dict[str, Any]] = None,
        custom_agents: Optional[Dict[str, AgentConfig]] = None
    )
```

#### Parameters:
- `api_key` (str): API key for LLM service
- `base_config` (Optional[Dict]): Base configuration for the orchestrator
- `custom_agents` (Optional[Dict]): Custom agent configurations

#### Methods:

##### process_input
```python
async def process_input(
    self,
    session_id: str,
    user_input: Any,
    context_updates: Optional[Dict] = None
) -> Dict
```
Process user input and orchestrate AI responses.

##### create_session
```python
async def create_session(
    self,
    session_id: Optional[str] = None,
    initial_context: Optional[Dict] = None
) -> ExecutionContext
```
Create a new orchestration session.

##### get_session_info
```python
def get_session_info(
    self,
    session_id: str
) -> Optional[Dict]
```
Get information about a session.

##### close_session
```python
async def close_session(
    self,
    session_id: str
) -> bool
```
Close and cleanup a session.

### AgentConfig

Configuration class for AI agents.

```python
@dataclass
class AgentConfig:
    role: str
    capabilities: List[str]
    model: str = "claude-3-5-sonnet-20241022"
    temperature: float = 0.7
    context_window: int = 100000
    max_tokens: int = 4000
```

### OrchestrationMode

Enum for execution modes.

```python
class OrchestrationMode(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"
```

## Usage Examples

### Basic Usage
```python
from ai_orchestrator import AIOrchestrator

orchestrator = AIOrchestrator(api_key="your-api-key")

# Process input
result = await orchestrator.process_input(
    session_id="unique-session-id",
    user_input="Your input text",
    context_updates={"domain": "analysis"}
)
```

### Custom Agent Configuration
```python
from ai_orchestrator import AIOrchestrator, AgentConfig

custom_agent = AgentConfig(
    role="analyst",
    capabilities=["data_analysis", "visualization"],
    temperature=0.3
)

orchestrator = AIOrchestrator(
    api_key="your-api-key",
    custom_agents={"analyst": custom_agent}
)
```

### Session Management
```python
# Create session
session = await orchestrator.create_session(
    session_id="my-session",
    initial_context={"domain": "analysis"}
)

# Get session info
info = orchestrator.get_session_info("my-session")

# Close session
await orchestrator.close_session("my-session")
```

## Error Handling

The library provides comprehensive error handling:

1. Retries with backoff for transient errors
2. Automatic error recovery attempts
3. Detailed error information in responses
4. Session cleanup for error states

## Best Practices

1. Use session management for related operations
2. Implement proper error handling
3. Configure custom agents for specific tasks
4. Monitor and cleanup old sessions
5. Use appropriate execution modes for tasks