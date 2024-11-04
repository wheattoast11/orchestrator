# 🤖 AIOrchestrator

[![PyPI version](https://badge.fury.io/py/ai-orchestrator.svg)](https://badge.fury.io/py/ai-orchestrator)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

AI Orchestrator is a powerful, easy-to-use library that helps you integrate and manage AI capabilities in your applications. It provides intelligent orchestration of AI tasks using advanced language models like Claude 3.5 Sonnet.

## 🌟 Features

- 🔌 **Plug-and-Play Integration**: Easy integration with existing applications
- 🧠 **Intelligent Task Management**: Automatic task analysis and orchestration
- 🔄 **Flexible Execution Modes**: Sequential, parallel, or adaptive execution
- 💾 **Built-in State Management**: Session-based context and history tracking
- 🛠️ **Customizable**: Extensible for different AI models and use cases
- 🔐 **Error Handling**: Robust error recovery mechanisms
- 📈 **Scalable**: Async support for high-performance applications

## 🚀 Quick Start

### Installation

```bash
pip install ai-orchestrator
```

### Basic Usage

```python
from ai_orchestrator import AIOrchestrator

# Initialize orchestrator
orchestrator = AIOrchestrator(
    api_key="your-api-key",
    base_config={
        "default_model": "claude-3-5-sonnet-20241022",
        "default_temperature": 0.7
    }
)

# Use in async context
async def process_task():
    result = await orchestrator.process_input(
        session_id="unique-session-id",
        user_input="Analyze this text for sentiment",
        context_updates={"domain": "sentiment-analysis"}
    )
    print(result)

# Run the task
import asyncio
asyncio.run(process_task())
```

### FastAPI Integration Example

```python
from fastapi import FastAPI
from ai_orchestrator import AIOrchestrator

app = FastAPI()
orchestrator = AIOrchestrator(api_key="your-api-key")

@app.post("/analyze")
async def analyze_text(text: str):
    result = await orchestrator.process_input(
        session_id="unique-session-id",
        user_input=text
    )
    return result
```

## 🎯 Use Cases

- 📊 **Data Analysis**: Intelligent processing of complex datasets
- 📝 **Content Generation**: Orchestrated content creation and modification
- 🔍 **Research Assistance**: Coordinated research and analysis tasks
- 🤝 **Customer Support**: Intelligent routing and handling of support queries
- 🎨 **Creative Tasks**: Coordinated creative content generation
- 📈 **Business Intelligence**: Complex analysis and reporting

## 🛠️ Advanced Configuration

### Custom Agent Configuration

```python
from ai_orchestrator import AIOrchestrator, AgentConfig

custom_agents = {
    "analyst": AgentConfig(
        role="data_analyst",
        capabilities=["statistical_analysis", "visualization"],
        model="claude-3-5-sonnet-20241022",
        temperature=0.3,
        context_window=100000,
        max_tokens=4000
    )
}

orchestrator = AIOrchestrator(
    api_key="your-api-key",
    custom_agents=custom_agents
)
```

### Execution Modes

```python
# Sequential Execution
result = await orchestrator.process_input(
    session_id="session-id",
    user_input="Complex task requiring steps",
    context_updates={"mode": "sequential"}
)

# Parallel Execution
result = await orchestrator.process_input(
    session_id="session-id",
    user_input="Multiple independent subtasks",
    context_updates={"mode": "parallel"}
)

# Adaptive Execution (Default)
result = await orchestrator.process_input(
    session_id="session-id",
    user_input="Dynamic task",
    context_updates={"mode": "adaptive"}
)
```

## 📚 Documentation

For detailed documentation, visit our [documentation site](https://ai-orchestrator.readthedocs.io/).

### Key Concepts

- **Sessions**: Maintain context and state across multiple interactions
- **Execution Modes**: Different strategies for task execution
- **Agents**: Specialized AI models for specific tasks
- **Context Management**: State and history tracking
- **Error Handling**: Recovery and fallback mechanisms

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Anthropic for Claude 3.5 Sonnet
- The open-source community

## 📮 Contact

- Create an issue for bug reports or feature requests
- Connect with me on [LinkedIn](https://www.linkedin.com/in/tej-desai-a4858b62/)
- Star the repository if you find it helpful!
