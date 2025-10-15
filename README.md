# Dexter 🤖

> **Note**: This is a fork of [virattt/dexter](https://github.com/virattt/dexter) with modifications to support Anthropic Claude API instead of OpenAI.

An autonomous financial research agent that performs deep analysis using task planning, introspection, and real-time market data. Inspired by Claude Code, but built specifically for financial research.

## 🔧 Changes in This Fork

- **LLM Provider**: Migrated from OpenAI GPT-4 to Anthropic Claude Sonnet 4.5
- **Dependencies**: Replaced `langchain-openai` with `langchain-anthropic`
- **Model**: Now using `claude-sonnet-4-5-20250929` (the latest and most capable Claude model)
- **API Key**: Updated environment configuration to use `ANTHROPIC_API_KEY` instead of `OPENAI_API_KEY`

All core functionality remains unchanged - task planning, tool execution, and financial analysis work exactly the same way.
<img width="979" height="651" alt="Screenshot 2025-10-14 at 6 12 35 PM" src="https://github.com/user-attachments/assets/5a2859d4-53cf-4638-998a-15cef3c98038" />

## Overview

Dexter breaks down complex financial queries into actionable tasks, executes them intelligently using real-time data APIs, and synthesizes comprehensive, data-driven answers. Unlike simple chatbots, Dexter plans ahead, validates its progress, and iterates until it gathers the information needed to thoroughly answer your questions.

**Key Capabilities:**
- **Intelligent Task Planning**: Automatically decomposes complex queries into structured research steps
- **Autonomous Execution**: Selects and executes the right tools to gather financial data
- **Self-Validation**: Checks its own work and iterates until tasks are complete
- **Real-Time Financial Data**: Access to income statements, balance sheets, and cash flow statements
- **Safety Features**: Built-in loop detection and step limits to prevent runaway execution

[![Twitter Follow](https://img.shields.io/twitter/follow/virattt?style=social)](https://twitter.com/virattt)

## Quick Start

### Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- **Anthropic API key** (get one at [console.anthropic.com](https://console.anthropic.com))
- Financial Datasets API key (get one at [financialdatasets.ai](https://financialdatasets.ai))

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd dexter
```

2. Install dependencies with uv:
```bash
uv sync
```

3. Set up your environment variables:
```bash
# Copy the example environment file
cp env.example .env

# Edit .env and add your API keys
# ANTHROPIC_API_KEY=your-anthropic-api-key
# FINANCIAL_DATASETS_API_KEY=your-financial-datasets-api-key
```

### Usage

Run Dexter in interactive mode:
```bash
uv run dexter-agent
```

### Example Queries

Try asking Dexter questions like:
- "What was Apple's revenue growth over the last 4 quarters?"
- "Compare Microsoft and Google's operating margins for 2023"
- "Analyze Tesla's cash flow trends over the past year"
- "What is Amazon's debt-to-equity ratio based on recent financials?"

Dexter will automatically:
1. Break down your question into research tasks
2. Fetch the necessary financial data
3. Perform calculations and analysis
4. Provide a comprehensive, data-rich answer

## Architecture

Dexter uses a multi-agent architecture with specialized components:

- **Planning Agent**: Analyzes queries and creates structured task lists
- **Action Agent**: Selects appropriate tools and executes research steps
- **Validation Agent**: Verifies task completion and data sufficiency
- **Answer Agent**: Synthesizes findings into comprehensive responses

## Project Structure

```
dexter/
├── src/
│   ├── dexter/
│   │   ├── agent.py      # Main agent orchestration logic
│   │   ├── model.py      # LLM interface
│   │   ├── tools.py      # Financial data tools
│   │   ├── prompts.py    # System prompts for each component
│   │   ├── schemas.py    # Pydantic models
│   │   ├── utils/        # Utility functions
│   │   └── cli.py        # CLI entry point
├── pyproject.toml
└── uv.lock
```

## Configuration

Dexter supports configuration via the `Agent` class initialization:

```python
from dexter.agent import Agent

agent = Agent(
    max_steps=20,              # Global safety limit
    max_steps_per_task=5       # Per-task iteration limit
)
```

## How to Contribute

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

**Important**: Please keep your pull requests small and focused.  This will make it easier to review and merge.


## License

This project is licensed under the MIT License.

