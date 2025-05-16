<div align="center">

# Thinking Agent

<div align="center" style="font-family: Arial, sans-serif;">
  <p>
    <a href="#overview" style="text-decoration: none; font-weight: bold;">Overview</a> •
    <a href="#getting-started" style="text-decoration: none; font-weight: bold;">Getting Started</a> •
    <a href="#evaluation" style="text-decoration: none; font-weight: bold;">Evaluation</a> •
    <a href="#results" style="text-decoration: none; font-weight: bold;">Results</a> •
    <a href="#citation" style="text-decoration: none; font-weight: bold;">Citation</a>
  </p>
</div>

</div>

# Overview

AgentThink is a systematic evaluation framework that automatically identifies failure patterns in large language models.

# Getting Started

First, clone the repository and install the required packages:

```shell
git clone https://github.com/AlexCuadron/ThinkingAgent.git
cd ThinkingAgent
pip install -r requirements.txt
```

The framework consists of two main components:

1. `format_message.py`: Processes and formats interaction logs into a standardized format
2. `analyze_agent_think.py`: Analyzes the formatted interactions and produces overthinking scores

## Configuration

The framework uses a `config.toml` file to configure the LLM settings:

```toml
[llm]
model = "claude-3-5-sonnet-20241022"
api_key = ""  # Set via environment variable LLM_API_KEY
temperature = 0.0
max_output_tokens = 4096
num_retries = 3
retry_min_wait = 4
retry_max_wait = 10
retry_multiplier = 2
```

# Evaluation

The evaluation process follows these steps:

1. **Data Collection**: Gather interaction logs from models performing agentic tasks
2. **Message Formatting**: Use `format_message.py` to standardize the interaction format
3. **Analysis**: Run `analyze_overthinking.py` to evaluate overthinking behaviors
4. **Scoring**: Generate scores (0-10) for each interaction based on:
   - 0-3: Always interacting with the environment
   - 4-7: Sometimes relies on internal reasoning
   - 8-10: Completely relies on internal reasoning

## Usage

To analyze a set of interactions:

```python
# Load configuration and initialize LLM
config = load_config()
llm = LLM(config)

# Analyze responses
analyze_responses("path/to/interactions", iteration_number=None)
```
