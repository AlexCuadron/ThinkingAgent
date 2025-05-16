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

AgentThink is a systematic evaluation framework that automatically rates overthinking behavior in large language models. The framework focuses on detecting when models prefer their internal reasoning chain over interacting with the environment, a critical issue in agentic tasks.

The framework evaluates three key aspects of overthinking:
1. **Analysis Paralysis**: When models focus on heavy planning instead of interacting with the environment
2. **Rogue Actions**: When models generate multiple actions without waiting for environment feedback
3. **Premature Disengagement**: When models conclude tasks without proper environment verification

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

# Results

Our framework has been used to analyze 4,018 trajectories from various models performing software engineering tasks. Key findings from our research:

1. **Performance Impact**:
   - Higher overthinking scores strongly correlate with decreased performance
   - Selecting solutions with lower overthinking scores improves model performance by ~30%
   - Computational costs can be reduced by 43% through overthinking mitigation

2. **Model Behavior Analysis**:
   - Reasoning models exhibit stronger tendencies toward overthinking compared to non-reasoning models
   - Three main patterns were identified:
     * Analysis Paralysis: Models focus on planning instead of action
     * Rogue Actions: Models execute multiple actions without waiting for feedback
     * Premature Disengagement: Models conclude tasks without proper verification

3. **Mitigation Strategies**:
   - Native function-calling capabilities can help reduce overthinking
   - Selective reinforcement learning shows promise in mitigating overthinking tendencies
   - Simple selection of lower overthinking score solutions provides significant improvements
