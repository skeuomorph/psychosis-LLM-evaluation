# Evals for the safety of LLMs interacting with people experiencing psychosis

This project evaluates how different Large Language Models (LLMs) respond to psychosis scenarios and assesses their responses against predefined criteria to determine which models are most suitable for mental health crisis situations.

## Project Overview

The project follows this workflow:
1. **Prompt Generation**: Creates first-person prompts from psychosis clinical vignettes in `pipeline/src/data/psychosis_scenarios.csv`
2. **LLM Interaction**: Sends prompts to 4 different LLMs (ChatGPT, Claude, DeepSeek, Llama)
3. **Response Collection**: Gathers and stores responses from each model

> **Note:** Steps 4 (**Criteria Evaluation**) and 5 (**Scoring**) are not yet implemented.
4. **Criteria Evaluation**: Assesses responses against safety and helpfulness criteria
5. **Scoring**: Assigns scores based on performance against criteria

## Project Structure

```
mental-health-crises-LLM-evaluation/
├── .env
├── .gitignore
├── .python-version
├── main.py
├── poetry.lock
├── pyproject.toml
├── README.md
├── notebooks/
│   ├── base-prompt-variations.ipynb
│   ├── models-mental-health-cannot-help-list.ipynb
│   ├── README.md
│   └── scenario-base.ipynb
├── old_data/
│   ├── prompts_with_variations.csv
│   ├── psychosis_scenarios_base.csv
│   ├── psychosis_scenarios.csv
│   ├── README.md
│   └── cannot_help_list/
│       ├── chatgpt.csv
│       ├── claude.csv
│       ├── deepseek.csv
│       └── llama.csv
├── pipeline/
│   ├── generate_scenarios/
│   │   ├── generate_scenarios.py
│   │   └── __pycache__/
│   ├── responses_to_scenarios/
│   │   └── responses_to_scenarios.py
│   ├── src/
│   │   ├── data/
│   │   └── utils/
│   │       └── logger/
│   │           └── token_logger.py
└── output/
    └── token_usage.log
```

## Quick Start

1. **Setup Environment**:
   ```bash
   poetry install
   ```

2. **Generate Scenarios**:
   ```bash
   python pipeline/generate_scenarios/generate_scenarios.py
   ```

3. **Generate Model Responses**:
   ```bash
   python pipeline/responses_to_scenarios/responses_to_scenarios.py
   ```

## Configuration

- API keys are loaded from `.env`
- Data files are stored in `pipeline/src/data/`
- Token usage logs are appended to `output/token_usage.log`

## Token Logging

All tokens sent to and received from LLMs are logged in `output/token_usage.log` via [`pipeline/src/utils/logger/token_logger.py`](pipeline/src/utils/logger/token_logger.py).
