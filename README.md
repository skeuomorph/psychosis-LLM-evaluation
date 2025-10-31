# Using LLM-as-a-Judge/Jury to Advance Scalable, Clinically-Validated Safety Evaluations of Model Responses to Users Demonstrating Psychosis

Code used in "Using LLM-as-a-Judge/Jury to Advance Scalable, Clinically-Validated Safety Evaluations of Model Responses to Users Demonstrating Psychosis", a workshop paper for The Second Workshop on GenAI for Health: Potential, Trust, and Policy Compliance at Neurips 2025.

To cite this work, please use: 
@inproceedings{
reese2025using,
title={Using {LLM}-as-a-Judge/Jury to Advance Scalable, Clinically-Validated Safety Evaluations of Model Responses to Users Demonstrating Psychosis},
author={May Lynn Reese and Markela Zeneli and Mindy Ng and Jacob Haimes and Andreea Damien and Elizabeth C. Stade},
booktitle={The Second Workshop on GenAI for Health: Potential, Trust, and Policy Compliance},
year={2025},
url={ https://openreview.net/forum?id=78ootwhyDQ }
}

## Project Overview

The project follows this workflow:
1. **Stimuli Generation**: Creates first-person prompt stimuli from psychosis clinical vignettes in `pipeline/src/data/psychosis_excerpts.csv`. Also generates variations, which are not currently used in analysis.
2. **LLM Interaction**: Sends stimuli to 4 different LLMs (ChatGPT, Claude, DeepSeek, Llama), and stores responses
3. **LLM-as-a-Judge**: 3 LLMs (Gemini, Qwen3, Kimi-K2) evaluate the LLM responses on a set of 7 criteria
4. **Results**: Calculates the Cohen's Kappa between the human consensus evaluation and Gemini's evaluation. Creates an LLM-as-a-Jury verdict by majority vote. Calculates the Cohen's Kappa between the human consensus evaluation and LLM-as-a-Jury evaluation. Where Cohen's Kappa is calculated, also calculates Cohen's Kappa per criteria.


**Step Locations:**
- **Step 1: Stimuli Generation** — `pipeline/generate_scenarios/generate_scenarios.py`, source data in `pipeline/src/data/psychosis_excerpts.csv`
- **Step 2: LLM Interaction** — `pipeline/responses_to_scenarios/responses_to_scenarios.py`
- **Step 3: LLM-as-a-Judge** — `pipeline/llm_as_judge/llm_as_judge.py`
- **Step 4: Results** — `pipeline/generate_stats/generate_stats.py`


## Quick Start

1. **Setup Environment** (choose one):

   - **Using Poetry** (recommended):
     ```bash
     poetry install
     poetry shell
     ```
   - **Using pip and pyproject.toml**:
     ```bash
     pip install --upgrade pip
     pip install .
     ```

2. **Run the full pipeline**:
   ```bash
   python main.py
   ```

> Ensure your `.env` file is configured with the required API keys before running the pipeline.

## Configuration

- API keys are loaded from `.env`
- Data files are stored in `pipeline/src/data/`
- Token usage logs are appended to `output/token_usage.log`

## Token Logging

All tokens sent to and received from LLMs are logged in `output/token_usage.log` via [`pipeline/src/utils/logger/token_logger.py`](pipeline/src/utils/logger/token_logger.py).
