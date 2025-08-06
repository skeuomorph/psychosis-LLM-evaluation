# Mental Health Crises LLM Evaluation

This project evaluates how different Large Language Models (LLMs) respond to psychosis scenarios and assesses their responses against predefined criteria to determine which models are most suitable for mental health crisis situations.

## Project Overview

The project follows this workflow:
1. **Prompt Generation**: Creates prompts from psychosis scenarios in `data/psychosis_scenarios.csv`
2. **LLM Interaction**: Sends prompts to 4 different LLMs (ChatGPT, Claude, DeepSeek, Llama)
3. **Response Collection**: Gathers and stores responses from each model
4. **Criteria Evaluation**: Assesses responses against safety and helpfulness criteria
5. **Scoring**: Assigns scores based on performance against criteria

## Project Structure

```
mental-health-crises-LLM-evaluation/
├── README.md                           # This file
├── requirements.txt                     # Python dependencies
├── data/
│   ├── README.md                      # Data documentation
│   ├── psychosis_scenarios.csv        # Base scenarios
│   ├── prompts_with_variations.csv    # Generated prompts
│   └── cannot_help_list/             # LLM responses to "cannot help" prompt
│       ├── chatgpt.csv
│       ├── claude.csv
│       ├── deepseek.csv
│       └── llama.csv
├── pipeline/
│   ├── __init__.py
│   ├── main.py                        # Main execution pipeline
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py               # Configuration settings
│   │   └── model_configs.py         # LLM-specific configurations
│   ├── src/
│   │   ├── __init__.py
│   │   ├── prompt_generation/
│   │   │   ├── __init__.py
│   │   │   ├── prompt_builder.py    # Creates prompts from scenarios
│   │   │   └── variation_generator.py # Generates prompt variations
│   │   ├── llm_interaction/
│   │   │   ├── __init__.py
│   │   │   ├── llm_client.py        # Base LLM client interface
│   │   │   ├── chatgpt_client.py    # ChatGPT API client
│   │   │   ├── claude_client.py     # Claude API client
│   │   │   ├── deepseek_client.py   # DeepSeek API client
│   │   │   └── llama_client.py      # Llama API client
│   │   ├── evaluation/
│   │   │   ├── __init__.py
│   │   │   ├── criteria_checker.py  # Evaluates responses against criteria
│   │   │   ├── safety_evaluator.py  # Safety-specific evaluation
│   │   │   └── helpfulness_evaluator.py # Helpfulness evaluation
│   │   ├── scoring/
│   │   │   ├── __init__.py
│   │   │   ├── score_calculator.py  # Calculates final scores
│   │   │   └── ranking_system.py    # Ranks models by performance
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── data_loader.py       # Data loading utilities
│   │       ├── logger.py            # Logging configuration
│   │       └── file_utils.py       # File handling utilities
│   ├── pipeline_modules/
│   │   ├── __init__.py
│   │   ├── prompt_pipeline.py       # Prompt generation pipeline
│   │   ├── llm_pipeline.py         # LLM interaction pipeline
│   │   └── evaluation_pipeline.py   # Evaluation and scoring pipeline
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_prompt_generation/
│   │   ├── test_llm_interaction/
│   │   ├── test_evaluation/
│   │   └── test_scoring/
│   ├── notebooks/
│   │   ├── exploratory_analysis.ipynb # Data exploration
│   │   ├── results_analysis.ipynb     # Results visualization
│   │   └── model_comparison.ipynb     # Model performance comparison
│   ├── docs/
│   │   ├── api_documentation.md       # API documentation
│   │   ├── evaluation_criteria.md     # Detailed criteria documentation
│   │   └── deployment_guide.md        # Deployment instructions
│   ├── scripts/
│   │   ├── setup_environment.py       # Environment setup script
│   │   ├── run_full_pipeline.py       # Complete pipeline execution
│   │   └── generate_report.py         # Report generation script
│   └── output/
│       ├── llm_responses/            # Raw LLM responses
│       │   ├── chatgpt/
│       │   ├── claude/
│       │   ├── deepseek/
│       │   └── llama/
│       └── evaluated_responses/      # Responses with criteria scores
└── code/                             # Legacy code directory
    ├── base-prompt-variations.ipynb
    ├── models-mental-health-cannot-help-list.ipynb
    ├── README.md
    └── scenario-base.ipynb
```

## Quick Start

1. **Setup Environment**:
   ```bash
   pip install -r requirements.txt
   python pipeline/scripts/setup_environment.py
   ```

2. **Run Full Pipeline**:
   ```bash
   python pipeline/scripts/run_full_pipeline.py
   ```

3. **Generate Report**:
   ```bash
   python pipeline/scripts/generate_report.py
   ```

## Configuration

Update `pipeline/config/settings.py` with your API keys and preferences:
- LLM API keys
- Evaluation criteria weights
- Output file paths
- Logging levels

## Evaluation Criteria

The project evaluates LLM responses based on:
- **Safety**: Appropriate crisis intervention guidance
- **Helpfulness**: Practical and actionable advice
- **Compliance**: Adherence to mental health guidelines
- **Sensitivity**: Appropriate tone and language

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

[Add your license information here]
