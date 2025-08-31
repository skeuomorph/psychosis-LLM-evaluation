import time
from typing import Tuple

from pipeline.src.utils.logger.token_logger import log_token_usage
from pipeline.src.prompts.criteria_prompts.criteria_prompts import criteria_prompts

import pandas as pd
import os
import openai
from google import genai
from google.genai import types
from openai import OpenAI
from datetime import datetime
from concurrent.futures import TimeoutError as FuturesTimeout

class LLMAsJudge:
    def __init__(self, model_name=None):
        self.model_name = model_name

    @staticmethod
    def clean_response_text(variation: str) -> str:
        """Remove leading/trailing whitespace from the response text."""
        variation = variation.strip()
        return variation

    @staticmethod
    def split_answer_explanation(text: str) -> Tuple[str, str]:
        """Split the response into answer and explanation parts."""
        lines = text.strip().split('\n', 1)
        answer = lines[0].strip().lower()
        explanation = lines[1].strip() if len(lines) > 1 else ""
        return answer, explanation

    def assess_response_qwen(self, scenario: str, model_response: str, criteria_prompt: str) -> Tuple[str, str]:
        """Assess a model response using Qwen and return answer and explanation."""
        prompt = criteria_prompt + f"\nUSER'S MESSAGE:\n{scenario}\nLLM RESPONSE:\n{model_response}"
        api_key = os.getenv("LAMBDA")
        client = openai.OpenAI(api_key=api_key, base_url="https://api.lambda.ai/v1")
        try:
            response = client.chat.completions.create(
                model="qwen3-32b-fp8",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            dump = response.model_dump()
            prompt_tok = dump.get('usage', {}).get('prompt_tokens')
            response_tok = dump.get('usage', {}).get('completion_tokens')
            content = dump['choices'][0]['message']['content']
            log_token_usage("qwen3-32b-fp8", prompt, content, prompt_tokens=prompt_tok, thought_tokens=None, response_tokens=response_tok)
            cleaned = self.clean_response_text(content)
            return self.split_answer_explanation(cleaned)
        except Exception as e:
            raise e

    def assess_response_gemini(self, scenario: str, model_response: str, criteria_prompt: str) -> Tuple[str, str]:
        """Assess a model response using Gemini and return answer and explanation."""
        prompt = criteria_prompt + f"\nUSER'S MESSAGE:\n{scenario}\nLLM RESPONSE:\n{model_response}"
        api_key = os.getenv("GEMINI")
        client = genai.Client(api_key=api_key)
        try:
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0
                )
            )
            dump = response.model_dump()
            prompt_tok = dump.get('usage_metadata', {}).get('prompt_token_count')
            thought_tok = dump.get('usage_metadata', {}).get('thoughts_token_count')
            response_tok = dump.get('usage_metadata', {}).get('candidates_token_count')
            content = dump['candidates'][0]['content']['parts'][0]['text']
            log_token_usage("gemini-2.5-pro", prompt, content, prompt_tokens=prompt_tok, thought_tokens=thought_tok, response_tokens=response_tok)
            cleaned = self.clean_response_text(content)
            return self.split_answer_explanation(cleaned)
        except Exception as e:
            raise e

    def assess_response_kimi(self, scenario: str, model_response: str, criteria_prompt: str):
        """Assess a model response using Kimi and return answer and explanation."""
        prompt = criteria_prompt + f"\nUSER'S MESSAGE:\n{scenario}\nLLM RESPONSE:\n{model_response}"
        client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.getenv("HF_TOKEN"),
        )
        response = client.chat.completions.create(
            model="moonshotai/Kimi-K2-Instruct",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                },
            ],
            temperature=0
        )
        dump = response.model_dump()
        prompt_tok = dump.get('usage', {}).get('prompt_tokens')
        response_tok = dump.get('usage', {}).get('completion_tokens')
        content = dump['choices'][0]['message']['content']
        print("done")
        log_token_usage("Kimi-K2-Instruct", prompt, content, prompt_tokens=prompt_tok, thought_tokens=None, response_tokens=response_tok)
        cleaned = self.clean_response_text(content)
        return self.split_answer_explanation(cleaned)

    def run(self) -> None:
        """Run the LLM judge pipeline and save results for each model."""
        input_path = os.path.join(
            os.path.dirname(__file__), "..", "src", "data", "model_responses_to_base_scenarios_2025-08-07.csv"
        )
        df = pd.read_csv(input_path)

        models = [
            ("qwen", self.assess_response_qwen),
            ("gemini", self.assess_response_gemini),
            ("kimi", self.assess_response_kimi)
        ]
        criteria = list(criteria_prompts.keys())

        for model_name, assess_fn in models:
            yesno_rows = []
            explanation_rows = []
            for _, row in df.iterrows():
                scenario = row["scenarios"]
                yesno_row = {"id": row.get("id", None)}
                explanation_row = {"id": row.get("id", None)}
                for base_model in ["chatgpt", "claude", "deepseek", "llama"]:
                    response = row.get(f"{base_model}_response", "")
                    for criterion in criteria:
                        try:
                            result = assess_fn(scenario, response, criteria_prompts[criterion])
                            if result is None or not isinstance(result, tuple) or len(result) != 2:
                                raise ValueError("Invalid response format")
                            else:
                                answer, explanation = result
                        except Exception as e:
                            raise e
                        yesno_row[f"{base_model}_{criterion}"] = answer
                        explanation_row[f"{base_model}_{criterion}"] = explanation
                yesno_rows.append(yesno_row)
                explanation_rows.append(explanation_row)

            yesno_df = pd.DataFrame(yesno_rows)
            explanation_df = pd.DataFrame(explanation_rows)
            today = datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
            base_path = os.path.join(os.path.dirname(__file__), "..", "src", "data", "judges")
            yesno_df.to_csv(os.path.join(base_path, f"{model_name}_as_judge_binary_{today}.csv"), index=False)
            explanation_df.to_csv(os.path.join(base_path, f"{model_name}_as_judge_explanations_{today}.csv"), index=False)
            print(f"Saved results for {model_name}")