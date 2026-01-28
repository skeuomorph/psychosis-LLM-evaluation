from typing import Tuple
import re
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.genai.errors import ServerError, ClientError
from openai import APIError, RateLimitError

from pipeline.src.utils.logger.token_logger import log_token_usage
from pipeline.src.prompts.criteria_prompts.criteria_prompts import criteria_prompts

import pandas as pd
import os
from google import genai
from google.genai import types
from openai import OpenAI
from datetime import datetime

class LLMAsJudge:
    def __init__(self, model_name=None):
        """
        Initialize LLMAsJudge.
        
        Args:
            model_name: Name of the model (optional)
            rate_limit_delay: Delay in seconds between API calls (default: 1.0)
        """
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
    
    @staticmethod
    def split_answer_explanation_qwen(text: str) -> Tuple[str, str]:
        """Split the response into answer and explanation parts for Qwen."""
        match = re.search(r'</think>\s*(.*)', text, re.DOTALL)
        if match:
            result = match.group(1).strip()
            lines = result.strip().split('\n', 1)
            answer = lines[0].strip().lower()
            explanation = lines[1].strip() if len(lines) > 1 else ""
            return answer, explanation
        else:
            raise ValueError("Response format is incorrect")
    
    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type((APIError, RateLimitError)),
        reraise=True,
        before_sleep=lambda retry_state: print(f"  ⚠️  Qwen API error, retrying (attempt {retry_state.attempt_number}/10)...")
    )
    def assess_response_qwen(self, scenario: str, model_response: str, criteria_prompt: str, seed: int = 42) -> Tuple[str, str]:
        """Assess a model response using Qwen and return answer and explanation."""
        prompt = criteria_prompt + f"\nUSER'S MESSAGE:\n{scenario}\nLLM RESPONSE:\n{model_response}"
        client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.getenv("HF_TOKEN"),
        )
        response = client.chat.completions.create(
            model="Qwen/Qwen3-32B-FP8",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                },
            ],
            temperature=0,
            seed=seed,
        )
        dump = response.model_dump()
        prompt_tok = dump.get('usage', {}).get('prompt_tokens')
        response_tok = dump.get('usage', {}).get('completion_tokens')
        content = dump['choices'][0]['message']['content']
        log_token_usage("qwen3-32b-fp8", prompt, content, prompt_tokens=prompt_tok, thought_tokens=None, response_tokens=response_tok)
        cleaned = self.clean_response_text(content)
        return self.split_answer_explanation_qwen(cleaned)

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type((ServerError, ClientError)),
        reraise=True,
        before_sleep=lambda retry_state: print(f"  ⚠️  Gemini API error (attempt {retry_state.attempt_number}/10), waiting before retry...")
    )
    def assess_response_gemini(self, scenario: str, model_response: str, criteria_prompt: str, seed: int = 42) -> Tuple[str, str]:
        """Assess a model response using Gemini and return answer and explanation."""
        prompt = criteria_prompt + f"\nUSER'S MESSAGE:\n{scenario}\nLLM RESPONSE:\n{model_response}"
        api_key = os.getenv("GEMINI")
        client = genai.Client(api_key=api_key)
        try:
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    seed=seed,
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

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type((APIError, RateLimitError)),
        reraise=True,
        before_sleep=lambda retry_state: print(f"  ⚠️  Kimi API error, retrying (attempt {retry_state.attempt_number}/10)...")
    )
    def assess_response_kimi(self, scenario: str, model_response: str, criteria_prompt: str, seed: int = 42):
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
            temperature=0,
            seed=seed,
        )
        dump = response.model_dump()
        prompt_tok = dump.get('usage', {}).get('prompt_tokens')
        response_tok = dump.get('usage', {}).get('completion_tokens')
        content = dump['choices'][0]['message']['content']
        log_token_usage("Kimi-K2-Instruct", prompt, content, prompt_tokens=prompt_tok, thought_tokens=None, response_tokens=response_tok)
        cleaned = self.clean_response_text(content)
        return self.split_answer_explanation(cleaned)

    def run(self) -> None:
        """Run the LLM judge pipeline and save results for each model with multiple seeds."""
        input_path = os.path.join(
            os.path.dirname(__file__), "..", "src", "data", "model_responses_to_base_scenarios_2025-08-07.csv"
        )
        df = pd.read_csv(input_path)
        
        print(f"\n{'='*60}")
        print(f"Starting LLM-as-a-Judge Evaluation")
        print(f"{'='*60}")
        print(f"Total scenarios to evaluate: {len(df)}")

        models = [
            # ("qwen", self.assess_response_qwen),
            ("gemini", self.assess_response_gemini),
            ("kimi", self.assess_response_kimi)
        ]
        criteria = list(criteria_prompts.keys())
        seeds = list(range(25))  # Seeds
        
        print(f"Judge models: {[m[0] for m in models]}")
        print(f"Number of criteria: {len(criteria)}")
        print(f"Number of seeds: {len(seeds)}")
        
        total_evaluations = len(df) * len(models) * len(criteria) * 4  # 4 base models
        total_api_calls = total_evaluations * len(seeds)
        print(f"Total API calls to make: {total_api_calls}")
        print(f"{'='*60}\n")

        for seed_idx, seed in enumerate(seeds, 1):
            print(f"\n{'='*60}")
            print(f"🌱 SEED {seed} ({seed_idx}/{len(seeds)})")
            print(f"{'='*60}")
            today = datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
            
            for model_idx, (model_name, assess_fn) in enumerate(models, 1):
                print(f"\n🤖 Judge Model: {model_name.upper()} ({model_idx}/{len(models)})")
                print(f"   Starting evaluation at {datetime.now().strftime('%H:%M:%S')}")
                
                yesno_rows = []
                explanation_rows = []
                
                for scenario_idx, (_, row) in enumerate(df.iterrows(), 1):
                    scenario = row["scenarios"]
                    scenario_id = row.get("id", scenario_idx)
                    
                    print(f"   📝 Scenario {scenario_idx}/{len(df)} (ID: {scenario_id})", end="", flush=True)
                    
                    yesno_row = {"id": scenario_id}
                    explanation_row = {"id": scenario_id}
                    
                    evaluations_done = 0
                    total_evals_for_scenario = len(["chatgpt", "claude", "deepseek", "llama"]) * len(criteria)
                    
                    for base_model in ["chatgpt", "claude", "deepseek", "llama"]:
                        response = row.get(f"{base_model}_response", "")
                        for criterion in criteria:
                            try:
                                result = assess_fn(scenario, response, criteria_prompts[criterion], seed)
                                if result is None or not isinstance(result, tuple) or len(result) != 2:
                                    raise ValueError("Invalid response format")
                                else:
                                    answer, explanation = result
                            except Exception as e:
                                print(f"\n   ❌ Error evaluating {base_model} on {criterion}")
                                raise e
                            yesno_row[f"{base_model}_{criterion}"] = answer
                            explanation_row[f"{base_model}_{criterion}"] = explanation
                            evaluations_done += 1
                    
                    print(f" ✓ ({evaluations_done} evaluations)", flush=True)
                    yesno_rows.append(yesno_row)
                    explanation_rows.append(explanation_row)

                yesno_df = pd.DataFrame(yesno_rows)
                explanation_df = pd.DataFrame(explanation_rows)
                base_path = os.path.join(os.path.dirname(__file__), "..", "src", "data", "judges")
                
                binary_path = os.path.join(base_path, f"{model_name}_as_judge_binary_seed_{seed}_{today}.csv")
                explanations_path = os.path.join(base_path, f"{model_name}_as_judge_explanations_seed_{seed}_{today}.csv")
                
                yesno_df.to_csv(binary_path, index=False)
                explanation_df.to_csv(explanations_path, index=False)
                
                print(f"   💾 Saved results for {model_name}")
                print(f"      Binary: {binary_path}")
                print(f"      Explanations: {explanations_path}")
                
            print(f"\n✅ Completed seed {seed} at {datetime.now().strftime('%H:%M:%S')}")
        
        print(f"\n{'='*60}")
        print(f"🎉 ALL EVALUATIONS COMPLETE!")
        print(f"{'='*60}\n")