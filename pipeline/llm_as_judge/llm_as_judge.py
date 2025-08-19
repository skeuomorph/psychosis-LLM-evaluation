from pipeline.src.utils.logger.token_logger import log_token_usage
from pipeline.src.prompts.criteria_prompts.criteria_prompts import criteria_prompts

import pandas as pd
import os
import openai
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

class LLMAsJudge:
    def __init__(self, model_name=None):
        self.model_name = model_name

    @staticmethod
    def clean_variation_text(variation):
        variation = variation.strip()
        return variation

    @staticmethod
    def split_answer_explanation(text):
        """
        Splits the LLM response into 'yes/no' and explanation.
        Assumes the answer is the first word (yes/no), followed by a newline and explanation.
        """
        lines = text.strip().split('\n', 1)
        answer = lines[0].strip().lower()
        explanation = lines[1].strip() if len(lines) > 1 else ""
        return answer, explanation

    def assess_response(self, scenario, model_response, criteria_prompt):
        # Compose the prompt for the LLM-as-judge
        prompt = criteria_prompt + f"\nUSER'S MESSAGE:\n{scenario}\nLLM RESPONSE:\n{model_response}"
        api_key = os.getenv("LAMBDA")
        client = openai.OpenAI(api_key=api_key, base_url="https://api.lambda.ai/v1")
        try:
            response = client.chat.completions.create(
                model="qwen3-32b-fp8",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            dump = response.model_dump()
            prompt_tokens = dump.get('usage', {}).get('prompt_tokens')
            response_tokens = dump.get('usage', {}).get('completion_tokens')
            content = dump['choices'][0]['message']['content']
            log_token_usage("qwen3-32b-fp8", prompt, content, prompt_tokens, response_tokens)
            cleaned = self.clean_variation_text(content)
            return self.split_answer_explanation(cleaned)
        except Exception as e:
            return ("error", str(e))

    def run(self):
        input_path = os.path.join(os.path.dirname(__file__), "..", "src", "data", "holdout_sample_with_responses.csv")
        df = pd.read_csv(input_path)

        models = ["chatgpt", "claude", "deepseek", "llama"]
        criteria = list(criteria_prompts.keys())

        yesno_rows = []
        explanation_rows = []

        for _, row in df.iterrows():
            scenario = row["scenarios"]
            yesno_row = {"id": row.get("id", None)}
            explanation_row = {"id": row.get("id", None)}
            for model in models:
                response = row.get(f"{model}_response", "")
                with ThreadPoolExecutor() as executor:
                    futures = {
                        criterion: executor.submit(
                            self.assess_response, scenario, response, criteria_prompts[criterion]
                        )
                        for criterion in criteria
                    }
                    for criterion, future in futures.items():
                        answer, explanation = future.result()
                        yesno_row[f"{model}_{criterion}"] = answer
                        explanation_row[f"{model}_{criterion}"] = explanation
            yesno_rows.append(yesno_row)
            explanation_rows.append(explanation_row)

        yesno_df = pd.DataFrame(yesno_rows)
        explanation_df = pd.DataFrame(explanation_rows)
        today = datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
        base_path = os.path.join(os.path.dirname(__file__), "..", "src", "data")
        yesno_df.to_csv(os.path.join(base_path, f"holdout_sample_with_judge_yesno_{today}.csv"), index=False)
        explanation_df.to_csv(os.path.join(base_path, f"holdout_sample_with_judge_explanations_{today}.csv"), index=False)