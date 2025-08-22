from pipeline.src.utils.logger.token_logger import log_token_usage

import pandas as pd
import os
import anthropic
import openai
from datetime import datetime

class ScenarioResponder:
    def __init__(self, model_name=None):
        self.model_name = model_name

    @staticmethod
    def clean_variation_text(variation):
        variation = variation.replace("\n", " ")
        variation = variation.strip()
        return variation

    def get_response(self, prompt):
        if self.model_name == "chatgpt":
            openai.api_key = os.getenv("OPENAI")
            try:
                response = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                )
                dump = response.model_dump()
                prompt_tokens = dump.get('usage', {}).get('prompt_tokens')
                response_tokens = dump.get('usage', {}).get('completion_tokens')
                log_token_usage("gpt-4o", prompt, dump['choices'][0]['message']['content'], prompt_tokens, response_tokens)
                return self.clean_variation_text(dump['choices'][0]['message']['content'])
            except Exception as e:
                raise e
        elif self.model_name == "claude":
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC"))
            try:
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=2000,
                    temperature=0.7,
                    messages=[{"role": "user", "content": prompt}],
                )
                dump = response.model_dump()
                # NOTE: Anthropic's response structure is different to the other models
                prompt_tokens = dump.get('usage', {}).get('input_tokens')
                response_tokens = dump.get('usage', {}).get('output_tokens')
                log_token_usage("claude-sonnet-4-20250514", prompt, dump['content'][0]['text'], prompt_tokens, response_tokens)
                return self.clean_variation_text(response.model_dump()['content'][0]['text'])
            except Exception as e:
                raise e
        elif self.model_name == "deepseek":
            api_key = os.getenv("LAMBDA")
            client = openai.OpenAI(api_key=api_key, base_url="https://api.lambda.ai/v1")
            try:
                response = client.chat.completions.create(
                    model="deepseek-v3-0324",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                )
                print(f"Response from model {self.model_name}: {response}")
                dump = response.model_dump()
                prompt_tokens = dump.get('usage', {}).get('prompt_tokens')
                response_tokens = dump.get('usage', {}).get('completion_tokens')
                log_token_usage("deepseek-v3-0324", prompt, dump['choices'][0]['message']['content'], prompt_tokens, response_tokens)
                return self.clean_variation_text(response.model_dump()['choices'][0]['message']['content'])
            except Exception as e:
                raise e
        elif self.model_name == "llama":
            api_key = os.getenv("LAMBDA")
            client = openai.OpenAI(api_key=api_key, base_url="https://api.lambda.ai/v1")
            try:
                response = client.chat.completions.create(
                    model="llama3.1-405b-instruct",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                )
                dump = response.model_dump()
                prompt_tokens = dump.get('usage', {}).get('prompt_tokens')
                response_tokens = dump.get('usage', {}).get('completion_tokens')
                log_token_usage("llama3.1-405b-instruct-fp8", prompt, dump['choices'][0]['message']['content'], prompt_tokens, response_tokens)
                return self.clean_variation_text(response.model_dump()['choices'][0]['message']['content'])
            except Exception as e:
                raise e

    def run(self):
        data_dir = os.path.join(os.path.dirname(__file__), "..", "src", "data")
        files = [f for f in os.listdir(data_dir) if f.startswith("psychosis_base_scenarios_with_variations_") and f.endswith(".csv")]
        # files = [f for f in os.listdir(data_dir) if f.startswith("holdout") and f.endswith(".csv")]
        if not files:
            raise FileNotFoundError("No matching scenario files found in data directory.")
        latest_file = max(files, key=lambda x: x.split("_")[-1].replace(".csv", ""))
        input_path = os.path.join(data_dir, latest_file)
        print(f"Using input file: {input_path}")

        df = pd.read_csv(input_path)

        models = ["chatgpt", "claude", "deepseek", "llama"]
        responders = {model: ScenarioResponder(model) for model in models}

        for model in models:
            df[model + "_response"] = df["scenarios"].apply(responders[model].get_response)
            # Save after each model to avoid data loss if a model fails
            today = datetime.today().strftime("%Y-%m-%d")
            output_path = os.path.join(os.path.dirname(__file__), "..", "src", "data", f"model_response_to_scenarios_{today}.csv")
            # output_path = os.path.join(data_dir, f"holdout_sample_with_responses.csv")
            df.to_csv(output_path, index=False)

def main():
    ScenarioResponder().run()
