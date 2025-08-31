from pipeline.src.utils.logger.token_logger import log_token_usage

import os
from datetime import datetime

from dotenv import load_dotenv
import pandas as pd
import anthropic

from pipeline.src.prompts.scenario_prompts.scenario_prompts import scenario_prompts

load_dotenv()

class ScenarioGenerator:
    def __init__(self):
        today_str = datetime.today().strftime("%Y-%m-%d")
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.csv_path = os.path.join(
            base_dir, "src", "data", "psychosis_excerpts.csv"
        )
        self.base_scenarios_path = os.path.join(
            base_dir, "src", "data", f"psychosis_base_scenarios_{today_str}.csv"
        )
        self.output_path = os.path.join(
            base_dir, "src", "data", f"psychosis_scenarios_with_variations_{today_str}.csv"
        )
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC"))
        self.df = pd.read_csv(self.csv_path)
        if "base_scenario" not in self.df.columns:
            self.df["base_scenario"] = ""
        if "scenario_variations" not in self.df.columns:
            self.df["scenario_variations"] = ""

    @staticmethod
    def clean_variation_text(variation: str) -> str:
        """Clean and format scenario variation text."""
        variation = variation.replace("\n", " ")
        variation = variation.strip()
        return variation

    def generate_base_scenarios(self) -> None:
        """Generate stimuli from vignette excerpts and save to CSV."""
        for idx, excerpt in self.df["excerpt"].items():
            prompt = scenario_prompts["base_prompt"] + "\n" + excerpt
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            dump = response.model_dump()
            prompt_tok = dump.get('usage', {}).get('input_tokens')
            response_tok = dump.get('usage', {}).get('output_tokens')
            log_token_usage("claude-sonnet-4-20250514", prompt, dump['content'][0]['text'], prompt_tokens=prompt_tok, thought_tokens=None, response_tokens=response_tok)
            self.df.at[idx, "base_scenario"] = self.clean_variation_text(dump['content'][0]['text'])
        self.df.to_csv(self.base_scenarios_path, index=False)

    def generate_variations(self) -> None:
        """Generate stimuli variations from base stimuli."""
        for idx, base_scenario in self.df["base_scenario"].items():
            prompt = scenario_prompts["variations_prompt"] + "\n" + str(base_scenario)
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            dump = response.model_dump()
            prompt_tok = dump.get('usage', {}).get('input_tokens')
            response_tok = dump.get('usage', {}).get('output_tokens')
            log_token_usage("claude-sonnet-4-20250514", prompt, dump['content'][0]['text'], prompt_tokens=prompt_tok, thought_tokens=None, response_tokens=response_tok)
            self.df.at[idx, "scenario_variations"] = self.clean_variation_text(dump['content'][0]['text'])

    def split_and_save_variations(self) -> None:
        """Split stimuli variations and save all stimuli to output CSV."""
        rows = []
        for idx, (_, row) in enumerate(self.df.iterrows()):
            base_id = f"{idx+1}_a"
            rows.append({
                "id": base_id,
                "excerpt": row["excerpt"],
                "scenarios": row["base_scenario"],
            })
            variations_text = row["scenario_variations"]
            variations = []
            for i in range(1, 6):
                label = f"Variation {i}:"
                next_label = f"Variation {i+1}:"
                start = variations_text.find(label)
                if start == -1:
                    continue
                start += len(label)
                end = variations_text.find(next_label)
                if end == -1:
                    end = len(variations_text)
                variation = variations_text[start:end]
                variation = self.clean_variation_text(variation)
                variations.append(variation)
            for v_idx, variation in enumerate(variations):
                var_id = chr(ord('b') + v_idx)
                rows.append({
                    "id": f"{idx+1}_{var_id}",
                    "excerpt": row["excerpt"],
                    "scenarios": variation,
                })
        output_df = pd.DataFrame(rows, columns=["id", "excerpt", "scenarios"])
        output_df.to_csv(self.output_path, index=False)

    def run(self):
        self.generate_base_scenarios()
        self.generate_variations()
        self.split_and_save_variations()


def main():
    ScenarioGenerator().run()
