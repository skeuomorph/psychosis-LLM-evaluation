import os
from datetime import datetime

LOG_PATH = os.path.join(os.path.dirname(__file__), "token_usage.log")

# Ensure the directory for the log file exists
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

def log_token_usage(model_name, prompt, response, prompt_tokens=None, thought_tokens=None, response_tokens=None):
    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "model": model_name,
        "prompt_tokens": prompt_tokens if prompt_tokens is not None else "N/A",
        "thought_tokens": thought_tokens if thought_tokens is not None else "N/A",
        "response_tokens": response_tokens if response_tokens is not None else "N/A",
        "prompt": prompt.replace("\n", "\\n"),
        "response": response.replace("\n", "\\n"),
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(str(log_entry) + "\n")