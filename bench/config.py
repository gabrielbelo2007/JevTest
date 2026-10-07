from __future__ import annotations

import os
from pathlib import Path
from dotenv import dotenv_values, load_dotenv

# Base paths
ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
MANIFEST_PATH = RESULTS_DIR / "manifest.jsonl"
RAW_RESULTS_DIR = RESULTS_DIR / "raw"
CHARTS_DIR = RESULTS_DIR / "charts"
REPORT_PATH = RESULTS_DIR / "report.md"

# Load .env explicitly handling spaces around '=' and lowercase names
load_dotenv(ROOT_DIR / ".env")
env_vals = dotenv_values(ROOT_DIR / ".env")

def get_env_var(*names: str, default: str = "") -> str:
    for name in names:
        val = os.getenv(name) or env_vals.get(name) or env_vals.get(name.lower()) or env_vals.get(name.upper())
        if val:
            return val.strip()
    return default

OPENROUTER_API_KEY = get_env_var("OPENROUTER_API_KEY", "openrouter_api_key")
OPENROUTER_ENDPOINT = os.getenv("OPENROUTER_ENDPOINT", "https://openrouter.ai/api/alpha/decisions")
JEV_MODEL = os.getenv("JEV_MODEL", "typesafe/jev-1.13")

# CLM-8B (Stanford & NVIDIA Contrastive Language Model) settings
CLM_ENDPOINT = os.getenv("CLM_ENDPOINT", "http://127.0.0.1:8700/v1/systemone")
CLM_MODEL = os.getenv("CLM_MODEL", "Contrastive-LM/CLM-v0.1-8B")
CLM_API_KEY = get_env_var("CLM_API_KEY", "clm_api_key", default="")

# Benchmark settings
SEED = 42
SAMPLES_PER_TASK = 200
WARMUP_RUNS = 3
TIMEOUT_SECONDS = 30.0

# Supported tasks
TASKS = ["ag_news", "banking77", "emotion", "sst5", "sst2"]
