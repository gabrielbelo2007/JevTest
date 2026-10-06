from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm

from bench.clients.base import Prediction
from bench.clients.jev import JevClient
from bench.clients.laya_local import LayaClient
from bench.config import (
    MANIFEST_PATH,
    RAW_RESULTS_DIR,
    REPORT_PATH,
    RESULTS_DIR,
    SAMPLES_PER_TASK,
    TASKS,
    WARMUP_RUNS,
)
from bench.report import generate_report
from bench.tasks import (
    Case,
    build_ag_news,
    build_banking77,
    build_emotion,
    build_sst2,
    build_sst5,
    get_smoke_cases,
)


def load_manifest(path: Path = MANIFEST_PATH) -> list[Case]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest {path} not found. Run 'jl-bench prepare' first.")
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(Case.from_dict(json.loads(line)))
    return cases


def save_manifest(cases: list[Case], path: Path = MANIFEST_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c.to_dict()) + "\n")


def cmd_prepare(args):
    print("Preparing benchmark dataset manifest (200 samples per task)...")
    cases: list[Case] = []

    print("- AG News...")
    cases.extend(build_ag_news(n_samples=SAMPLES_PER_TASK))

    print("- Banking77...")
    cases.extend(build_banking77(n_samples=SAMPLES_PER_TASK))

    print("- Emotion...")
    cases.extend(build_emotion(n_samples=SAMPLES_PER_TASK))

    print("- SST-5...")
    cases.extend(build_sst5(n_samples=SAMPLES_PER_TASK))

    print("- SST-2...")
    cases.extend(build_sst2(n_samples=SAMPLES_PER_TASK))

    save_manifest(cases)
    print(f"Successfully prepared and frozen {len(cases)} cases into {MANIFEST_PATH}")


def cmd_smoke(args):
    print("Running Smoke Test across all 3 models on 5 representative tasks...")
    cases = get_smoke_cases()
    results_dir = RESULTS_DIR / "smoke"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Jev
    print("\n--- Testing Jev via Vercel AI Gateway ---")
    try:
        with JevClient() as jev:
            for c in cases:
                pred = jev.predict(c)
                print(f"[{c.task}] OK={pred.ok} Gold={c.gold} Pred={pred.pred} Latency={pred.latency_ms:.1f}ms")
                if not pred.ok:
                    print(f"  Error: {pred.error}")
    except Exception as e:
        print(f"Jev initialization failed: {e}")

    # 2. Laya (Default)
    print("\n--- Testing Laya (Default) Local MPS ---")
    try:
        laya = LayaClient(tuned=False)
        print(f"Laya loaded in {laya.load_time_ms:.1f}ms")
        for c in cases:
            pred = laya.predict(c)
            print(f"[{c.task}] OK={pred.ok} Gold={c.gold} Pred={pred.pred} Latency={pred.latency_ms:.1f}ms")
    except Exception as e:
        print(f"Laya initialization failed: {e}")

    # 3. Laya (Tuned)
    print("\n--- Testing Laya (Tuned) Local MPS ---")
    try:
        laya_t = LayaClient(tuned=True)
        print(f"Laya Tuned loaded in {laya_t.load_time_ms:.1f}ms")
        for c in cases:
            pred = laya_t.predict(c)
            print(f"[{c.task}] OK={pred.ok} Gold={c.gold} Pred={pred.pred} Latency={pred.latency_ms:.1f}ms")
    except Exception as e:
        print(f"Laya Tuned initialization failed: {e}")


from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def run_model_on_cases(
    model_name: str, client, cases: list[Case], concurrency: int = 1
) -> list[Prediction]:
    RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RAW_RESULTS_DIR / f"{model_name}.jsonl"

    existing_preds: dict[str, Prediction] = {}
    if out_file.exists():
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    existing_preds[data["case_id"]] = Prediction(**data)

    print(f"Model {model_name}: {len(existing_preds)}/{len(cases)} already completed.")

    # Warmup
    if cases:
        print(f"Performing {WARMUP_RUNS} warmup runs...")
        for _ in range(WARMUP_RUNS):
            client.predict(cases[0])

    preds: list[Prediction] = list(existing_preds.values())
    pending_cases = [c for c in cases if c.case_id not in existing_preds]

    if not pending_cases:
        return preds

    file_lock = threading.Lock()

    if concurrency > 1:
        with open(out_file, "a", encoding="utf-8") as f:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                future_to_case = {executor.submit(client.predict, c): c for c in pending_cases}
                for future in tqdm(as_completed(future_to_case), total=len(pending_cases), desc=f"Running {model_name} (x{concurrency})"):
                    pred = future.result()
                    with file_lock:
                        preds.append(pred)
                        f.write(json.dumps(pred.to_dict()) + "\n")
                        f.flush()
    else:
        with open(out_file, "a", encoding="utf-8") as f:
            for c in tqdm(pending_cases, desc=f"Running {model_name}"):
                pred = client.predict(c)
                preds.append(pred)
                f.write(json.dumps(pred.to_dict()) + "\n")
                f.flush()

    return preds


def cmd_run(args):
    cases = load_manifest()
    models_to_run = [args.model] if args.model != "all" else ["jev", "laya", "laya-tuned"]

    for model_name in models_to_run:
        print(f"\n==========================================")
        print(f"Starting execution for model: {model_name}")
        print(f"==========================================")

        if model_name == "jev":
            with JevClient() as client:
                concurrency = getattr(args, "concurrency", 5) or 5
                run_model_on_cases("jev", client, cases, concurrency=concurrency)
        elif model_name == "laya":
            client = LayaClient(tuned=False)
            run_model_on_cases("laya", client, cases, concurrency=1)
        elif model_name == "laya-tuned":
            client = LayaClient(tuned=True)
            run_model_on_cases("laya-tuned", client, cases, concurrency=1)
        else:
            raise ValueError(f"Unknown model: {model_name}")


def cmd_report(args):
    cases = load_manifest()
    all_predictions: dict[str, list[Prediction]] = {}

    for model_file in sorted(RAW_RESULTS_DIR.glob("*.jsonl")):
        model_name = model_file.stem
        preds = []
        with open(model_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    preds.append(Prediction(**json.loads(line)))
        all_predictions[model_name] = preds

    if not all_predictions:
        print("No predictions found in results/raw/. Run 'jl-bench run' first.")
        return

    print(f"Generating benchmark report for models: {list(all_predictions.keys())}...")
    generate_report(cases, all_predictions)
    print(f"Report generated at {REPORT_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Jev vs Laya")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # prepare
    subparsers.add_parser("prepare", help="Download datasets and prepare frozen manifest")

    # smoke
    subparsers.add_parser("smoke", help="Run quick smoke test across all models")

    # run
    p_run = subparsers.add_parser("run", help="Run benchmark on dataset")
    p_run.add_argument(
        "--model",
        choices=["jev", "laya", "laya-tuned", "all"],
        default="all",
        help="Which model to run",
    )
    p_run.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Concurrency level for remote API calls (default 5)",
    )

    # report
    subparsers.add_parser("report", help="Generate Markdown report and charts")

    args = parser.parse_args()

    if args.cmd == "prepare":
        cmd_prepare(args)
    elif args.cmd == "smoke":
        cmd_smoke(args)
    elif args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "report":
        cmd_report(args)


if __name__ == "__main__":
    main()
