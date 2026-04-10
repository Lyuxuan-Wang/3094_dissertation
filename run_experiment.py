"""
    Top-level entry point. Runs the full pipeline in order:
    1. generate_dataset
    2. run_denoise
    3. run_sr
    4. evaluate all results
"""
import sys
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPT_DIR = PROJECT_ROOT / "scripts"
METADATA_DIR = PROJECT_ROOT / "metadata"
RESULTS_DIR = PROJECT_ROOT / "results"

sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_dataset import generate_dataset
from scripts.run_denoise import run_denoise
from scripts.run_sr import run_sr
from pipeline.evaluate import evaluate

BASICSR_ROOT = PROJECT_ROOT / "models/BasicSR"
EDSR_YML = BASICSR_ROOT / "options/test/EDSR/test_EDSR_Lx4.yml"

DATASET_CSV = METADATA_DIR / "dataset.csv"
RESULTS_CSV = METADATA_DIR / "results.csv"
EVAL_CSV = METADATA_DIR / "evaluation.csv"

def step_generate():
    print("Generating dataset...")
    generate_dataset()
    print("Generating results...")

def step_run_denoise():
    print("Running denoise...")
    run_denoise(
        dataset_csv=DATASET_CSV,
        output_dir=RESULTS_DIR / "DN",
        experiment_name="denoise"
    )

def step_run_sr():
    print("Running SR...")
    run_sr(
        dataset_csv=DATASET_CSV,
        output_dir=RESULTS_DIR / "SR",
        basicsr_root=BASICSR_ROOT,
        yml_path=EDSR_YML,
        tile=128,
        tile_pad=10,
        experiment_name="edsr_x4"
    )

def step_run_evaluate():
    print("Running evaluate...")
    dataset = pd.read_csv(DATASET_CSV)
    results = pd.read_csv(RESULTS_CSV)

    records = []
    for i in range(len(results)):
        row = results.iloc[i]
        record = evaluate(row, dataset)
        records.append(record)

    eval_df = pd.DataFrame(records)
    eval_df.to_csv(EVAL_CSV, index=False)
    print("Evaluation written")

if __name__ == "__main__":
    step_generate()
    step_run_denoise()
    step_run_sr()
    step_run_evaluate()