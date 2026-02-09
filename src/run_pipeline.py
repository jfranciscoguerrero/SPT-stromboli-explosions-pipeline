# run_pipeline.py
import sys
import os
import subprocess
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", ".")).resolve()
RESULTS_DIR = (PROJECT_ROOT / os.getenv("RESULTS_DIR", "results")).resolve()

PYTHON = sys.executable
REPO_ROOT = Path(__file__).resolve().parent.parent


def run_script(script_name: str):
    script_path = REPO_ROOT / "src" / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    res = subprocess.run([PYTHON, str(script_path)], cwd=str(REPO_ROOT))
    if res.returncode != 0:
        raise RuntimeError(f"{script_name} failed with exit code {res.returncode}")


def main():
    t0 = time.time()

    print("[1/5] Running frame indexing")
    run_script("index_frame.py")
    print("Frame indexing completed (^‿^)\n")

    print("[2/5] Running LSTM analyzer")
    run_script("LSTM_analyzer.py")
    print("LSTM analysis completed (^‿^)\n")

    print("[3/5] Running Retina_U-Net analyzer")
    run_script("RetinaNet_Unet_analyzer.py")
    print("Retina_U-Net analyzer completed (^‿^)\n")

    print("[4/5] Creating Report")
    run_script("file_final.py")
    print("Report generated (^‿^)\n")

    print("[5/5] Creating Plots")
    run_script("plots.py")
    print("Plots generated (^‿^)\n")

    total_min = (time.time() - t0) / 60.0

    print("\n==============================")
    print(f"Pipeline DONE! (*‿*) files save at: {RESULTS_DIR}")
    print(f"Total time: {total_min:.2f} minutes")
    print("==============================")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error to run pipeline: (>_<)", e)
        sys.exit(1)
