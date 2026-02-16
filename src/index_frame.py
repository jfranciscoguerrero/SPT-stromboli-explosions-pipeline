import os
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from dotenv import load_dotenv
from frames_indexer.video_indexer import index_one_day


CURRENT_DIR = Path(__file__).resolve().parent     
REPO_ROOT = CURRENT_DIR.parent                    

load_dotenv(dotenv_path=REPO_ROOT / ".env")     
sys.path.append(str(REPO_ROOT))

#.env
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "."))
PROJECT_ROOT = (REPO_ROOT / PROJECT_ROOT).resolve() if not PROJECT_ROOT.is_absolute() else PROJECT_ROOT.resolve()

CSV_DESTINATION = (PROJECT_ROOT / os.getenv("CSV_OUTPUT_DIR", "results")).resolve()
START_ID = int(os.getenv("START_ID", "0"))

SAMPLES_DIR = (PROJECT_ROOT / os.getenv("SAMPLES_DIR", "data/samples/SPT_videos_sample")).resolve()
print("Using samples dir:", SAMPLES_DIR)

MAX_WORKERS_ENV = int(os.getenv("MAX_WORKERS", "2"))


def process_one_day(day_str: str):
    day_start = time.time()
    print(f"\n=== [PID {os.getpid()}] Processing sample {day_str}")

    index_one_day(
        day_str=day_str,
        samples_dir=SAMPLES_DIR,
        csv_destination=CSV_DESTINATION,
        start_id=START_ID,
    )

    elapsed = time.time() - day_start
    print(f"Sample {day_str} finished in {elapsed/60:.2f} minutes")
    return day_str, elapsed


def list_available_samples(samples_dir: Path):
    if not samples_dir.exists():
        return []
    folders = sorted([p for p in samples_dir.iterdir() if p.is_dir() and p.name.startswith("SPT_")])
    return folders


def choose_samples_interactive(folders):
    print("\nAvailable videos samples ------ (5 minutes of register each):")
    for i, folder in enumerate(folders, start=1):
        n_videos = len(list(folder.glob("*.avi")))
        print(f"  [{i}] {folder.name} ({n_videos} videos)")
    print("  [A] All samples")

    while True:
        choice = input("\nSelect a sample to process (1, 2 or A) and press Enter: ").strip().lower()
        if choice == "a":
            return folders
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(folders):
                return [folders[idx - 1]]
        print("Invalid choice. Please try again.")


def folder_to_day_str(folder: Path) -> str:
    return folder.name.replace("SPT_", "")


def main():
    folders = list_available_samples(SAMPLES_DIR)

    if not folders:
        print(f"No demo sample folders found in: {SAMPLES_DIR}")
        print("Expected folders like: SPT_YYYYMMDD")
        sys.exit(1)

    selected_folders = choose_samples_interactive(folders)
    selected_days = [folder_to_day_str(f) for f in selected_folders]


    selection_file = CSV_DESTINATION / "selected_samples.txt"
    selection_file.parent.mkdir(parents=True, exist_ok=True)
    with open(selection_file, "w", encoding="utf-8") as f:
        for d in selected_days:
            f.write(d + "\n")

    print("\nSelected samples:", [f.name for f in selected_folders])
    print("Days to process:", selected_days)

    CSV_DESTINATION.mkdir(parents=True, exist_ok=True)

    total_start = time.time()

    max_workers = min(MAX_WORKERS_ENV, len(selected_days))
    print(f"\nUsing max_workers = {max_workers}")

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_day = {executor.submit(process_one_day, d): d for d in selected_days}

        for future in as_completed(future_to_day):
            day_str = future_to_day[future]
            try:
                day_str_done, elapsed = future.result()
                results.append((day_str_done, elapsed))
            except Exception as e:
                print(f"⚠️ Error processing sample {day_str}: (x_x) {e}")

    total_end = time.time()

    print("Summary by sample:")
    for d, elapsed in sorted(results):
        print(f"  {d}: {elapsed/60:.2f} minutes")
    print("(*^_^*) Indexing DONE. (*^_^*)")


if __name__ == "__main__":
    main()
