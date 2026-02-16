import os
import sys
import time
import gc
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from dotenv import load_dotenv

# LOAD .env

CURRENT_DIR = Path(__file__).resolve().parent   # .../src
REPO_ROOT = CURRENT_DIR.parent                 # repo root
load_dotenv(dotenv_path=REPO_ROOT / ".env")

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "."))
PROJECT_ROOT = (REPO_ROOT / PROJECT_ROOT).resolve() if not PROJECT_ROOT.is_absolute() else PROJECT_ROOT.resolve()
CSV_OUTPUT_DIR = (PROJECT_ROOT / os.getenv("CSV_OUTPUT_DIR", "metadati")).resolve()
RESULTS_DIR = (PROJECT_ROOT / os.getenv("RESULTS_DIR", "results")).resolve()

MODEL_LSTM = os.getenv("MODEL_LSTM", "")
if not MODEL_LSTM:
    sys.exit("Error: MODEL_LSTM is not defined in .env")

MODEL_LSTM = str((PROJECT_ROOT / MODEL_LSTM).resolve()) if not Path(MODEL_LSTM).is_absolute() else MODEL_LSTM

SELECTION_FILE = (CSV_OUTPUT_DIR / "selected_samples.txt").resolve()

if not SELECTION_FILE.exists():
    sys.exit(
        f"Selection file not found: {SELECTION_FILE}\n"
        f"Run Step 1 first: python src/index_frame.py"
    )

with open(SELECTION_FILE, "r", encoding="utf-8") as f:
    SELECTED_DAYS = [line.strip() for line in f if line.strip()]

if not SELECTED_DAYS:
    sys.exit(f"Selection file is empty: {SELECTION_FILE}")

print("Selected samples loaded:", SELECTED_DAYS)

CLASS_NAMES = ['EXPLOSION', 'NO_EXPLOSION', 'NO_SIGNAL', 'SPATTERING']
POSITIVE_CLASS = 0
MIN_RUN_LEN = 4
STRIDE = 50
MAX_GAP_BETWEEN_ZEROS = 3
MIN_CONSEC_ZEROS = 4

BASE_W, BASE_H = 704, 520

### change ROI's coordinates if change the training settings.
ROI_X, ROI_Y = 145, 70
ROI_W, ROI_H = 505, 450 

def preprocess_img(img_bgr, target_size):
    W, H = target_size
    img = cv2.resize(img_bgr, (W, H)).astype(np.float32)
    return img / 255.0


def tiene_3_ceros_consecutivos(lst, min_consec=MIN_CONSEC_ZEROS):
    if len(lst) < min_consec:
        return False
    run = 1
    for i in range(1, len(lst)):
        if lst[i] == lst[i - 1] + 1:
            run += 1
            if run >= min_consec:
                return True
        else:
            run = 1
    return False


def frame_time_to_dt(s):
    date = s[:8]
    time_str = s[9:]
    dt = datetime.strptime(date + time_str[:6], "%Y%m%d%H%M%S")
    ms = int(time_str[6:])
    return dt.replace(microsecond=ms * 1000)


class VideoReaderCache:
    def __init__(self):
        self.caps = {}
        self.last_idx = {}

    def _get_cap(self, vpath):
        if vpath not in self.caps:
            cap = cv2.VideoCapture(vpath)
            if not cap.isOpened():
                self.caps[vpath] = None
                return None
            self.caps[vpath] = cap
            self.last_idx[vpath] = -1
        return self.caps[vpath]

    def read_frame(self, vpath, fidx):
        cap = self._get_cap(vpath)
        if cap is None:
            return None
        last = self.last_idx.get(vpath, -1)
        if last + 1 != fidx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
        ok, frame = cap.read()
        self.last_idx[vpath] = fidx
        return frame if ok else None

    def release_all(self):
        for cap in self.caps.values():
            if cap:
                cap.release()
        self.caps.clear()
        self.last_idx.clear()


video_cache = VideoReaderCache()


def load_preprocessed(i, vpaths, fidxs, target_size):
    vpath = str(vpaths[i]).replace("/", "\\")
    frame = video_cache.read_frame(vpath, int(fidxs[i]))
    if frame is None:
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.float32)

    img_std = frame[0:BASE_H, 0:BASE_W]
    img_clean = img_std[ROI_Y: ROI_Y + ROI_H, ROI_X: ROI_X + ROI_W]
    return preprocess_img(img_clean, target_size)

def process_one_day(day_str: str):
    csv_path = CSV_OUTPUT_DIR / f"SPT_{day_str}.csv"

    if not csv_path.exists():
        print(f"[{day_str}] CSV not found: {csv_path}")
        return

    out_dir = RESULTS_DIR / day_str

    out_frames_dir = out_dir / "frames_detected_by_lstm"
    out_csv_per_frame = out_dir / f"lstm_predictions_by_frame_{day_str}.csv"
    out_csv_intervals = out_dir / f"lstm_explosion_dates_{day_str}.csv"

    if out_csv_per_frame.exists():
        print(f"[{day_str}] Results already exist. Skipping...")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    out_frames_dir.mkdir(parents=True, exist_ok=True)

    print("\n======================")
    print(f"Sample {day_str}")
    print(f"Input CSV:        {csv_path}")
    print(f"Output dir:       {out_dir}")
    print(f"Frames class 0:   {out_frames_dir}")

    # 1) Read and validate CSV
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    required_cols = {
        "global_frame_idx", "day_str", "video_path",
        "video_name", "frame_idx_in_video",
        "frame_time", "timestamp_ms"
    }

    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        print(f"[{day_str}] Missing columns: {missing}. Skipping.")
        return

    if df.empty:
        print(f"[{day_str}] Empty CSV. Skipping.")
        return

    df = df.sort_values("global_frame_idx").reset_index(drop=True)
    N = len(df)
    print(f"[{day_str}] Frames in CSV: {N}")

    vpaths = df["video_path"].astype(str).tolist()
    fidxs = df["frame_idx_in_video"].astype(int).tolist()
    frame_times = df["frame_time"].astype(str).tolist()
    global_ids = df["global_frame_idx"].astype(int).tolist()

    # Load model
    model = tf.keras.models.load_model(MODEL_LSTM, compile=False)
    _, L, H, W, C = model.input_shape
    target_size = (W, H)
    n_classes = model.output_shape[-1]

    prob_sum = np.zeros((N, n_classes), dtype=np.float32)
    counts = np.zeros((N,), dtype=np.int32)

    t0 = time.time()

    for s in range(0, N - L + 1, STRIDE):
        window_idx = range(s, s + L)
        frames = [load_preprocessed(t, vpaths, fidxs, target_size) for t in window_idx]
        buffer_np = np.expand_dims(np.stack(frames), axis=0)

        preds = model.predict(buffer_np, verbose=0)[0]
        prob_sum[s:s + L] += preds
        counts[s:s + L] += 1

        if (s // STRIDE) % 20 == 0:
            print(f"   > Progress: {s}/{N} frames | {time.time() - t0:.1f}s")

    if (N - L) >= 0 and counts[N - 1] == 0:
        s_final = N - L
        frames = [load_preprocessed(t, vpaths, fidxs, target_size) for t in range(s_final, N)]
        buffer_np = np.expand_dims(np.stack(frames), axis=0)
        preds = model.predict(buffer_np, verbose=0)[0]
        prob_sum[s_final:N] += preds
        counts[s_final:N] += 1

    counts[counts == 0] = 1
    probs_avg = prob_sum / counts[:, None]
    pred_idx = np.argmax(probs_avg, axis=-1)

    # No-signal
    for i in range(N):
        if probs_avg[i, 2] > 0.5:
            pred_idx[i] = 2

    # Save per-frame CSV
    df["class_idx"] = pred_idx
    df["class_label"] = [CLASS_NAMES[idx] for idx in pred_idx]
    df.to_csv(out_csv_per_frame, index=False)

    # Explosion intervals
    zero_pos = np.where(pred_idx == POSITIVE_CLASS)[0]
    zero_runs = []
    if len(zero_pos) > 0:
        curr = [zero_pos[0]]
        for p in zero_pos[1:]:
            if p - curr[-1] <= MAX_GAP_BETWEEN_ZEROS:
                curr.append(p)
            else:
                zero_runs.append(curr)
                curr = [p]
        zero_runs.append(curr)

    valid_runs = [r for r in zero_runs if len(r) >= MIN_RUN_LEN and tiene_3_ceros_consecutivos(r)]

    rows = []
    saved_imgs = 0

    for run in valid_runs:
        for k in run:
            f_full = video_cache.read_frame(vpaths[k].replace("/", "\\"), fidxs[k])
            if f_full is not None:
                fname = f"{day_str}_{global_ids[k]:08d}.jpg"
                cv2.imwrite(str(out_frames_dir / fname), f_full)
                saved_imgs += 1

        i_idx, e_idx = run[0], run[-1]
        dt_i = frame_time_to_dt(frame_times[i_idx])
        dt_e = frame_time_to_dt(frame_times[e_idx])

        rows.append({
            "day_str": day_str,
            "video_name": df.iloc[i_idx]["video_name"],
            "init_global_frame_idx": global_ids[i_idx],
            "end_global_frame_idx": global_ids[e_idx],
            "init_frame_time": frame_times[i_idx],
            "end_frame_time": frame_times[e_idx],
            "num_tot_frames": len(run),
            "duration_sec": (dt_e - dt_i).total_seconds()
        })

    if not rows:
        rows.append({"day_str": day_str, "init_global_frame_idx": "None_detected", "num_tot_frames": 0})

    pd.DataFrame(rows).to_csv(out_csv_intervals, index=False)

    print(f"[{day_str}] DONE. Explosions: {len(valid_runs)} | Frames saved: {saved_imgs}")

    video_cache.release_all()
    gc.collect()


def main():
    print("LSTM outputs")
    print("CSV input dir:", CSV_OUTPUT_DIR)
    print("Results dir:", RESULTS_DIR)
    print("Selection file:", SELECTION_FILE)
    print("Model:", MODEL_LSTM)

    for day_str in SELECTED_DAYS:
        process_one_day(day_str)


if __name__ == "__main__":
    main()
