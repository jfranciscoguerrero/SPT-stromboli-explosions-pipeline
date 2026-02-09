import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv


def frame_time_to_dt(s: str) -> Optional[datetime]:
    # expects: YYYYMMDD-HHMMSSmmm
    try:
        date = s[:8]
        time_str = s[9:]
        dt = datetime.strptime(date + time_str[:6], "%Y%m%d%H%M%S")
        ms = int(time_str[6:])
        return dt.replace(microsecond=ms * 1000)
    except Exception:
        return None


def read_selected_days(selection_file: Path) -> List[str]:
    if not selection_file.exists():
        raise SystemExit(
            f"Selection file not found: {selection_file}\n"
            f"Run Step 1 first (index_frame)."
        )
    with selection_file.open("r", encoding="utf-8") as f:
        days = [line.strip() for line in f if line.strip()]
    days = [d for d in days if len(d) == 8 and d.isdigit()]
    days = sorted(days)
    if not days:
        raise SystemExit(f"Selection file is empty/invalid: {selection_file}")
    return days


def main():
    # Load .env
    current_dir = Path(__file__).resolve().parent   # .../src
    repo_root = current_dir.parent                  # repo root
    load_dotenv(dotenv_path=repo_root / ".env")

    project_root_env = Path(os.getenv("PROJECT_ROOT", "."))
    project_root = project_root_env.resolve() if project_root_env.is_absolute() else (repo_root / project_root_env).resolve()

    csv_output_dir = (project_root / os.getenv("CSV_OUTPUT_DIR", "results/metadati")).resolve()
    results_dir = (project_root / os.getenv("RESULTS_DIR", "results")).resolve()
    selection_file = (csv_output_dir / "selected_samples.txt").resolve()

    selected_days = read_selected_days(selection_file)

    day_str = selected_days[0]

    day_dir = results_dir / day_str
    if not day_dir.exists():
        raise SystemExit(f"Day folder not found: {day_dir}")

    # Inputs
    lstm_events_csv = day_dir / f"lstm_explosion_dates_{day_str}.csv"
    lstm_frames_csv = day_dir / f"lstm_predictions_by_frame_{day_str}.csv"
    retina_unet_csv = day_dir / f"retinanet_unet_predictions_by_frame_{day_str}.csv"

    if not lstm_events_csv.exists():
        raise SystemExit(f"Missing: {lstm_events_csv}")
    if not retina_unet_csv.exists():
        raise SystemExit(f"Missing: {retina_unet_csv}")

    # lstm_predictions_by_frame
    df_time = None
    if lstm_frames_csv.exists():
        df_time = pd.read_csv(lstm_frames_csv)
        df_time.columns = [c.strip().lower() for c in df_time.columns]
        if "global_frame_idx" in df_time.columns and "frame_time" in df_time.columns:
            df_time = df_time[["global_frame_idx", "frame_time"]].copy()
            df_time["global_frame_idx"] = df_time["global_frame_idx"].astype(int)
            df_time["frame_time"] = df_time["frame_time"].astype(str)
        else:
            df_time = None

    df_events = pd.read_csv(lstm_events_csv)
    df_events.columns = [c.strip().lower() for c in df_events.columns]

    df_ret = pd.read_csv(retina_unet_csv)
    df_ret.columns = [c.strip().lower() for c in df_ret.columns]

    # Normalizar
    if "event_id" not in df_ret.columns:
        raise SystemExit("retinanet_unet_predictions_by_frame must contain 'event_id' column.")
    if "global_frame_idx" not in df_ret.columns:
        raise SystemExit("retinanet_unet_predictions_by_frame must contain 'global_frame_idx' column.")
    if "explosion_location" not in df_ret.columns:
        raise SystemExit("retinanet_unet_predictions_by_frame must contain 'explosion_location' column.")

    df_ret["event_id"] = df_ret["event_id"].astype(int)
    df_ret["global_frame_idx"] = df_ret["global_frame_idx"].astype(int)
    df_ret["explosion_location"] = df_ret["explosion_location"].fillna("").astype(str)

    for col in ["hot_area", "dispersal_area", "total_area"]:
        if col not in df_ret.columns:
            df_ret[col] = 0
        df_ret[col] = pd.to_numeric(df_ret[col], errors="coerce").fillna(0).astype(int)

    if "bbox_x" in df_ret.columns:
        df_ret_valid = df_ret[df_ret["bbox_x"] != -1].copy()
    else:
        df_ret_valid = df_ret.copy()

    time_map: Dict[int, str] = {}
    if df_time is not None and not df_time.empty:
        time_map = dict(zip(df_time["global_frame_idx"].tolist(), df_time["frame_time"].tolist()))

    def get_time_str(gidx: int) -> str:
        return time_map.get(gidx, "")


    rows: List[Dict[str, Any]] = []

 
    if "init_global_frame_idx" not in df_events.columns or "end_global_frame_idx" not in df_events.columns:
        raise SystemExit("lstm_explosion_dates must contain init_global_frame_idx and end_global_frame_idx")


    df_events = df_events.reset_index().rename(columns={"index": "event_id"})
    df_events["event_id"] = df_events["event_id"].astype(int)

    for _, ev in df_events.iterrows():
        event_id = int(ev["event_id"])

        # Caso None_detected
        try:
            ev_init = int(ev["init_global_frame_idx"])
            ev_end = int(ev["end_global_frame_idx"])
        except Exception:
            rows.append({
                "day_str": day_str,
                "event_id": event_id,
                "crater": "-1",
                "init_global_frame_idx": "None_detected",
                "end_global_frame_idx": "None_detected",
                "init_frame_time": "",
                "end_frame_time": "",
                "init_frame_datetime": "",
                "end_frame_datetime": "",
                "num_tot_frames": 0,
                "duration_sec": 0,
                "max_hot_area": 0,
                "max_dispersal_area": 0,
                "max_total_area": 0
            })
            continue

        # Retina
        sub = df_ret_valid[df_ret_valid["event_id"] == event_id].copy()

        # Si no hay bbox detections
        if sub.empty:
            init_time = str(ev.get("init_frame_time", "")) if "init_frame_time" in ev else ""
            end_time = str(ev.get("end_frame_time", "")) if "end_frame_time" in ev else ""
            dt_i = frame_time_to_dt(init_time) if init_time else None
            dt_e = frame_time_to_dt(end_time) if end_time else None

            rows.append({
                "day_str": day_str,
                "event_id": event_id,
                "crater": "-1",
                "init_global_frame_idx": ev_init,
                "end_global_frame_idx": ev_end,
                "init_frame_time": init_time,
                "end_frame_time": end_time,
                "init_frame_datetime": dt_i.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] if dt_i else "",
                "end_frame_datetime": dt_e.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] if dt_e else "",
                "num_tot_frames": int(ev.get("num_tot_frames", 0)) if "num_tot_frames" in ev else (ev_end - ev_init + 1),
                "duration_sec": float(ev.get("duration_sec", 0)) if "duration_sec" in ev else 0,
                "max_hot_area": 0,
                "max_dispersal_area": 0,
                "max_total_area": 0
            })
            continue

        craters = sorted([c for c in sub["explosion_location"].unique().tolist() if c in ("N", "S", "ANN")])

        if not craters:
            craters = ["-1"]

        for crater in craters:
            subc = sub[sub["explosion_location"] == crater].copy() if crater != "-1" else sub.copy()

            if subc.empty:
                continue

            frames = sorted(subc["global_frame_idx"].unique().tolist())
            c_init = int(min(frames))
            c_end = int(max(frames))

            init_time = get_time_str(c_init)
            end_time = get_time_str(c_end)

            if not init_time and "init_frame_time" in ev:
                init_time = str(ev.get("init_frame_time", ""))
            if not end_time and "end_frame_time" in ev:
                end_time = str(ev.get("end_frame_time", ""))

            dt_i = frame_time_to_dt(init_time) if init_time else None
            dt_e = frame_time_to_dt(end_time) if end_time else None

            duration_sec = (dt_e - dt_i).total_seconds() if (dt_i and dt_e) else 0.0

            max_hot = int(subc["hot_area"].max())
            max_disp = int(subc["dispersal_area"].max())
            max_tot = int(subc["total_area"].max())

            delta_value = (max_tot / max_hot) if max_hot > 0 else float("nan")

            rows.append({
                "day_str": day_str,
                "event_id": event_id,
                "crater": crater,
                "init_global_frame_idx": c_init,
                "end_global_frame_idx": c_end,
                "init_frame_time": init_time,
                "end_frame_time": end_time,
                "init_frame_datetime": dt_i.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] if dt_i else "",
                "end_frame_datetime": dt_e.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] if dt_e else "",
                "num_tot_frames": len(frames),
                "duration_sec": duration_sec,
                "max_value_hot_area": max_hot,
                "max_value_dispersal_area": max_disp,
                "max_value_total_area": max_tot,
                "delta_value": delta_value
            })



    df_final = pd.DataFrame(rows)

    out_csv = day_dir / f"final_report_{day_str}.csv"
    df_final.to_csv(out_csv, index=False, encoding="utf-8")


    print("Final report generated")
    print("Day:", day_str)
    print("Input:", lstm_events_csv.name, "+", retina_unet_csv.name)
    print("Output:", out_csv)
    print("Rows:", len(df_final))
    print("")


if __name__ == "__main__":
    main()
