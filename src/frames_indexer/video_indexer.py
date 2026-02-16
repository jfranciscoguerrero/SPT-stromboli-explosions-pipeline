import os
import csv
from pathlib import Path
from datetime import datetime
import cv2
from .string_to_date import (convert_start_date_from_filename)

def list_videos(folder: Path):
    if not folder.exists():
        return []
    return sorted([
        f.name for f in folder.iterdir()
        if f.is_file() and not f.name.startswith(".")
    ])


def index_one_day(
    day_str: str,
    samples_dir: Path,
    csv_destination: Path,
    start_id: int = 0,
):

    samples_dir = Path(samples_dir)
    csv_destination = Path(csv_destination)

    day_path = samples_dir / f"SPT_{day_str}"

    print(f"\n Day {day_str}")
    print("Day folder:", day_path)

    csv_destination.mkdir(parents=True, exist_ok=True)
    csv_path = csv_destination / f"SPT_{day_str}.csv"

    if csv_path.exists():
        print(f" CSV already exists for {day_str}, skipping.")
        return start_id

    video_names = list_videos(day_path)
    if not video_names:
        print("No videos found in", day_path)
        return start_id

    print("Videos found:", len(video_names))
    print("CSV output:", csv_path)

    currentframe = start_id

    with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow([
            "global_frame_idx",
            "day_str",
            "video_path",
            "video_name",
            "frame_idx_in_video",
            "frame_time",
            "timestamp_ms",
        ])

        for video_name in video_names:
            full_video_path = day_path / video_name

            print("  Processing video:", video_name)


            fecha_inicial = convert_start_date_from_filename(video_name)
            t_ms = int(fecha_inicial.timestamp() * 1000)

            frame_idx = 0
            cam = cv2.VideoCapture(str(full_video_path))
            if not cam.isOpened():
                print("Could not open:", full_video_path)
                continue

            while True:
                ret, frame = cam.read()
                if not ret:
                    break

                dt = datetime.fromtimestamp(t_ms / 1000.0)
                frame_time_str = dt.strftime("%Y%m%d-%H%M%S%f")[:-3]  # milliseconds

                writer.writerow([
                    currentframe,
                    day_str,
                    str(full_video_path.as_posix()),
                    video_name,
                    frame_idx,
                    frame_time_str,
                    t_ms,
                ])

                currentframe += 1
                frame_idx += 1
                t_ms += 500  # 2 fps

            cam.release()

    print(f"  Frames indexed for {day_str}: {currentframe - start_id}")
    return currentframe
