import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import cv2 as cv

from dotenv import load_dotenv

# ============== IMPORTS RETINANET =============
from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# ============== U-NET / segmentation_models =============
import tensorflow as tf
from segmentation_models.metrics import iou_score
from PIL import Image


# =========================
# LOAD .env (robust)
# =========================
CURRENT_DIR = Path(__file__).resolve().parent   # .../src
REPO_ROOT = CURRENT_DIR.parent                 # repo root
load_dotenv(dotenv_path=REPO_ROOT / ".env")

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "."))
PROJECT_ROOT = (REPO_ROOT / PROJECT_ROOT).resolve() if not PROJECT_ROOT.is_absolute() else PROJECT_ROOT.resolve()

CSV_OUTPUT_DIR = (PROJECT_ROOT / os.getenv("CSV_OUTPUT_DIR", "results/metadati")).resolve()
RESULTS_DIR = (PROJECT_ROOT / os.getenv("RESULTS_DIR", "results")).resolve()

MODEL_RETINANET = os.getenv("MODEL_RETINANET", "").strip()
MODEL_UNET = os.getenv("MODEL_UNET", "").strip()

if not MODEL_RETINANET:
    sys.exit("MODEL_RETINANET not defined in .env")

MODEL_RETINANET = str((PROJECT_ROOT / MODEL_RETINANET).resolve()) if not Path(MODEL_RETINANET).is_absolute() else MODEL_RETINANET
MODEL_UNET = str((PROJECT_ROOT / MODEL_UNET).resolve()) if (MODEL_UNET and not Path(MODEL_UNET).is_absolute()) else MODEL_UNET


SELECTION_FILE = (CSV_OUTPUT_DIR / "selected_samples.txt").resolve()
if not SELECTION_FILE.exists():
    sys.exit(f"Selection file not found: {SELECTION_FILE}\nRun Step 1 first: python src/index_frame.py")

with open(SELECTION_FILE, "r", encoding="utf-8") as f:
    SELECTED_DAYS = [line.strip() for line in f if line.strip()]

if not SELECTED_DAYS:
    sys.exit(f"Selection file is empty: {SELECTION_FILE}")

print("Selected samples loaded:", SELECTED_DAYS)


# ================== PARAMETERS ==================
THRESH = 0.20
BATCH_SIZE_RETINA = 32
BATCH_SIZE_UNET = 128
POSITIVE_CLASS = 0

WHITE = (255, 255, 255)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)


def mask_to_rgb(pred_mask, threshold=0.5):
    h, w, c = pred_mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)

    max_indices = np.argmax(pred_mask, axis=-1)
    max_probs = np.max(pred_mask, axis=-1)
    valid_pixels = (max_probs > threshold) & (max_indices != 0)

    rgb_mask[valid_pixels & (max_indices == 1)] = WHITE
    rgb_mask[valid_pixels & (max_indices == 2)] = RED
    rgb_mask[valid_pixels & (max_indices == 3)] = YELLOW
    rgb_mask[valid_pixels & (max_indices == 4)] = GREEN

    return rgb_mask



try:
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
except Exception:
    pass



print("\nLoading RetinaNet...")
detector = models.load_model(MODEL_RETINANET, backbone_name="resnet50")
print("RetinaNet loaded.\n")


unet_model = None
if MODEL_UNET and Path(MODEL_UNET).exists():
    try:
        print("Loading U-Net...")
        custom_objects = {"iou_score": iou_score}
        tf.keras.backend.set_image_data_format("channels_last")
        unet_model = tf.keras.models.load_model(MODEL_UNET, custom_objects=custom_objects, compile=False)
        print("U-Net loaded.")
    except Exception as e:
        print(f"ERROR loading U-Net. Continuing without it: {e}")
        unet_model = None
else:
    print("U-Net path missing/not found. Continuing without U-Net.")


BACKGROUND_IMAGE_PATH = os.getenv("BACKGROUND_IMAGE_PATH", "").strip()
if BACKGROUND_IMAGE_PATH:
    bg_path = (PROJECT_ROOT / BACKGROUND_IMAGE_PATH).resolve() if not Path(BACKGROUND_IMAGE_PATH).is_absolute() else Path(BACKGROUND_IMAGE_PATH)
else:
    bg_path = None

if bg_path and bg_path.exists():
    try:
        _bg_pil = Image.open(str(bg_path)).convert("RGB")
        _bg_pil = _bg_pil.resize((520, 520))
        background_image_np = np.array(_bg_pil)
        print(f"Using background image for U-Net: {bg_path}")
    except Exception as _e:
        print(f"ERROR loading BACKGROUND_IMAGE_PATH. Using black background. Detail: {_e}")
        background_image_np = np.zeros((520, 520, 3), dtype=np.uint8)
else:
    print("BACKGROUND_IMAGE_PATH not defined or not found. Using black background.")
    background_image_np = np.zeros((520, 520, 3), dtype=np.uint8)


def process_one_day_retina(day_str: str):

    day_dir = RESULTS_DIR / day_str

    in_frames_dir = day_dir / "frames_detected_by_lstm"
    csv_frames = day_dir / f"lstm_predictions_by_frame_{day_str}.csv"
    csv_intervals = day_dir / f"lstm_explosion_dates_{day_str}.csv"

    if not in_frames_dir.exists():
        print(f"[{day_str}] Missing folder: {in_frames_dir} (run LSTM step first). Skipping.")
        return
    if not csv_frames.exists() or not csv_intervals.exists():
        print(f"[{day_str}] Missing LSTM CSV files. Skipping.")
        return

    frames_out_dir = day_dir / "frames_with_bounding_box"
    masks_out_dir = day_dir / "frames_predicted_masks"
    frames_out_dir.mkdir(parents=True, exist_ok=True)
    masks_out_dir.mkdir(parents=True, exist_ok=True)

    print("\n======================")
    print(f"Day sample: {day_str}")
    print(f"Input frames: {in_frames_dir}")
    print(f"Output bbox : {frames_out_dir}")
    print(f"Output masks: {masks_out_dir}")

    df_frames = pd.read_csv(csv_frames)
    df_intervals = pd.read_csv(csv_intervals)

    if df_frames.empty:
        print(f"[{day_str}] lstm_predictions_by_frame is empty. Skipping.")
        return

    df_frames = df_frames.sort_values("global_frame_idx").reset_index(drop=True)
    df_frames.columns = [c.strip() for c in df_frames.columns]

    # Map gid
    all_frames = sorted(in_frames_dir.glob("*.jpg"))
    if not all_frames:
        print(f"[{day_str}] No frames in {in_frames_dir}. Skipping.")
        return

    frame_map = {}
    for p in all_frames:
        try:
            gid = int(p.stem.split("_")[-1])
            frame_map[gid] = p
        except Exception:
            continue


    mascaras_512 = []   
    grupo_frames = []  
    bboxes_by_frame = []   
    rows_retina = []      
    white_px_list, red_px_list, yel_px_list, gry_px_list = [], [], [], []

    # Retina batch
    batch_in = []
    batch_scales = []
    batch_draw = []
    batch_name_frame = []
    batch_path_folder = []
    batch_global_ids = []
    batch_day_strs = []
    batch_event_ids = []

    def procesar_batch():
        nonlocal batch_in, batch_scales, batch_draw, batch_name_frame
        nonlocal batch_path_folder, batch_global_ids, batch_day_strs, batch_event_ids
        nonlocal rows_retina, mascaras_512, grupo_frames, bboxes_by_frame

        if not batch_in:
            return

        X = np.stack(batch_in, axis=0)
        boxes_b, scores_b, labels_b = detector.predict(X)

        for k in range(len(batch_in)):
            draw = batch_draw[k]  # RGB 520x520
            name_frame = batch_name_frame[k]
            path_fold = batch_path_folder[k]
            scale = batch_scales[k]
            gidx = batch_global_ids[k]
            day_k = batch_day_strs[k]
            event_id_k = batch_event_ids[k]

            boxes = boxes_b[k] / scale
            scores = scores_b[k]
            labels = labels_b[k]

            clean_image_for_unet = draw.copy()

            frame_bboxes = []
            num_boxes = []

            for box, score, label in zip(boxes, scores, labels):
                if score < THRESH:
                    break

                b = box.astype(int)
                x1, y1, x2, y2 = b
                draw_box(draw, b, color=(255,255,255), thickness=1)

                loc_bbox = ""
                cx= ((x1+x2)/2)
                if int(label) == 0:
                    if cx < 270:
                        loc_bbox = "S"
                        caption = f"South: {score:.2f}"
                    elif cx > 271:
                        loc_bbox = "N"
                        caption = f"North: {score:.2f}"
                    else:
                        loc_bbox = "ANN"
                        caption = f"Anomala: {score:.2f}"
                else:
                    caption = f"Label {int(label)}: {score:.2f}"

                draw_caption(draw, b, caption)

                value_x = int(x1 + 1)
                value_y = int(y1 + 1)
                value_xw = int(x2)
                value_yh = int(y2)

                num_boxes.append((value_x, value_y, value_xw, value_yh))

                if int(label) == 0:
                    frame_bboxes.append((value_x, value_y, value_xw, value_yh, loc_bbox))

            # Save bbox frame
            if len(num_boxes) > 0:
                out_img_path = frames_out_dir / f"{name_frame}.jpg"
                cv.imwrite(str(out_img_path), cv.cvtColor(draw, cv.COLOR_RGB2BGR))

            bboxes_by_frame.append(frame_bboxes)
            grupo_frames.append(name_frame)

            # Prepare U-Net input
            if unet_model is not None:
                bg_np = background_image_np.copy()
                mascara = bg_np.copy()
                for (xa, ya, xb, yb) in num_boxes:
                    mascara[ya:yb, xa:xb] = clean_image_for_unet[ya:yb, xa:xb]
                mascara_final = cv.resize(mascara, (512, 512), interpolation=cv.INTER_NEAREST)
                mascaras_512.append(mascara_final)

            if frame_bboxes:
                for (xa, ya, xb, yb, loc_bbox) in frame_bboxes:
                    rows_retina.append({
                        "global_frame_idx": gidx,
                        "day_str": day_k,
                        "event_id": event_id_k,
                        "frame_name": name_frame,
                        "video_path": path_fold,
                        "explosion_location": loc_bbox,
                        "bbox_x": xa,
                        "bbox_y": ya,
                        "bbox_xw": xb,
                        "bbox_yh": yb,
                    })
            else:
                rows_retina.append({
                    "global_frame_idx": gidx,
                    "day_str": day_k,
                    "event_id": event_id_k,
                    "frame_name": name_frame,
                    "video_path": path_fold,
                    "explosion_location": "",
                    "bbox_x": -1,
                    "bbox_y": -1,
                    "bbox_xw": -1,
                    "bbox_yh": -1,
                })

        batch_in.clear()
        batch_scales.clear()
        batch_draw.clear()
        batch_name_frame.clear()
        batch_path_folder.clear()
        batch_global_ids.clear()
        batch_day_strs.clear()
        batch_event_ids.clear()

    print(f"[{day_str}] Processing explosions with RetinaNet...")

    for exp_id, row in df_intervals.reset_index().iterrows():
        try:
            start_id = int(row["init_global_frame_idx"])
            end_id = int(row["end_global_frame_idx"])
        except Exception:
            continue

        df_ev = df_frames[
            (df_frames["global_frame_idx"] >= start_id) &
            (df_frames["global_frame_idx"] <= end_id) &
            (df_frames["class_idx"] == POSITIVE_CLASS)
        ].sort_values("global_frame_idx").copy()

        if df_ev.empty:
            continue

        path_folder = f"{day_str}_exp_{exp_id:03d}"

        for _, fr in df_ev.iterrows():
            gidx = int(fr["global_frame_idx"])
            day_k = str(fr["day_str"])
            name_frame = f"{day_k}_{gidx:08d}"

            img_path = frame_map.get(gidx)
            if img_path is None:
                continue

            frame_bgr = cv.imread(str(img_path))
            if frame_bgr is None:
                continue

            imagen500_bgr = frame_bgr[0:520, 100:620]

            img_in = preprocess_image(imagen500_bgr)
            img_in, scale = resize_image(img_in)

            imagen500_rgb = cv.cvtColor(imagen500_bgr, cv.COLOR_BGR2RGB)

            batch_in.append(img_in)
            batch_scales.append(scale)
            batch_draw.append(imagen500_rgb.copy())
            batch_name_frame.append(name_frame)
            batch_path_folder.append(path_folder)
            batch_global_ids.append(gidx)
            batch_day_strs.append(day_k)
            batch_event_ids.append(exp_id)

            if len(batch_in) == BATCH_SIZE_RETINA:
                procesar_batch()

        procesar_batch()

    print(f"[{day_str}] RetinaNet done. Rows: {len(rows_retina)}")

    if unet_model is not None and mascaras_512:
        print(f"[{day_str}] U-Net on {len(mascaras_512)} masked frames...")

        n_frames = len(mascaras_512)
        for start in range(0, n_frames, BATCH_SIZE_UNET):
            end = min(n_frames, start + BATCH_SIZE_UNET)
            chunk_masks = mascaras_512[start:end]

            X = (np.stack(chunk_masks, axis=0).astype(np.float32)) / 255.0
            preds = unet_model.predict(X, verbose=0)

            for j in range(preds.shape[0]):
                i_global = start + j
                pred = preds[j]
                fname = grupo_frames[i_global]

                if np.max(pred[:, :, 1:]) > 0.1:
                    rgb_mask = mask_to_rgb(pred, threshold=0.5)
                    mask_path = masks_out_dir / f"{fname}.png"
                    cv.imwrite(str(mask_path), rgb_mask)

                w = int(np.sum(pred[:, :, 1] > 0.5))
                r = int(np.sum(pred[:, :, 2] > 0.5))
                y = int(np.sum(pred[:, :, 3] > 0.5))
                g = int(np.sum(pred[:, :, 4] > 0.5))

                white_px_list.append(w)
                red_px_list.append(r)
                yel_px_list.append(y)
                gry_px_list.append(g)

    if len(grupo_frames) == 0:
        print(f"[{day_str}] No frames processed. Skipping output CSV.")
        return

    if unet_model is None or (unet_model is not None and not mascaras_512):
        hot_area = [0] * len(grupo_frames)
        dispersal_area = [0] * len(grupo_frames)
        total_area = [0] * len(grupo_frames)
    else:
        white_px = np.array(white_px_list, dtype=np.int64)
        red_px = np.array(red_px_list, dtype=np.int64)
        yel_px = np.array(yel_px_list, dtype=np.int64)
        gry_px = np.array(gry_px_list, dtype=np.int64)

        total_px = white_px + red_px + yel_px + gry_px
        disp_px = red_px + yel_px + gry_px

        hot_area = white_px.astype(int).tolist()
        dispersal_area = disp_px.astype(int).tolist()
        total_area = total_px.astype(int).tolist()

    df_frame_areas = pd.DataFrame({
        "frame_name": grupo_frames,
        "hot_area": hot_area,
        "dispersal_area": dispersal_area,
        "total_area": total_area,
    }).drop_duplicates(subset=["frame_name"])

    df_retina = pd.DataFrame(rows_retina) if rows_retina else pd.DataFrame(
        columns=[
            "event_id", "global_frame_idx", "day_str", "frame_name", "video_path",
            "explosion_location", "bbox_x", "bbox_y", "bbox_xw", "bbox_yh"
        ]
    )

    df_retina = df_retina.merge(df_frame_areas, on="frame_name", how="left")
    df_retina[["hot_area", "dispersal_area", "total_area"]] = (
        df_retina[["hot_area", "dispersal_area", "total_area"]].fillna(0).astype(int)
    )

    out_csv = day_dir / f"retinanet_unet_predictions_by_frame_{day_str}.csv"
    df_retina.to_csv(out_csv, index=False, encoding="utf-8")

    print(f"[{day_str}] Saved: {out_csv.name}")
    print(f"[{day_str}] DONE RetinaNet+U-Net step.")


def main():
    for day_str in SELECTED_DAYS:
        process_one_day_retina(day_str)


if __name__ == "__main__":
    main()
