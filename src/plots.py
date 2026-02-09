#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def main():
    repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(dotenv_path=repo_root / ".env")

    pr_env = Path(os.getenv("PROJECT_ROOT", "."))
    project_root = pr_env.resolve() if pr_env.is_absolute() else (repo_root / pr_env).resolve()
    results_dir = (project_root / os.getenv("RESULTS_DIR", "results")).resolve()
    csv_out_dir = (project_root / os.getenv("CSV_OUTPUT_DIR", "results/metadati")).resolve()

    day_str = (csv_out_dir / "selected_samples.txt").read_text(encoding="utf-8").splitlines()[0].strip()
    day_dir = results_dir / day_str

    df_final = pd.read_csv(day_dir / f"final_report_{day_str}.csv")
    df_pf = pd.read_csv(day_dir / f"retinanet_unet_predictions_by_frame_{day_str}.csv")
    df_lstm = pd.read_csv(day_dir / f"lstm_predictions_by_frame_{day_str}.csv")

    out_html = day_dir / f"report_PLOTS_{day_str}.html"

    df_final.columns = [c.strip() for c in df_final.columns]
    df_pf.columns = [c.strip() for c in df_pf.columns]
    df_lstm.columns = [c.strip() for c in df_lstm.columns]

    crater_col = "crater" if "crater" in df_final.columns else ("craters" if "craters" in df_final.columns else None)
    if crater_col is None:
        df_final["crater"] = "ANN"
        crater_col = "crater"

    df_final["crater"] = df_final[crater_col].astype(str).replace({"S-N": "N-S", "-1": "ANN"}).fillna("ANN")
    label_map = {"N": "North", "S": "South", "N-S": "North-South", "ANN": "ANN"}
    df_final["crater_label"] = df_final["crater"].map(label_map).fillna(df_final["crater"])

    if "init_frame_datetime" in df_final.columns:
        df_final["start_dt"] = pd.to_datetime(df_final["init_frame_datetime"], errors="coerce")
    else:
        def _parse_ft(v):
            try:
                d, t = str(v).split("-")
                return pd.to_datetime(d + t[:6], format="%Y%m%d%H%M%S", errors="coerce")
            except:
                return pd.NaT
        df_final["start_dt"] = df_final.get("init_frame_time", pd.Series([None] * len(df_final))).apply(_parse_ft)

    df_final["duration_sec"] = pd.to_numeric(df_final.get("duration_sec", 0), errors="coerce").fillna(0)

    if "delta_value" not in df_final.columns:
        if "max_total_area" in df_final.columns and "max_hot_area" in df_final.columns:
            df_final["delta_value"] = pd.to_numeric(df_final["max_total_area"], errors="coerce") / pd.to_numeric(df_final["max_hot_area"], errors="coerce")
        else:
            df_final["delta_value"] = pd.NA
    df_final["delta_value"] = pd.to_numeric(df_final["delta_value"], errors="coerce")

    color_map = {"North": "#3989FF", "South": "#E81C45", "North-South": "#F08C19", "ANN": "#000000"}

    ccount = df_final.groupby("crater_label").size().reset_index(name="count")
    fig1 = px.bar(ccount, x="count", y="crater_label", orientation="h", text="count",
                  color="crater_label", color_discrete_map=color_map,
                  labels={"crater_label": "Crater", "count": "Occurrences"})
    fig1.update_traces(textposition="outside", showlegend=False)
    fig1.update_layout(xaxis_title="Occurrences", yaxis_title="Crater location")

    df2 = df_final.dropna(subset=["start_dt"]).copy()
    fig2 = px.scatter(df2, x="start_dt", y="duration_sec", color="crater_label",
                      color_discrete_map=color_map,
                      hover_data=["event_id"] if "event_id" in df2.columns else None,
                      labels={"start_dt": "Start explosion", "duration_sec": "Duration explosion", "crater_label": "Crater"})
    fig2.update_yaxes(rangemode="tozero", title_text="Seconds")
    fig2.update_xaxes(title_text="Time")


    time_map = dict(zip(df_lstm["global_frame_idx"].astype(int), df_lstm["frame_time"].astype(str)))
    df_pf["frame_time"] = df_pf["global_frame_idx"].astype(int).map(time_map) if "global_frame_idx" in df_pf.columns else ""

    def only_time(v):
        try:
            _, t = str(v).split("-")
            return f"{t[0:2]}:{t[2:4]}:{t[4:6]}.{t[6:9] if len(t)>=9 else '000'}"
        except:
            return ""

    df_pf["t_label"] = df_pf["frame_time"].apply(only_time)
    for c in ["hot_area", "dispersal_area", "total_area"]:
        df_pf[c] = pd.to_numeric(df_pf[c], errors="coerce").fillna(0) if c in df_pf.columns else 0
    if "frame_name" in df_pf.columns:
        df_pf = df_pf.drop_duplicates(subset=["frame_name"])
    if "event_id" not in df_pf.columns:
        df_pf["event_id"] = ""

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(name="hot_area", x=df_pf["t_label"], y=df_pf["hot_area"], marker_color="#F08C19",
                          customdata=df_pf["event_id"], hovertemplate="Timestamp: %{x}<br>Event: %{customdata}<br>hot_area: %{y}<extra></extra>"))
    fig3.add_trace(go.Bar(name="dispersal_area", x=df_pf["t_label"], y=df_pf["dispersal_area"], marker_color="#00bd45",
                          customdata=df_pf["event_id"], hovertemplate="Timestamp: %{x}<br>Event: %{customdata}<br>dispersal_area: %{y}<extra></extra>"))
    fig3.add_trace(go.Bar(name="total_area", x=df_pf["t_label"], y=df_pf["total_area"], marker_color="#7f7f7f",
                          customdata=df_pf["event_id"], hovertemplate="Timestamp: %{x}<br>Event: %{customdata}<br>total_area: %{y}<extra></extra>"))
    fig3.update_layout(barmode="group", xaxis_title="Timestamp", yaxis_title="Area in (Pixels)")
    fig3.update_yaxes(rangemode="tozero")
    fig3.update_xaxes(tickangle=-90)

    df4 = df_final.dropna(subset=["start_dt", "delta_value"]).copy()
    fig4 = px.scatter(df4, x="start_dt", y="delta_value", color="crater_label",
                      color_discrete_map=color_map,
                      hover_data=["event_id"] if "event_id" in df4.columns else None,
                      labels={"start_dt": "Timestamp", "delta_value": "Delta values", "crater_label": "Crater"})
    fig4.update_yaxes(rangemode="tozero", title_text="Delta values")
    fig4.update_xaxes(title_text="Timestamp")

    fig = make_subplots(
        rows=4, cols=1, vertical_spacing=0.10,
        subplot_titles=(
            "Number of explosions by crater location",
            "Start time of each explosion detected",
            "Areas per frame (all detected explosion frames)",
            "Delta values of each explosion",
        )
    )

    for tr in fig1.data: fig.add_trace(tr, 1, 1)
    for tr in fig2.data: fig.add_trace(tr, 2, 1)
    for tr in fig3.data: fig.add_trace(tr, 3, 1)
    for tr in fig4.data: fig.add_trace(tr, 4, 1)
    fig.update_xaxes(title_text="Occurrences", row=1, col=1)
    fig.update_yaxes(title_text="Crater location", row=1, col=1)

    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Seconds", row=2, col=1)

    fig.update_xaxes(title_text="Timestamp", row=3, col=1)
    fig.update_yaxes(title_text="Area in (Pixels)", row=3, col=1)

    fig.update_xaxes(title_text="Timestamp", row=4, col=1)
    fig.update_yaxes(title_text="Delta values", row=4, col=1)

    seen = set()
    for tr in fig.data:
        if tr.name in seen:
            tr.showlegend = False
        else:
            tr.showlegend = True
            seen.add(tr.name)

    fig.update_layout(
        title=dict(text=f"Stromboli SPT Plot report - day {day_str}", x=0.5, y=0.98, xanchor="center", yanchor="top"),
        height=1700,
        margin=dict(l=70, r=40, t=160, b=60),
        legend=dict(orientation="h", yanchor="top", y=1.05, xanchor="center", x=0.5),
    )

    fig.write_html(str(out_html), include_plotlyjs="cdn")


if __name__ == "__main__":
    main()
