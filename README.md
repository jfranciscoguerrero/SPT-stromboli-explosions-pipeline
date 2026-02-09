# Multi-stage Deep Learning for Real-Time Analysis of Explosive Eruptions

This repository contains the code and resources associated with the paper:

**“A Multi-Stage Deep Learning for Real-Time Analysis of Explosive Eruptions in Thermal Images of Stromboli Volcano”**

The project implements a multi-stage deep learning pipeline based on three CNN models for:

- Classification of thermal frames from SPT,
- Detection of explosive events (object detection),
- Segmentation of eruptive regions,

and the automatic extraction of physical parameters from thermal video data for real-time volcanic monitoring.

---
Repository structure

.
├── cnn_models/ # Trained models (classifier, detector, segmenter)
├── data/ # Sample data and metadata
│ ├── samples/ # Sample videos (subset of the dataset)
│ ├── metadata/ # CSV files with frame indexing / metadata
│ └── mask/ # Auxiliary resources (e.g., black image mask)
├── results/ # Output results (CSV, plots, logs, etc.)
├── src/ # Source code (.py scripts)
├── requirements.txt / environment.yml # environment setup
└── .env # environment local paths
└── README.md


## Installation

Using conda:

```bash
conda env create -f environment.yml
conda activate stromboli_dl

Using pip:

pip install -r requirements.txt
