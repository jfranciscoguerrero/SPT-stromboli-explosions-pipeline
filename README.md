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
├── notebooks/ # Notebooks with the neural network scripts
│ ├── retinanet/ RetinaNet scripts
├── data/ # Sample data and metadata
│ ├── samples/ # Sample videos (subset of the dataset)
│ ├── metadata/ # CSV files with frame indexing / metadata
│ └── mask/ # Auxiliary resources (e.g., black image mask)
├── results/ # Output results (CSV, plots, logs, etc.)
├── src/ # Source code (.py scripts)
├── requirements.txt / environment.yml /Dockerfile # environment setup.
└── .env # environment local paths.
└── README.md

Requirements

Python 3.8+ (recommended for TensorFlow 2.9)
Conda (optional)
Docker (optional, recommended for reproducibility)


Clone the repository

git clone git@github.com:jfranciscoguerrero/SPT-stromboli-explosions-pipeline.git

cd /SPT-stromboli-explosions-pipeline/

## Installation

1. Option A: Using Conda

Step 1: Create the environment

    Run in the repository root:

    conda env create -f environment.yml

Step 2: Activate the environment

    conda activate stromboli_dl

Step 3: Run the pipeline

    python src/run_pipeline.py

2. Option B: Using Pip

Step 1: Install dependencies

    (Optional) Create a virtual environment

    pip install -r requirements.txt

    python src/run_pipeline.py

3. Option C: using Docker

Step 1: Build the Docker image

    Linux/macOS:
    docker run -it --rm -v "$(pwd)":/app stromboli-pipeline

    Windows PowerShell:
    docker run -it --rm -v ${PWD}:/app stromboli-pipeline

Step 3: Run the pipeline inside the container
    
    python src/run_pipeline.py

4. The full dataset is hosted on Zenodo.

    Zenodo link (DOI): https://doi.org/10.5281/zenodo.18657032

    You can use this dataset to:

        Train the models provided in this repository,

        Reproduce the experiments described in the paper,

        Run your own experiments or personal projects based on thermal imagery.
        If you use this dataset in your work (research or personal projects), please cite the Zenodo DOI.
