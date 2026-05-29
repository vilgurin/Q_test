## Overview
The directory contains the solution for Task 2 - tabular data regression task. 
The objective of the task is to predict target based on 53 features.
The solution includes Exploratory Data Analysis (EDA) notebook and scripts for training and inference.

## Solution
The Exploratory Data Analysis successfully identified the exact analytical solution to the problem:

$target = feature_6^2 + feature_7$

There is no need to approximate the target with a machine learning algorithms, thus the model is implemented as an analytical deterministic algorithm.

## Repository Structure
* eda.ipynb - Exploratory Data Analysis
* models.py - Contains the AnalyticalModel class.
* train.py - Script to validate the rule and serialize the model artifact.
* predict.py - Inference script to generate final predictions.
* requirements.txt - Python package dependencies.
* predictions.csv - The final inference output on the hidden test set.

## Project Setup

1. Create and activate a virtual environment:

    python3 -m venv venv

    source venv/bin/activate


2. Install dependencies:
    pip install -r requirements.txt

## Execution Guide

The scripts are designed to be executed directly from the terminal. 

### 1. Training & Serialization
To validate the representation and generate the model.pkl artifact, run:

    python train.py

### 2. Inference
To load the artifact and generate predictions on the hidden test set, run:

    python predict.py
