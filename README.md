# Traffic — Traffic Sign Classifier (CNN)
Train a small convolutional neural network to classify traffic sign images
from a directory-structured dataset (one folder per category).

## Dataset
Download the GTSRB dataset and extract it into the project folder:
```bash
curl -O https://cdn.cs50.net/ai/2023/x/projects/5/gtsrb.zip
unzip gtsrb.zip
```
Or download manually: https://cdn.cs50.net/ai/2023/x/projects/5/gtsrb.zip

## Overview
- `traffic.py` loads image data, builds a Keras CNN, trains and evaluates it,
  and can save the trained model.

## Requirements
- Python 3.8+
- `opencv-python`, `tensorflow` (or `keras`), `scikit-learn`, `numpy`

## Quick Start
```bash
cd traffic
python traffic.py gtsrb
python traffic.py gtsrb model.keras
```
