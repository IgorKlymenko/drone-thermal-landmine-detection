import pathlib
import itertools
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO

# Hyperparams
# lr = 0.001429 default
# momentum = 0.9 default
DROPOUTS      = [0.3, 0.5, 0.8]
BATCH_SIZE    = 32
IMG_SIZE      = 320
EPOCHS        = 20

## trained in Google Collab, so the dir folder is content
DATA_YAML = pathlib.Path("/content")
records = []


for dropout in DROPOUTS:
    print(f"\n>>> Training with dropout={dropout} ...")

    model = YOLO("yolo11n-cls.pt")

    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        dropout=dropout,
        save=True,           
    )

    # Validation
    metrics = model.val()