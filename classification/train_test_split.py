from ultralytics import YOLO
import pathlib

### Train with the best config from the dropout sweep (0.3) on the depth-stratified
### train/val/test split from preprocessing/stratify.py, then score once on the untouched test set
BASE_DIR = pathlib.Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "data" / "stratified"

model = YOLO("yolo11n-cls.pt")
model.train(
    data=str(DATA_DIR),
    epochs=20,
    imgsz=320,
    batch=32,
    dropout=0.3,
    seed=42,
    project=str(BASE_DIR / "runs" / "classify"),
    name="stratified_test",
    exist_ok=True,
)

# Best weights were picked on val; test is only touched here
best = YOLO(BASE_DIR / "runs" / "classify" / "stratified_test" / "weights" / "best.pt")
metrics = best.val(
    data=str(DATA_DIR),
    split="test",
    imgsz=320,
    project=str(BASE_DIR / "runs" / "classify"),
    name="stratified_test_eval",
    exist_ok=True,
)
print(f"TEST top-1 accuracy: {metrics.top1:.4f}")
