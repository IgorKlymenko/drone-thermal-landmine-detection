from ultralytics import YOLO
import pathlib


### Initial lcoal training testing before experiment in google collab
DATA_DIR = pathlib.Path(__file__).parent  
TEST_DIR = DATA_DIR / "test.yaml"

model = YOLO("yolo11n-cls.pt")
results = model.train(
    data=str(DATA_DIR),
    epochs=5,
    imgsz=320,
    batch=32,
    dropout=0.75,
    save=True
)

# Validate on test set
metrics = model.val(save_json=True)