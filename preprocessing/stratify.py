import os
from pathlib import Path
import random
import shutil
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

AUGMENT_FREE_ZONE = True

def augment_image(img):
    """Return a list of augmented PIL Images from the input image (same as step_1_preprocessing.ipynb)."""
    augmented = []
    # 1. Horizontal flip
    augmented.append(img.transpose(Image.FLIP_LEFT_RIGHT))
    # 2. Random rotation
    angle = random.choice([90, 180, 270])
    augmented.append(img.rotate(angle))
    # 3. Color jitter (simple brightness change)
    factor = random.uniform(0.7, 1.3)
    augmented.append(Image.fromarray(np.clip(np.array(img) * factor, 0, 255).astype(np.uint8)))
    return augmented

def copy_unique(src_path, dst_dir):
    """Copy src_path into dst_dir, renaming on name clashes. Returns the destination path."""
    dst_path = dst_dir / src_path.name
    counter = 1
    while dst_path.exists():
        base, ext = os.path.splitext(src_path.name)
        dst_path = dst_dir / f"{base}_{counter}{ext}"
        counter += 1
    shutil.copy2(src_path, dst_path)
    return dst_path

def stratify():
    # Use absolute paths
    base_path = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'sorted')))
    output_base = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'stratified')))
    random.seed(42)

    # Collect samples and labels (depth in cm, -1 for free zone)
    samples, labels = [], []
    for folder in sorted(os.listdir(base_path)):
        folder_path = base_path / folder
        if os.path.isdir(folder_path):
            label = -1 if folder == "free_zone" else int(folder.replace("cm", ""))
            for img in sorted(os.listdir(folder_path)):
                if img.lower().endswith(('.jpg', '.jpeg')):
                    samples.append(str(folder_path / img))
                    labels.append(label)

    # Create stratified splits: 70% train, 10% val, 20% test, balanced across depths
    indices = np.arange(len(samples))
    train_val_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=labels, random_state=42)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.125, stratify=[labels[i] for i in train_val_idx], random_state=42)

    # Remove old splits, then copy into <split>/<positive|negative> so Ultralytics can read the folder directly
    if output_base.exists():
        shutil.rmtree(output_base)
    for split_name, split_idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        for idx in split_idx:
            cls = 'negative' if labels[idx] == -1 else 'positive'
            split_dir = output_base / split_name / cls
            split_dir.mkdir(parents=True, exist_ok=True)
            copy_unique(Path(samples[idx]), split_dir)

    # Augment only train/negative after the split, so no augmented copy leaks into val or test
    if AUGMENT_FREE_ZONE:
        train_neg_dir = output_base / 'train' / 'negative'
        for src_file in sorted(train_neg_dir.iterdir()):
            base, ext = os.path.splitext(src_file.name)
            with Image.open(src_file) as img:
                for i, aug_img in enumerate(augment_image(img)):
                    aug_img.save(train_neg_dir / f"{base}_aug{i+1}{ext}")

    # Print statistics
    print(f"\nTotal samples: {len(samples)}")
    for split_name, split_idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        split_labels = [labels[i] for i in split_idx]
        unique_labels, counts = np.unique(split_labels, return_counts=True)
        print(f"\n{split_name.capitalize()} set ({len(split_idx)} samples):")
        for label, count in zip(unique_labels, counts):
            label_name = "free_zone" if label == -1 else f"{label}cm"
            print(f"{label_name}: {count} samples")
    for split_name in ['train', 'val', 'test']:
        counts = {cls: len(list((output_base / split_name / cls).iterdir())) for cls in ['positive', 'negative']}
        print(f"{split_name} on disk: {counts}")

if __name__ == "__main__":
    stratify()
