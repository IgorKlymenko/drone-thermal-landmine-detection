import os
from pathlib import Path
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

def stratify():
    # Use absolute paths
    base_path = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'sorted')))
    output_base = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'stratified')))
    
    # Collect samples and labels
    samples, labels = [], []
    for folder in os.listdir(base_path):
        folder_path = base_path / folder
        if os.path.isdir(folder_path):
            label = -1 if folder == "free_zone" else int(folder.replace("cm", ""))
            for img in os.listdir(folder_path):
                if img.lower().endswith(('.jpg', '.jpeg')):
                    samples.append(str(folder_path / img))
                    labels.append(label)
    
    # Create stratified splits
    indices = np.arange(len(samples))
    train_val_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=labels, random_state=42)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.125, stratify=[labels[i] for i in train_val_idx], random_state=42)
    
    # Copy files to split directories
    for split_name, split_idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        split_dir = output_base / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        for idx in split_idx:
            src_path = Path(samples[idx])
            dst_path = split_dir / src_path.name
            counter = 1
            while dst_path.exists():
                base, ext = os.path.splitext(src_path.name)
                dst_path = split_dir / f"{base}_{counter}{ext}"
                counter += 1
            shutil.copy2(src_path, dst_path)
    
    # Print statistics
    print(f"\nTotal samples: {len(samples)}")
    for split_name, split_idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        split_labels = [labels[i] for i in split_idx]
        unique_labels, counts = np.unique(split_labels, return_counts=True)
        print(f"\n{split_name.capitalize()} set ({len(split_idx)} samples):")
        for label, count in zip(unique_labels, counts):
            label_name = "free_zone" if label == -1 else f"{label}cm"
            print(f"{label_name}: {count} samples")

if __name__ == "__main__":
    stratify()