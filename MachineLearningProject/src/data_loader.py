import os
import yaml
import numpy as np
import albumentations as A
from PIL import Image
from glob import glob
from sklearn.model_selection import train_test_split

from src.texture_features import extract_all_features


def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_lc25000_features(feature_extractor, data_path, config):
    base = r'C:/Users/Oliver/Masters_thesis/datasets/LC25000/lung_colon_image_set/colon_image_sets'

    tumor_paths = sorted(glob(os.path.join(base, 'colon_aca/*.jp*g')))
    healthy_paths = sorted(glob(os.path.join(base, 'colon_n/*.jp*g')))

    X, y = [], []

    healthy_sample = list(healthy_paths[:2500]) + list(np.random.choice(healthy_paths, 2500, replace=False))
    for i, path in enumerate(healthy_sample[:5000]):
        img = np.array(Image.open(path).convert('RGB'))
        feats = feature_extractor(img)  # ROI auto-handled
        X.append(feats);
        y.append(0)
        if i % 1000 == 0: print(f"  Healthy: {i}/5000")

    for i, path in enumerate(tumor_paths[:5000]):
        img = np.array(Image.open(path).convert('RGB'))
        feats = feature_extractor(img)
        X.append(feats);
        y.append(1)
        if i % 1000 == 0: print(f"  Tumor: {i}/5000")

    X, y = np.array(X), np.array(y)
    print(f"✅ LC25000 Final: X.shape={X.shape}, y={np.bincount(y)}")
    return X, y


def load_crchgd_features(feature_extractor, data_path):
    base = os.path.join(data_path, "Graded colon tissue")  # Your root dataset
    tumor_paths = []
    grade_patterns = ['CRC_Grade__1__Well_Diff/20x', 'CRC_Grade__2__Mod_Diff/20x', 'CRC_Grade__3__Poorly_Diff/20x']

    for grade_pat in grade_patterns:
        grade_path = os.path.join(base, grade_pat)
        found = sorted(glob(os.path.join(grade_path, '*.jp*g')))
        tumor_paths.extend(found)
        print(f"  📁 {grade_pat}: {len(found)} images")

    healthy_path = os.path.join(base, 'Normal_Colon/20x')
    healthy_paths = sorted(glob(os.path.join(healthy_path, '*.jp*g')))

    X, y = [], []

    for i, path in enumerate(healthy_paths[:5000]):
        img = np.array(Image.open(path).convert('RGB'))
        feats = feature_extractor(img)
        X.append(feats)
        y.append(0)
        if i % 500 == 0:
            print(f"  Healthy: {i}/{len(healthy_paths)}")

    for i, path in enumerate(tumor_paths[:5000]):
        img = np.array(Image.open(path).convert('RGB'))
        feats = feature_extractor(img)
        X.append(feats)
        y.append(1)
        if i % 500 == 0:
            print(f"  Tumor: {i}/{len(tumor_paths)}")

    X, y = np.array(X), np.array(y)
    return X, y


def load_crchgd_balanced(feature_extractor, data_path, config):
    base = os.path.join(data_path, "Graded colon tissue")

    print("CRC-HGD Balanced")
    tumor_paths = []
    grade_patterns = [
        'CRC_Grade__1__Well_Diff/40x',
        'CRC_Grade__2__Mod_Diff/40x',
        'CRC_Grade__3__Poorly_Diff/40x'
    ]

    for grade_pat in grade_patterns:
        grade_path = os.path.join(base, grade_pat)
        found = sorted(glob(os.path.join(grade_path, '*.jp*g')))
        tumor_paths.extend(found)
        print(f"   {grade_pat}: {len(found)} images")

    print(f"✅ Total tumor images: {len(tumor_paths)}")

    X_tumor, y_tumor = [], []
    for i, path in enumerate(tumor_paths):
        img = np.array(Image.open(path).convert('RGB'))
        feats = feature_extractor(img)
        X_tumor.append(feats)
        y_tumor.append(1)
        if i % 100 == 0:
            print(f"   Tumor: {i}/{len(tumor_paths)}")



    healthy_path = os.path.join(base, 'Normal_Colon/40x')
    healthy_paths = sorted(glob(os.path.join(healthy_path, '*.jp*g')))

    print(f"   Original healthy images: {len(healthy_paths)}")
    print(f"   SEVERELY imbalanced! ({len(healthy_paths)} vs {len(tumor_paths)} tumor)")

    # Load original healthy features
    X_healthy_orig, y_healthy_orig = [], []
    for path in healthy_paths:
        img = np.array(Image.open(path).convert('RGB'))
        feats = feature_extractor(img)
        X_healthy_orig.append(feats)
        y_healthy_orig.append(0)

    print(f"Augmenting healthy tissue...")
    X_aug, y_aug = augment_healthy(healthy_paths, target_n=len(tumor_paths), config=config)


    X_balanced = np.vstack([X_tumor, X_healthy_orig, X_aug])
    y_balanced = np.concatenate([y_tumor, y_healthy_orig, y_aug])

    print(f"✅ Balanced dataset ready!")
    print(f"   Total samples: {len(X_balanced)}")
    print(f"   Tumor: {np.sum(y_balanced == 1)}")
    print(f"   Healthy (original + augmented): {np.sum(y_balanced == 0)}")
    print(f"   Class balance ratio: {np.sum(y_balanced == 0) / np.sum(y_balanced == 1):.2f}")
    print(f"   Feature shape: {X_balanced.shape}")

    return X_balanced, y_balanced

def augment_healthy(healthy_paths, target_n=560, config=None):
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=20, p=0.7),
        A.RandomBrightnessContrast(p=0.3, brightness_limit=0.2, contrast_limit=0.2),
        A.GaussNoise(p=0.2, std_range=(0.01, 0.05)),
        A.ElasticTransform(p=0.3, alpha=20, sigma=5),
        A.GridDistortion(p=0.2, num_steps=5, distort_limit=0.2),
    ], bbox_params=None)

    X_aug, y_aug = [], []
    healthy_cycles = (target_n // len(healthy_paths)) + 1

    img_count = 0
    for cycle in range(healthy_cycles):
        for path_idx, path in enumerate(healthy_paths):
            if img_count >= target_n:
                break

            try:
                img = np.array(Image.open(path).convert('RGB'))
                aug_img = transform(image=img)['image']
                feats = extract_all_features(aug_img, config)

                X_aug.append(feats)
                y_aug.append(0)

                img_count += 1

                if img_count % 100 == 0:
                    print(f"  ✓ Generated {img_count}/{target_n} augmented healthy samples")

            except Exception as e:
                print(f"  ⚠️ Error processing {path}: {e}")
                continue

        if img_count >= target_n:
            break

    X_aug = np.array(X_aug)
    y_aug = np.array(y_aug)

    print(f"✅ Augmentation complete: {X_aug.shape[0]} samples generated")
    print(f"   X_aug shape: {X_aug.shape} (samples, features)")
    print(f"   y_aug shape: {y_aug.shape}")
    print(f"   Feature vector: L2-normalized 139D (GLCM+LBP+GLRLM)")

    return X_aug, y_aug

def get_splits(X, y, config):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=config['train']['test_size'],
        random_state=config['train']['random_state'], stratify=y)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5,
        random_state=config['train']['random_state'], stratify=y_temp)

    print(f"✅ Splits - Train:{X_train.shape[0]}, Val:{X_val.shape[0]}, Test:{X_test.shape[0]}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
