import sys
print(f"Python interpreter being used: {sys.executable}")
import torch
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from collections import Counter
from torchvision import transforms
import random
from sklearn.cluster import KMeans

# Load the full LCZ map
lcz_map_path = "../dataset/milan/LCZ_MAP.tif"
lcz_map_full = tifffile.imread(lcz_map_path)
unique_classes_full, class_counts_full = np.unique(lcz_map_full, return_counts=True)
total_pixels_full = np.sum(class_counts_full)
class_percentages_full = (class_counts_full / total_pixels_full) * 100
H_full, W_full = lcz_map_full.shape

print("--- Full LCZ Map Class Distribution (Before Stratification) ---")
for cls, count in zip(unique_classes_full, class_counts_full):
    print(f"Class {cls}: {count} pixels ({class_percentages_full[np.where(unique_classes_full == cls)[0][0]]:.2f}%)")

# --- Function for Stratified Splitting ---
def create_stratified_split_coords(lcz_map, patch_size, stride, train_ratio=0.7, seed=42):
    unique_classes = np.unique(lcz_map)
    train_tile_coords = []
    rng = np.random.RandomState(seed)
    H, W = lcz_map.shape
    tile_coords_by_class = {cls: [] for cls in unique_classes}
    for r in range(0, H - patch_size + 1, stride):
        for c in range(0, W - patch_size + 1, stride):
            top_left_class = lcz_map[r, c]
            tile_coords_by_class[top_left_class].append((r, c))
    for cls in unique_classes:
        coords = tile_coords_by_class[cls]
        rng.shuffle(coords)
        train_split = int(train_ratio * len(coords))
        train_tile_coords.extend(coords[:train_split])
    return train_tile_coords

# --- Perform Stratified Split ---
patch_size = 64
stride = 32
train_tile_coords_stratified = create_stratified_split_coords(lcz_map_full, patch_size, stride)

# --- Create a simplified Dataset for label extraction ---
class SimpleLCZDataset(Dataset):
    def __init__(self, lcz_map_path, patch_size, tile_coords):
        self.lcz_map_path = lcz_map_path
        self.patch_size = patch_size
        self.tile_coords = tile_coords
        self.lcz_map = tifffile.imread(lcz_map_path)

    def __len__(self):
        return len(self.tile_coords)

    def __getitem__(self, idx):
        r, c = self.tile_coords[idx]
        lcz_patch = self.lcz_map[r:r+self.patch_size, c:c+self.patch_size]
        label_counts = Counter(lcz_patch.flatten())
        most_common_label = label_counts.most_common(1)[0][0]
        return {"label": torch.tensor(most_common_label).item()}

train_ds_stratified = SimpleLCZDataset("../dataset/milan/LCZ_MAP.tif", patch_size, train_tile_coords_stratified)
train_labels_stratified = [train_ds_stratified[i]['label'] for i in range(len(train_ds_stratified))]
unique_classes_train_stratified, class_counts_train_stratified = np.unique(train_labels_stratified, return_counts=True)
class_percentages_train_stratified = (class_counts_train_stratified / len(train_labels_stratified)) * 100

print("\n--- Approximate Class Distribution in Stratified Training Tiles ---")
for cls, count in zip(unique_classes_train_stratified, class_counts_train_stratified):
    print(f"Class {cls}: {count} occurrences ({class_percentages_train_stratified[np.where(unique_classes_train_stratified == cls)[0][0]]:.2f}%)")

# --- Simulate Geographic Stratification ---
def create_geo_stratified_split_coords(lcz_map, patch_size, stride, train_ratio=0.7, n_regions_row=2, n_regions_col=2, seed=42):
    unique_classes = np.unique(lcz_map)
    train_tile_coords_geo = []
    rng = np.random.RandomState(seed)
    H, W = lcz_map.shape
    region_h = H // n_regions_row
    region_w = W // n_regions_col

    for r_reg in range(n_regions_row):
        for c_reg in range(n_regions_col):
            r_start = r_reg * region_h
            r_end = (r_reg + 1) * region_h
            c_start = c_reg * region_w
            c_end = (c_reg + 1) * region_w

            region_coords_by_class = {cls: [] for cls in unique_classes}
            for r in range(r_start, r_end - patch_size + 1, stride):
                for c in range(c_start, c_end - patch_size + 1, stride):
                    top_left_class = lcz_map[r, c]
                    region_coords_by_class[top_left_class].append((r, c))

            for cls in unique_classes:
                coords = region_coords_by_class[cls]
                rng.shuffle(coords)
                n_samples = len(coords)
                train_split = int(train_ratio * n_samples)
                train_tile_coords_geo.extend(coords[:train_split])

    return train_tile_coords_geo

train_tile_coords_geo_stratified = create_geo_stratified_split_coords(lcz_map_full, patch_size, stride)
train_ds_geo_stratified = SimpleLCZDataset("../dataset/milan/LCZ_MAP.tif", patch_size, train_tile_coords_geo_stratified)
train_labels_geo_stratified = [train_ds_geo_stratified[i]['label'] for i in range(len(train_ds_geo_stratified))]
unique_classes_train_geo_stratified, class_counts_train_geo_stratified = np.unique(train_labels_geo_stratified, return_counts=True)
class_percentages_train_geo_stratified = (class_counts_train_geo_stratified / len(train_labels_geo_stratified)) * 100

print("\n--- Approximate Class Distribution in Geographically Stratified Training Tiles ---")
for cls, count in zip(unique_classes_train_geo_stratified, class_counts_train_geo_stratified):
    print(f"Class {cls}: {count} occurrences ({class_percentages_train_geo_stratified[np.where(unique_classes_train_geo_stratified == cls)[0][0]]:.2f}%)")

# --- Undersampling with Target Level ---
def undersample_classes_target(labels, target_samples_per_class):
    class_counts = Counter(labels)
    undersampled_labels = []
    indices_by_class = {cls: [i for i, l in enumerate(labels) if l == cls] for cls in class_counts}
    for cls, indices in indices_by_class.items():
        n_samples = len(indices)
        samples_to_take = min(n_samples, target_samples_per_class)
        if samples_to_take > 0:
            undersampled_indices = random.sample(indices, samples_to_take)
            undersampled_labels.extend([labels[i] for i in undersampled_indices])
    return undersampled_labels

median_count = np.median(class_counts_train_stratified).astype(int) if len(class_counts_train_stratified) > 0 else 30
train_labels_undersampled_target = undersample_classes_target(train_labels_stratified, median_count)
unique_classes_train_undersampled_target, class_counts_train_undersampled_target = np.unique(train_labels_undersampled_target, return_counts=True)
class_percentages_train_undersampled_target = (class_counts_train_undersampled_target / len(train_labels_undersampled_target)) * 100

print(f"\n--- Class Distribution After Undersampling to {median_count} samples per class (max) ---")
for cls, count in zip(unique_classes_train_undersampled_target, class_counts_train_undersampled_target):
    print(f"Class {cls}: {count} occurrences ({class_percentages_train_undersampled_target[np.where(unique_classes_train_undersampled_target == cls)[0][0]]:.2f}%)")

# --- Conceptual Spatially-Aware Oversampling ---
oversampling_target = median_count * 2 # Just an example target
class_counts_oversampled_spatial = {}
oversampled_labels_spatial = []
class_counts_stratified_dict = dict(zip(unique_classes_train_stratified, class_counts_train_stratified))

for cls in unique_classes_train_stratified:
    count = class_counts_stratified_dict.get(cls, 0)
    oversample_amount = max(0, oversampling_target - count)
    # In a real scenario, we'd generate spatially plausible synthetic samples
    # Here, we just conceptually increase the count
    oversampled_count = count + oversample_amount
    oversampled_labels_spatial.extend([cls] * oversampled_count)

unique_classes_train_oversampled_spatial, class_counts_train_oversampled_spatial = np.unique(oversampled_labels_spatial, return_counts=True)
class_percentages_train_oversampled_spatial = (class_counts_train_oversampled_spatial / len(oversampled_labels_spatial)) * 100

print(f"\n--- Conceptual Class Distribution After Spatially-Aware Oversampling (Target: {oversampling_target}) ---")
for cls, count in zip(unique_classes_train_oversampled_spatial, class_counts_train_oversampled_spatial):
    print(f"Class {cls}: {count} occurrences ({class_percentages_train_oversampled_spatial[np.where(unique_classes_train_oversampled_spatial == cls)[0][0]]:.2f}%)")

# --- Cluster-Based Splitting with Stratification (Conceptual) ---
def create_cluster_stratified_split_coords(lcz_map, patch_size, stride, train_ratio=0.7, n_clusters=5, seed=42):
    H, W = lcz_map.shape
    tile_coords = [(r, c) for r in range(0, H - patch_size + 1, stride) for c in range(0, W - patch_size + 1, stride)]
    rng = np.random.RandomState(seed)
    rng.shuffle(tile_coords)
    coords_array = np.array(tile_coords)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init='auto')
    cluster_labels = kmeans.fit_predict(coords_array)

    train_clusters = rng.choice(np.unique(cluster_labels), size=int(train_ratio * n_clusters), replace=False)
    train_tile_coords_cluster = [coords for i, coords in enumerate(tile_coords) if cluster_labels[i] in train_clusters]
    return train_tile_coords_cluster

train_tile_coords_cluster_stratified = create_cluster_stratified_split_coords(lcz_map_full, patch_size, stride)
train_ds_cluster_stratified = SimpleLCZDataset("../dataset/milan/LCZ_MAP.tif", patch_size, train_tile_coords_cluster_stratified)
train_labels_cluster_stratified = [train_ds_cluster_stratified[i]['label'] for i in range(len(train_ds_cluster_stratified))]
unique_classes_train_cluster_stratified, class_counts_train_cluster_stratified = np.unique(train_labels_cluster_stratified, return_counts=True)
class_percentages_train_cluster_stratified = (class_counts_train_cluster_stratified / len(train_labels_cluster_stratified)) * 100

print("\n--- Approximate Class Distribution in Cluster-Based Stratified Training Tiles ---")
for cls, count in zip(unique_classes_train_cluster_stratified, class_counts_train_cluster_stratified):
    print(f"Class {cls}: {count} occurrences ({class_percentages_train_cluster_stratified[np.where(unique_classes_train_cluster_stratified == cls)[0][0]]:.2f}%)")

# --- Conceptual Synthetic Data Generation with Spatial Awareness ---
# This is highly conceptual. A real implementation would involve generating new image patches.
synthetic_oversampling_target = median_count * 3
class_counts_synthetic_spatial = {}
synthetic_labels_spatial = []
for cls in unique_classes_train_stratified:
    count = class_counts_stratified_dict.get(cls, 0)
    oversample_amount = max(0, synthetic_oversampling_target - count)
    synthetic_counts = count + oversample_amount
    synthetic_labels_spatial.extend([cls] * synthetic_counts)

unique_classes_train_synthetic_spatial, class_counts_train_synthetic_spatial = np.unique(synthetic_labels_spatial, return_counts=True)
class_percentages_train_synthetic_spatial = (class_counts_train_synthetic_spatial / len(synthetic_labels_spatial)) * 100

print(f"\n--- Conceptual Synthetic Data Generation with Spatial Awareness (Target: {synthetic_oversampling_target}) ---")
for cls, count in zip(unique_classes_train_synthetic_spatial, class_counts_train_synthetic_spatial):
    print(f"Class {cls}: {count} occurrences ({class_percentages_train_synthetic_spatial[np.where(unique_classes_train_synthetic_spatial == cls)[0][0]]:.2f}%)")

# --- Plotting the Comparison Bar Chart ---
def plot_class_distribution_comparison(original_percentages, stratified_percentages, geo_stratified_percentages, undersampled_percentages, oversampled_percentages, cluster_stratified_percentages, synthetic_spatial_percentages, class_names):
    n_classes = len(class_names)
    bar_width = 0.1
    index = np.arange(n_classes)
    plt.figure(figsize=(24, 12))
    plt.bar(index - 3 * bar_width, original_percentages, bar_width, label='Original', color='b')
    plt.bar(index - 2 * bar_width, stratified_percentages, bar_width, label='Stratified', color='g')
    plt.bar(index - bar_width, geo_stratified_percentages, bar_width, label='Geo-Stratified (Sim.)', color='c')
    plt.bar(index, undersampled_percentages, bar_width, label=f'Undersampled (Target: {median_count})', color='purple')
    plt.bar(index + bar_width, oversampled_percentages, bar_width, label=f'Oversampled (Conceptual, Target: {median_count * 2})', color='orange')
    plt.bar(index + 2 * bar_width, cluster_stratified_percentages, bar_width, label='Cluster-Stratified (Sim.)', color='lime')
    plt.bar(index + 3 * bar_width, synthetic_spatial_percentages, bar_width, label=f'Synthetic Spatial (Conceptual, Target: {median_count * 3})', color='red')
    plt.xlabel('LCZ Class')
    plt.ylabel('Percentage')
    plt.title('Comparison of Class Distributions with Advanced Techniques (Conceptual) Milan Dataset')
    plt.xticks(index, class_names)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Prepare data for plotting
all_unique_classes = sorted(list(set(unique_classes_full) | set(unique_classes_train_stratified) | set(unique_classes_train_undersampled_target) | set(unique_classes_train_geo_stratified) | set(unique_classes_train_oversampled_spatial) | set(unique_classes_train_cluster_stratified) | set(unique_classes_train_synthetic_spatial)))
num_all_classes = len(all_unique_classes)
original_padded = np.zeros(num_all_classes)
stratified_padded = np.zeros(num_all_classes)
geo_stratified_padded = np.zeros(num_all_classes)
undersampled_padded = np.zeros(num_all_classes)
oversampled_padded = np.zeros(num_all_classes)
cluster_stratified_padded = np.zeros(num_all_classes)
synthetic_spatial_padded = np.zeros(num_all_classes)

for i, cls in enumerate(unique_classes_full):
    index = np.where(np.array(all_unique_classes) == cls)[0][0]
    original_padded[index] = class_percentages_full[i]

for i, cls in enumerate(unique_classes_train_stratified):
    index = np.where(np.array(all_unique_classes) == cls)[0][0]
    stratified_padded[index] = class_percentages_train_stratified[i]

for i, cls in enumerate(unique_classes_train_geo_stratified):
    index = np.where(np.array(all_unique_classes) == cls)[0][0]
    geo_stratified_padded[index] = class_percentages_train_geo_stratified[i]

for i, cls in enumerate(unique_classes_train_undersampled_target):
    index = np.where(np.array(all_unique_classes) == cls)[0][0]
    undersampled_padded[index] = class_percentages_train_undersampled_target[i]

for i, cls in enumerate(unique_classes_train_oversampled_spatial):
    index = np.where(np.array(all_unique_classes) == cls)[0][0]
    oversampled_padded[index] = class_percentages_train_oversampled_spatial[i]

for i, cls in enumerate(unique_classes_train_cluster_stratified):
    index = np.where(np.array(all_unique_classes) == cls)[0][0]
    cluster_stratified_padded[index] = class_percentages_train_cluster_stratified[i]

for i, cls in enumerate(unique_classes_train_synthetic_spatial):
    index = np.where(np.array(all_unique_classes) == cls)[0][0]
    synthetic_spatial_padded[index] = class_percentages_train_synthetic_spatial[i]

class_names = [str(cls) for cls in all_unique_classes]

# Plot the comparison
plot_class_distribution_comparison(original_padded, stratified_padded, geo_stratified_padded, undersampled_padded, oversampled_padded, cluster_stratified_padded, synthetic_spatial_padded, class_names)