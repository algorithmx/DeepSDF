# Training Data Generation Guide for DeepSDF

Complete guide for generating training data for DeepSDF from 3D mesh files (ShapeNet or custom datasets).

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Overview](#overview)
3. [Prerequisites](#prerequisites)
4. [Data Structure](#data-structure)
5. [Split Files](#split-files)
6. [Preprocessing Commands](#preprocessing-commands)
7. [Output File Formats](#output-file-formats)
8. [Preprocessing Parameters](#preprocessing-parameters)
9. [Training Configuration](#training-configuration)
10. [Complete Pipeline Example](#complete-pipeline-example)
11. [Performance Expectations](#performance-expectations)
12. [Troubleshooting](#troubleshooting)
13. [Validation](#validation)

---

## Quick Start

```bash
# 1. Activate conda environment
conda activate ml_env

# 2. Install dependencies
pip install trimesh scipy numpy plyfile

# 3. Preprocess training data
python preprocess_data_python.py \
    --data_dir data \
    --source ShapeNetCore.v2 \
    --name ShapeNetV2 \
    --split examples/splits/lamps_train_custom.json \
  --sign-method vote \
    --threads 8

# 4. Preprocess test data
python preprocess_data_python.py \
    --data_dir data \
    --source ShapeNetCore.v2 \
    --name ShapeNetV2 \
    --split examples/splits/lamps_test_custom.json \
    --test \
  --sign-method vote \
    --threads 8

# 5. Preprocess surface samples for evaluation
python preprocess_data_python.py \
    --data_dir data \
    --source ShapeNetCore.v2 \
    --name ShapeNetV2 \
    --split examples/splits/lamps_test_custom.json \
    --surface \
    --threads 8
```

---

## Overview

DeepSDF requires preprocessed SDF (Signed Distance Function) samples from mesh files. The preprocessing pipeline converts 3D mesh files (`.obj`, `.ply`, etc.) into SDF samples stored in NPZ format.

**Key Points:**
- **No C++/Pangolin needed** - Python preprocessing uses `trimesh` library
- **Three preprocessing modes**:
  - Training SDF samples
  - Test SDF samples (with different sampling parameters)
  - Surface samples (for evaluation metrics)
- **Data is organized by**: dataset → class → instance ID

---

## Prerequisites

### Required Dependencies

```bash
conda activate ml_env
pip install trimesh scipy numpy plyfile
```

Optional accelerators:
- `igl` (libigl Python bindings) enables `--sign-method igl`

**Package purposes:**
- `trimesh`: Mesh loading and geometry processing
- `scipy`: Fast KD-tree queries (optional but recommended)
- `numpy`: Array operations
- `plyfile`: PLY format support

### Verify Installation

```bash
python -c "import trimesh; import scipy; import numpy; print('✅ All dependencies installed')"
```

---

## Data Structure

### Input (Raw Meshes)

Your source data should follow this structure:

```
<source_directory>/                    # e.g., ShapeNetCore.v2
├── <class_id>/                        # e.g., 03636649 (lamps)
│   ├── <instance_hash>/               # e.g., 101d0e7dbd07d8247dfd6bf7196ba84d
│   │   └── models/
│   │       └── model_normalized.ply   # Mesh file (auto-detected)
│   ├── <instance_hash>/
│   │   └── models/
│   │       └── model.obj
│   └── ...
└── <class_id>/
    └── ...
```

**Supported mesh formats:** `.obj`, `.ply`, `.stl`, `.off`

The mesh file can be named anything - it's auto-detected by the preprocessing script.

### Output (Preprocessed Data)

After preprocessing, the data directory will contain:

```
<data_root>/
├── .datasources.json                  # Maps dataset names to source paths
├── SdfSamples/                        # Training SDF samples
│   └── <dataset_name>/                # e.g., ShapeNetV2
│       └── <class_id>/                # e.g., 03636649
│           ├── <instance_hash>.npz    # SDF samples
│           └── ...
├── NormalizationParameters/           # Mesh normalization data
│   └── <dataset_name>/
│       └── <class_id>/
│           ├── <instance_hash>.npz
│           └── ...
└── SurfaceSamples/                    # Surface point clouds (evaluation)
    └── <dataset_name>/
        └── <class_id>/
            ├── <instance_hash>.ply
            └── ...
```

---

## Split Files

Split files define which instances belong to training/testing sets. Format: JSON

### Format Structure

```json
{
  "<dataset_name>": {
    "<class_id>": [
      "<instance_hash_1>",
      "<instance_hash_2>",
      ...
    ],
    "<class_id_2>": [...]
  }
}
```

### Example

```json
{
  "ShapeNetV2": {
    "03636649": [
      "101d0e7dbd07d8247dfd6bf7196ba84d",
      "102273fdf8d1b90041fbc1e2da054acb",
      "107b8c870eade2481735ea0e092a805a"
    ]
  }
}
```

### Creating Custom Split Files

If your ShapeNet instance hashes differ from provided splits:

```bash
# List all instances in a class (first 200 for training)
ls ShapeNetCore.v2/03636649/ | head -200 > /tmp/lamps_train.txt

# Create remaining instances for test
ls ShapeNetCore.v2/03636649/ | tail -n +201 > /tmp/lamps_test.txt

# Convert to JSON format manually or with script
```

### Provided Split Files

Located in `examples/splits/`:
- `sv2_lamps_train.json` - 375 lamp instances (original hashes)
- `sv2_lamps_test.json` - 94 lamp instances
- `sv2_chairs_train.json` / `sv2_chairs_test.json`
- `sv2_tables_train.json` / `sv2_tables_test.json`
- And more for other ShapeNet categories

---

## Preprocessing Commands

### Command-Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--data_dir` | Yes | - | Root directory for preprocessed output |
| `--source` | Yes | - | Path to raw mesh source directory |
| `--name` | Yes | - | Dataset name (used in directory structure) |
| `--split` | Yes | - | JSON file defining shapes to process |
| `--threads` | No | 8 | Number of parallel processing threads |
| `--sign-method` | No | `vote` | SDF computation method: `vote`, `proximity`, or `igl` |
| `--skip` | No | False | Deprecated (existing outputs are skipped automatically) |
| `--test` | No | False | Use test sampling parameters |
| `--surface` | No | False | Generate surface samples instead of SDF |
| `--debug` | No | False | Verbose logging (otherwise only global progress is shown) |

### 1. Training SDF Samples

Generates signed distance function samples for training:

```bash
python preprocess_data_python.py \
    --data_dir data \
    --source ShapeNetCore.v2 \
    --name ShapeNetV2 \
    --split examples/splits/lamps_train_custom.json \
  --sign-method vote \
    --threads 8
```

**Output:** NPZ files with `pos` and `neg` arrays containing SDF samples

### 2. Test SDF Samples

Uses different sampling parameters for test data:

```bash
python preprocess_data_python.py \
    --data_dir data \
    --source ShapeNetCore.v2 \
    --name ShapeNetV2 \
    --split examples/splits/lamps_test_custom.json \
    --test \
  --sign-method vote \
    --threads 8
```

**Difference from training:** Larger variance, different near-surface sampling ratio

### 3. Surface Samples (Evaluation)

Generates surface point clouds for evaluation metrics:

```bash
python preprocess_data_python.py \
    --data_dir data \
    --source ShapeNetCore.v2 \
    --name ShapeNetV2 \
    --split examples/splits/lamps_test_custom.json \
    --surface \
    --threads 8
```

**Output:** PLY files with vertex positions and normals

### Resume with Skip Flag

Re-running the script will automatically skip any shape whose target output file already exists.
The `--skip` flag is kept for compatibility but is no longer required.

```bash
python preprocess_data_python.py \
    --data_dir data \
    --source ShapeNetCore.v2 \
    --name ShapeNetV2 \
    --split examples/splits/lamps_train_custom.json \
    --threads 8 \
    --skip
```

### Logging Levels

- Default: shows only global progress (samples + meshes) plus warnings/errors
- `--debug`: shows per-mesh logs (loading, normalization, sampled counts, etc.)

---

## Output File Formats

### SDF Samples (`.npz`)

```python
import numpy as np
data = np.load("data/SdfSamples/ShapeNetV2/03636649/instance.npz")

# Positive SDF samples (inside the shape)
pos = data["pos"]  # Shape: (N, 4) - [x, y, z, sdf_value]

# Negative SDF samples (outside the shape)
neg = data["neg"]  # Shape: (M, 4) - [x, y, z, sdf_value]
```

**Components:**
- **x, y, z**: 3D coordinates in normalized unit cube space
- **sdf_value**: Signed distance from surface
  - Positive = inside the shape
  - Negative = outside the shape
- **Typical sizes**: ~250,000 positive + ~250,000 negative samples (total ~500,000)

### Normalization Parameters (`.npz`)

```python
params = np.load("data/NormalizationParameters/ShapeNetV2/03636649/instance.npz")

# Translation vector to center the mesh
offset = params["offset"]  # Shape: (3,)

# Scale factor to normalize to unit cube
scale = params["scale"]    # Scalar
```

**Purpose:** These parameters allow reconstructing the original mesh from normalized coordinates.

### Surface Samples (`.ply`)

PLY format with vertex positions and normals:

```
ply
format ascii 1.0
element vertex N
property float x
property float y
property float z
property float nx
property float ny
property float nz
end_header
x1 y1 z1 nx1 ny1 nz1
x2 y2 z2 nx2 ny2 nz2
...
```

**Purpose:** Used for computing evaluation metrics (Chamfer distance, etc.)

---

## Preprocessing Parameters

### SDF Sampling (Training Data)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_sample` | 500,000 | Total number of SDF samples per mesh |
| `num_samp_near_surf` | 47/50 of total | Samples near surface |
| `variance` | 0.005 | Gaussian σ for perturbation |
| `second_variance` | variance/10 | Secondary σ for more spread |
| `num_votes` | 11 | Nearest neighbors for sign voting |
| `bounding_cube_dim` | 2.0 | Size of sampling volume |

**To modify:** Edit `preprocess_mesh_python.py` (lines with default parameter values)

### Test Parameters

When using `--test` flag:
- `variance`: 0.05 (10x larger than training)
- `second_variance`: variance/100 (more conservative)
- `num_samp_near_surf`: 45/50 of total

### Surface Sampling (Evaluation Data)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_samples` | 100,000 | Target number of surface samples |
| `num_views` | 100 | Camera viewpoints for visibility |
| `buffer` | 1.03 | Buffer factor for normalization (3%) |

---

## Training Configuration

### Experiment Specification (`specs.json`)

Create a `specs.json` file in your experiment directory:

```json
{
  "Description": ["DeepSDF training on ShapeNet lamps"],
  "DataSource": "data",
  "TrainSplit": "examples/splits/lamps_train_custom.json",
  "TestSplit": "examples/splits/lamps_test_custom.json",
  "NetworkArch": "deep_sdf_decoder",
  "NetworkSpecs": {
    "dims": [512, 512, 512, 512, 512, 512, 512, 512],
    "dropout": [0, 1, 2, 3, 4, 5, 6, 7],
    "dropout_prob": 0.2,
    "norm_layers": [0, 1, 2, 3, 4, 5, 6, 7],
    "latent_in": [4],
    "xyz_in_all": false,
    "use_tanh": false,
    "latent_dropout": false,
    "weight_norm": true
  },
  "CodeLength": 256,
  "NumEpochs": 2001,
  "SnapshotFrequency": 1000,
  "AdditionalSnapshots": [100, 500],
  "LearningRateSchedule": [
    {
      "Type": "Step",
      "Initial": 0.0005,
      "Interval": 500,
      "Factor": 0.5
    }
  ],
  "SamplesPerScene": 16384,
  "ScenesPerBatch": 64,
  "DataLoaderThreads": 16,
  "ClampingDistance": 0.1,
  "CodeRegularization": true,
  "CodeRegularizationLambda": 1e-4,
  "CodeBound": 1.0
}
```

### Key Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DataSource` | `"data"` | Root directory for preprocessed data |
| `TrainSplit` | Required | Path to train split JSON |
| `TestSplit` | Required | Path to test split JSON |
| `CodeLength` | 256 | Dimensionality of latent code |
| `NumEpochs` | 2001 | Total training epochs |
| `SamplesPerScene` | 16384 | SDF samples per shape per batch |
| `ScenesPerBatch` | 64 | Number of shapes in each batch |
| `ClampingDistance` | 0.1 | SDF values clamped to [-0.1, 0.1] |
| `DataLoaderThreads` | 16 | Parallel data loading workers |

---

## Complete Pipeline Example

### Step 1: Setup

```bash
# Activate conda environment
conda activate ml_env

# Install dependencies
pip install trimesh scipy numpy plyfile
```

### Step 2: Prepare Data

Ensure your ShapeNet data follows the correct structure:

```bash
ShapeNetCore.v2/
└── 03636649/                    # Lamps class ID
    ├── 101d0e7dbd07d8247dfd6bf7196ba84d/
    │   └── models/
    │       └── model_normalized.ply
    └── ...
```

### Step 3: Create Split Files

Create `examples/splits/lamps_train_custom.json`:

```json
{
  "ShapeNetV2": {
    "03636649": [
      "101d0e7dbd07d8247dfd6bf7196ba84d",
      "102273fdf8d1b90041fbc1e2da054acb",
      ...
    ]
  }
}
```

### Step 4: Preprocess Training Data

```bash
python preprocess_data_python.py \
    --data_dir data \
    --source ShapeNetCore.v2 \
    --name ShapeNetV2 \
    --split examples/splits/lamps_train_custom.json \
    --threads 8
```

**Expected output:**
```
data/
└── SdfSamples/
    └── ShapeNetV2/
        └── 03636649/
            ├── 101d0e7dbd07d8247dfd6bf7196ba84d.npz
            └── ...
```

### Step 5: Preprocess Test Data

```bash
# SDF samples for test shapes
python preprocess_data_python.py \
    --data_dir data \
    --source ShapeNetCore.v2 \
    --name ShapeNetV2 \
    --split examples/splits/lamps_test_custom.json \
    --test \
    --threads 8

# Surface samples for evaluation
python preprocess_data_python.py \
    --data_dir data \
    --source ShapeNetCore.v2 \
    --name ShapeNetV2 \
    --split examples/splits/lamps_test_custom.json \
    --surface \
    --threads 8
```

### Step 6: Configure Experiment

```bash
# Create experiment directory
mkdir -p experiments/my_lamps_experiment

# Copy and modify spec file
cp examples/lamps/specs.json experiments/my_lamps_experiment/
# Edit specs.json as needed
```

### Step 7: Train Model

```bash
python train_deep_sdf.py -e experiments/my_lamps_experiment
```

### Step 8: Evaluate

```bash
python evaluate.py \
    -e experiments/my_lamps_experiment \
    -c latest \
    -d data \
    -s examples/splits/lamps_test_custom.json
```

---

## Performance Expectations

### Processing Time per Mesh

| Mesh Complexity | Time (1 thread) | Time (8 threads) |
|----------------|-----------------|------------------|
| Simple (~5k faces) | ~5 seconds | ~5 seconds |
| Medium (~10k faces) | ~10 seconds | ~10 seconds |
| Complex (~20k faces) | ~15 seconds | ~15 seconds |

**Note:** With threading, multiple meshes are processed in parallel, so throughput increases nearly linearly with thread count (up to CPU cores).

### Total Processing Time

| Dataset Size | 1 Thread | 8 Threads |
|--------------|----------|------------|
| 50 meshes | ~8 minutes | ~1 minute |
| 100 meshes | ~15 minutes | ~2 minutes |
| 200 meshes | ~30 minutes | ~4 minutes |
| 500 meshes | ~75 minutes | ~10 minutes |

### Factors Affecting Speed

1. **Mesh complexity** (face/vertex count)
2. **Number of samples** (`num_sample` parameter)
3. **CPU cores available** (threads)
4. **Whether scipy is installed** (faster KD-tree)
5. **Disk I/O** (SSD vs HDD)

### Optimizing Performance

```bash
# Install scipy for faster KD-tree queries
pip install scipy

# Use more threads (if you have CPU cores)
--threads 16

# Reduce sample count for faster preprocessing
# (Edit preprocess_mesh_python.py, change num_sample)
```

---

## Troubleshooting

### "No mesh found for instance"

**Cause:** Mesh file format not supported or file missing.

**Solution:**
```bash
# Check file exists
ls <source>/<class_id>/<instance_id>/models/

# Check supported format
# Supported: .obj, .ply, .stl, .off
```

### Import Error: No module named 'trimesh'

```bash
pip install trimesh
```

### Import Error: No module named 'plyfile'

```bash
pip install plyfile
```

### Slow preprocessing

```bash
# Install scipy for faster KD-tree
pip install scipy

# Reduce threads if CPU is overloaded
--threads 4

# Reduce sample count (edit preprocess_mesh_python.py)
# Change: num_sample = 100000  # instead of 500000
```

### Out of memory

```bash
# Reduce number of threads
--threads 2

# Reduce sample count per mesh
# Edit preprocess_mesh_python.py
```

### Path duplication in logs

If you see paths like `ShapeNet/.../ShapeNet/...` in logs:
- This is just a logging artifact
- Files are created correctly if no errors appear
- Can be ignored

### Only some files processed

**Cause:** Some instance hashes in split file don't exist in source directory.

**Solution:**
```bash
# Check which instances exist
ls <source>/<class_id>/

# Update split file to only include existing instances
```

---

## Validation

### Verify Preprocessed Data

```python
import numpy as np
import os

# Check training data format
data_path = "data/SdfSamples/ShapeNetV2/03636649"
for npz_file in os.listdir(data_path):
    if npz_file.endswith('.npz'):
        data = np.load(os.path.join(data_path, npz_file))

        # Check arrays exist
        assert "pos" in data.files, "Missing 'pos' array"
        assert "neg" in data.files, "Missing 'neg' array"

        # Check shape (should be N x 4)
        assert data["pos"].shape[1] == 4, "Wrong 'pos' shape"
        assert data["neg"].shape[1] == 4, "Wrong 'neg' shape"

        print(f"✅ {npz_file}")
        print(f"   Positive samples: {len(data['pos'])}")
        print(f"   Negative samples: {len(data['neg'])}")
        break  # Just check first file

# Check normalization params
norm_path = "data/NormalizationParameters/ShapeNetV2/03636649"
for npz_file in os.listdir(norm_path):
    if npz_file.endswith('.npz'):
        params = np.load(os.path.join(norm_path, npz_file))

        assert "offset" in params.files, "Missing 'offset'"
        assert "scale" in params.files, "Missing 'scale'"

        print(f"✅ {npz_file}")
        print(f"   Offset: {params['offset']}")
        print(f"   Scale: {params['scale']}")
        break
```

### Count Processed Files

```bash
# Training samples
ls data/SdfSamples/ShapeNetV2/03636649/*.npz | wc -l

# Normalization params
ls data/NormalizationParameters/ShapeNetV2/03636649/*.npz | wc -l

# Surface samples
ls data/SurfaceSamples/ShapeNetV2/03636649/*.ply | wc -l
```

### Visualize Samples (Optional)

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load sample data
data = np.load("data/SdfSamples/ShapeNetV2/03636649/instance.npz")
pos = data["pos"]
neg = data["neg"]

# Plot
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(pos[:1000, 0], pos[:1000, 1], pos[:1000, 2], c=pos[:1000, 3], cmap='viridis')
ax1.set_title("Positive SDF Samples")

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(neg[:1000, 0], neg[:1000, 1], neg[:1000, 2], c=neg[:1000, 3], cmap='viridis')
ax2.set_title("Negative SDF Samples")

plt.tight_layout()
plt.savefig("sdf_samples.png")
```

---

## Summary Checklist

### Before Preprocessing
- [ ] Install dependencies: `pip install trimesh scipy numpy plyfile`
- [ ] Prepare raw mesh data in correct directory structure
- [ ] Create or verify split JSON files

### Preprocessing
- [ ] Preprocess training SDF samples
- [ ] Preprocess test SDF samples (with `--test` flag)
- [ ] Preprocess surface samples (with `--surface` flag)
- [ ] Verify output files exist and have correct format

### Training Setup
- [ ] Create experiment directory
- [ ] Configure `specs.json` with appropriate parameters
- [ ] Verify train/test split paths in specs.json

---

## See Also

- `PYTHON_PREPROCESSING.md` - Technical details on Python preprocessing implementation
- `examples/lamps/specs.json` - Example training configuration
- `examples/splits/` - Example train/test split files
- `README.md` - Original DeepSDF documentation
