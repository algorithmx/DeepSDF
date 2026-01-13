# REVIEW.md

This file provides guidance to working with code in this repository.

## Project Overview

DeepSDF is an implementation of the CVPR 2019 paper "DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation" by Park et al. It learns to represent 3D shapes as continuous signed distance functions using neural networks, enabling shape representation, reconstruction, and analysis in a compact latent space.

## Build Requirements

### C++ Dependencies (for preprocessing)
The preprocessing code requires C++ dependencies:
- **CLI11**: Command line interface library
- **Pangolin**: OpenGL toolkit for 3D visualization
- **nanoflann**: Fast k-nearest neighbor library
- **Eigen3**: Linear algebra library

Build with:
```bash
mkdir build && cd build
cmake ..
make -j
```

This creates executables in `bin/` used by `preprocess_data.py`.

### Headless Rendering
Preprocessing opens OpenGL windows via Pangolin. To avoid this, set:
```bash
export PANGOLIN_WINDOW_URI=headless://
```

## Common Commands

### Complete Workflow Example (Sofas)
```bash
# Preprocess training SDF samples
python preprocess_data.py --data_dir data --source /path/ShapeNetCore.v2/ --name ShapeNetV2 --split examples/splits/sv2_sofas_train.json --skip

# Train model
python train_deep_sdf.py -e examples/sofas

# Continue from checkpoint
python train_deep_sdf.py -e examples/sofas --continue 1000

# Reconstruct meshes from test set
python reconstruct.py -e examples/sofas -c 2000 --split examples/splits/sv2_sofas_test.json -d data --skip

# Preprocess surface samples for evaluation
python preprocess_data.py --data_dir data --source /path/ShapeNetCore.v2/ --name ShapeNetV2 --split examples/splits/sv2_sofas_test.json --surface --skip

# Evaluate reconstructions
python evaluate.py -e examples/sofas -c 2000 -d data -s examples/splits/sv2_sofas_test.json

# Visualize training progress
python plot_log.py -e examples/sofas
```

## Architecture

### Neural Network
- **Decoder**: MLP in `networks/deep_sdf_decoder.py` mapping latent codes + 3D coordinates to SDF values
- **Latent Codes**: 256-dimensional vectors per shape, stored in `LatentCodes/` directories
- **Architecture specs**: Defined in `specs.json` (layers, dimensions, dropout, normalization)

### Data Format
- **SDF Samples**: `.npz` files containing point coordinates + signed distances
- **Surface Samples**: `.ply` files for evaluation (point clouds from mesh surfaces)
- **Split Files**: JSON files in `examples/splits/` defining train/test subsets

### Experiment Directory Structure
```
<experiment_name>/
    specs.json                  # Training configuration
    Logs.pth                    # Training logs
    LatentCodes/<Epoch>.pth     # Shape embeddings
    ModelParameters/<Epoch>.pth # Decoder weights
    OptimizerParameters/<Epoch>.pth # Optimizer state
    Reconstructions/<Epoch>/    # Generated meshes
    Evaluations/                # Metrics (Chamfer, EMD)
```

### Unified Data Source Structure
```
<data_source_name>/
    .datasources.json           # Dataset path mappings
    SdfSamples/
        <dataset_name>/<class_name>/<instance_name>.npz
    SurfaceSamples/
        <dataset_name>/<class_name>/<instance_name>.ply
```

## Key Python Modules

- **`deep_sdf/workspace.py`**: Manages experiment directories, checkpointing, and I/O
- **`deep_sdf/data.py`**: Data loading and batching
- **`deep_sdf/mesh.py`**: Mesh processing utilities
- **`deep_sdf/metrics/chamfer.py`**: Chamfer distance for evaluation
- **`networks/deep_sdf_decoder.py`**: Neural network architecture

## Training Configuration

Training parameters are in `specs.json`:
- `NetworkSpecs`: Layer dimensions, dropout, normalization, latent injection point
- `CodeLength`: Latent vector dimension (default 256)
- `LearningRateSchedule`: Step-based decay for decoder and latent codes
- `SamplesPerScene`: SDF samples per shape (16384)
- `ScenesPerBatch`: Batch size (64 shapes)
- `CodeRegularization`: L2 regularization on latent codes
- `ClampingDistance`: Truncation distance for SDF values

## Important Notes

- **Stochastic Reconstruction**: Mesh reconstruction uses gradient descent with random initialization, so multiple runs may produce different results
- **Missing Shape Completion**: The current release does not include shape completion functionality
- **Evaluation Baselines**: Paper results used multiple reconstructions per shape (best selected by Chamfer distance); released code does not support this, so results may differ
- **Checkpoint Consistency**: All three components must exist to continue training: `ModelParameters`, `OptimizerParameters`, and `LatentCodes`
