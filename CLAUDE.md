# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NAVSIM v2 is a Python-based autonomous driving simulation and benchmarking framework that introduces **Pseudo-Simulation** - a novel evaluation methodology combining the efficiency of open-loop evaluation with the robustness of closed-loop simulation. Built on PyTorch and PyTorch Lightning, it serves as the official evaluation framework for the AGC2025 NAVSIM End-to-End Driving Challenge.

## Environment Setup

### Required Environment Variables

Set these in `~/.bashrc` (paths may vary based on your setup):

```bash
export NAVSIM_DEVKIT_ROOT="$HOME/navsim_workspace/navsim"
export NAVSIM_EXP_ROOT="$HOME/navsim_workspace/exp"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$HOME/navsim_workspace/dataset/maps"
export OPENSCENE_DATA_ROOT="$HOME/navsim_workspace/dataset"
```

### Installation

```bash
conda env create --name navsim -f environment.yml
conda activate navsim
pip install -e .
```

## Key Commands

### Training Agents

Training uses PyTorch Lightning via Hydra configuration. Main entry point: `navsim/planning/script/run_training.py`

```bash
# Example: Train TransFuser agent
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
    agent=transfuser_agent \
    dataloader.params.batch_size=64 \
    experiment_name=training_law_agent \
    train_test_split=navtrain \
    use_cache_without_dataset=True \
    force_cache_computation=False \
    cache_path=$NAVSIM_EXP_ROOT/cache_for_training
```

See `scripts/training/` for more training scripts.

### Evaluation

Evaluation uses the Extended PDM Score (EPDMS) with two-stage pseudo closed-loop simulation. Main entry point: `navsim/planning/script/run_pdm_score.py`

```bash
# Example: Evaluate TransFuser agent
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
    train_test_split=navhard_two_stage \
    agent=transfuser_agent \
    worker=single_machine_thread_pool \
    agent.checkpoint_path=/path/to/checkpoint.ckpt \
    experiment_name=transfuser_agent \
    metric_cache_path=$CACHE_PATH \
    synthetic_sensor_path=$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs \
    synthetic_scenes_path=$OPENSCENE_DATA_ROOT/navhard_two_stage/synthetic_scene_pickles
```

### Metric Caching

Pre-compute features and targets for efficient training:

```bash
# Single node
python navsim/planning/script/run_dataset_caching.py

# Multi-node (faster)
python navsim/planning/script/run_dataset_caching_multi_node.py
```

See `scripts/run_dataset_caching.sh` and `scripts/run_dataset_caching_multi_node.sh`

## Architecture

### Core Components

**Agent Interface** (`navsim/agents/abstract_agent.py`)
- Base class for all agents (rule-based and learning-based)
- Key methods: `compute_trajectory()`, `get_sensor_config()`, `initialize()`
- Learning-based agents also implement: `forward()`, `compute_loss()`, `get_optimizers()`, `get_feature_builders()`, `get_target_builders()`

**Sensor System** (`navsim/common/dataclasses.py`)
- 9 modalities: 8 cameras + merged LiDAR point cloud (5 LiDARs)
- 2 seconds history at 2Hz (4 frames)
- Configure via `SensorConfig` - loading sensors impacts runtime

**Evaluation Framework** (`navsim/evaluate/pdm_score.py`)
- Extended PDM Score (EPDMS) with 9 subscores
- Two-stage pseudo closed-loop simulation
- Reactive traffic agent policies
- Multiplier metrics (NC, DAC, DDC, TLC) and weighted metrics (EP, TTC, LK, HC, EC)

**Training Pipeline** (`navsim/planning/training/`)
- PyTorch Lightning-based
- Feature builders extract features from `AgentInput`
- Target builders extract labels from `Scene` (ground truth)
- Metric caching for efficient iteration
- Distributed training support

**Simulation Engine** (`navsim/planning/simulation/`)
- PDM (Predictive Driver Model) simulator with LQR controller
- Traffic agent simulation
- Two-stage scenario generation for pseudo closed-loop evaluation

### Baseline Agents

Located in `navsim/agents/`:

- **ConstantVelocityAgent** (`constant_velocity_agent.py`) - Naive baseline, straight-line trajectory
- **EgoStatusMLPAgent** (`ego_status_mlp_agent.py`) - Blind baseline using only ego state (velocity, acceleration, driving command)
- **TransfuserAgent** (`transfuser/`) - Camera + LiDAR fusion with transformer architecture
- **HumanAgent** (`human_agent.py`) - Privileged agent using ground-truth future

### Dataset Splits

Understanding splits is critical (see `docs/splits.md`):

- **trainval** - Full training/validation data
- **mini** - Small subset for testing
- **navtrain** - Filtered training subset (smaller than trainval)
- **test** / **navtest** - v1 test split
- **navhard_two_stage** - v2 local evaluation with two-stage simulation
- **warmup_two_stage** - Smaller dataset for HuggingFace warmup leaderboard validation
- **private_test_hard_two_stage** - Official challenge data

**IMPORTANT**: Using `test`/`navtest`/`navhard_two_stage`/`warmup_two_stage`/`private_test_two_stage` for training challenge submissions is NOT allowed.

## Configuration System

The project uses **Hydra** for configuration management. Config files are in `navsim/planning/script/config/`:

- `agent/` - Agent configurations
- `common/` - Common settings (dataset paths, dataloader, etc.)
- `dataloader/` - Data loading configurations

Override configs via command line: `python script.py param=value`

## Code Organization Tips

1. **Creating a new agent**: Inherit from `AbstractAgent`, implement required methods. Add config in `navsim/planning/script/config/agent/`

2. **Sensor selection**: Override `get_sensor_config()` to return `SensorConfig`. Unused sensors should be set to `False` for performance

3. **Trajectory output**: Must return `Trajectory` object with `x, y, heading` in local coordinates. Evaluation horizon is 4 seconds at 10Hz

4. **Feature/target builders**: One builder can return multiple tensors. Features use `AgentInput` (no ground truth), targets use `Scene` (has ground truth)

5. **Training workflow**: Cache features/targets first → Train using cache → Evaluate on test split

## Important File Locations

- Entry points: `navsim/planning/script/`
- Scripts: `scripts/training/`, `scripts/evaluation/`, `scripts/submission/`
- Agent implementations: `navsim/agents/`
- Data structures: `navsim/common/dataclasses.py`
- Evaluation logic: `navsim/evaluate/pdm_score.py`
- Documentation: `docs/`
