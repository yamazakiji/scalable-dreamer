# Dreamer v4 PyTorch Implementation

A clean PyTorch reimplementation of "Training Agents Inside of Scalable World Models" (Dreamer v4).

Paper: [arXiv:2509.24527](https://arxiv.org/abs/2509.24527)
Authors: Danijar Hafner, Wilson Yan, Timothy Lillicrap (Google DeepMind)

## Overview

Dreamer v4 is a scalable agent that learns to solve complex control tasks by reinforcement learning inside a fast and accurate world model. Key achievements from the paper:

- First agent to obtain diamonds in Minecraft purely from offline data (no environment interaction)
- Real-time interactive inference on a single GPU via shortcut forcing (K=4 sampling steps)
- Learns action conditioning from small amounts of labeled data, absorbing knowledge from unlabeled videos

## Architecture

The system consists of three main components:

### 1. Causal Tokenizer

Compresses video frames into continuous latent representations using masked autoencoding (MAE).

- Patch embedding with learnable latent tokens
- Tanh bottleneck for representation learning
- MSE + LPIPS reconstruction loss
- Random masking (p ~ U(0, 0.9)) during training

### 2. Interactive Dynamics Model

Predicts future latent representations given actions and context.

- 2D transformer with space + time attention
- Temporal attention every 4 layers (efficiency)
- Shortcut forcing objective for fast inference
- X-prediction parameterization (not v-prediction)
- Register tokens for temporal consistency
- Block-causal masking

### 3. Agent (Policy, Reward, Value Heads)

Learns behaviors through imagination training inside the world model.

- Multi-token prediction for behavior cloning
- PMPO objective for reinforcement learning
- Symexp twohot for reward/value outputs

## Training Phases

Following the paper:

1. **Phase 1: World Model Pretraining** - Train tokenizer on videos, then train dynamics model on tokenized videos + actions
2. **Phase 2: Agent Finetuning** - Add policy and reward heads, train with behavior cloning and reward modeling
3. **Phase 3: Imagination Training** - Optimize policy via RL on imagined rollouts from the world model

## Current Implementation Status

### Implemented

- [x] Configuration system (YAML + CLI overrides)
- [x] Causal Tokenizer with MAE training
- [x] Dynamics Model architecture with shortcut forcing
- [x] Action Encoder (discrete embeddings)
- [x] Transformer blocks (RoPE, SwiGLU, QKNorm, attention soft capping)
- [x] Video Dataset loading (mp4, avi)
- [x] Training infrastructure (DDP, FSDP)
- [x] Tokenizer training script
- [x] WandB logging integration
- [x] LPIPS reconstruction metrics
- [x] Random policy for rollout collection
- [x] Rollout collection script (stable-retro integration)
- [x] Tokenizer training script works

### TODO

- [ ] Train decent tokenizer
- [ ] Dynamics model training script
- [ ] Agent model (policy/reward/value heads)
- [ ] Agent finetuning phase
- [ ] Imagination training / RL
- [ ] Multi-token prediction for behavior cloning
- [ ] PMPO objective implementation
- [ ] Task embeddings for multi-task learning
- [ ] Symexp twohot outputs
- [ ] GQA (Grouped Query Attention) for dynamics
- [ ] Alternating batch length training

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd world_models

# please figure reqs yourself, but they are basic pytorch+wandb+stable retro
# I ship reqs file, but I am unsure that they would suffice

# Install dependencies
pip install -r requirements.txt

# Additional dependencies
pip install einops lpips
```

### Requirements

- Python 3.9+
- PyTorch >= 2.0.0
- CUDA-capable GPU

## Usage

### Collect Environment Rollouts

```bash
# Collect episodes from stable-retro environment I use MK2 but for licensing purpose cant share my hard earned ROM
python collect_rollouts.py --num-episodes 100 --output-dir data/rollouts

# With custom settings
python collect_rollouts.py --seed 42 --fps 60 --max-steps 1000
```

### Train Tokenizer

```bash
# Single GPU
python train_tokenizer.py --name my_experiment

# Multi-GPU with DDP
torchrun --nproc_per_node=2 train_tokenizer.py --name my_experiment

# Multi-GPU with FSDP (for large models)
torchrun --nproc_per_node=4 train_tokenizer.py --name my_experiment --fsdp

# With config file
python train_tokenizer.py --config wm/configs/presets/tokenizer_default.yaml

# Disable wandb logging
python train_tokenizer.py --no-wandb

# Resume from checkpoint
python train_tokenizer.py --resume outputs/my_experiment/checkpoint_step_10000.pt
```

## Project Structure

```
world_models/
├── wm/                              # Main package
│   ├── configs/                     # Configuration system
│   │   └── base.py                  # BaseConfig, TokenizerConfig, TrainingConfig
│   ├── data/                        # Data loading
│   │   └── video_dataset.py         # VideoDataset for mp4/avi files
│   ├── envs/                        # Environment wrappers (TODO)
│   ├── losses/                      # Loss functions (TODO)
│   ├── models/                      # Neural network models
│   │   ├── tokenizer/
│   │   │   └── causal_tokenizer.py  # MAE-based video tokenizer
│   │   ├── dynamics/                # A lot of work will be done here 
│   │   │   ├── dynamics_model.py    # Interactive dynamics with shortcut forcing
│   │   │   ├── action_encoder.py    # Discrete action embeddings
│   │   │   └── shortcut_forcing.py  # Shortcut forcing utilities
│   │   ├── transformer/
│   │   │   └── block.py             # RoPE, FFN, SpaceAttention, TimeAttention, TransformerBlock
│   │   └── agent/                   # Agent components (TODO)
│   ├── policies/                    # Action policies
│   │   ├── base.py                  # Policy base class
│   │   └── random_policy.py         # Random action sampling
│   ├── training/                    # Training utilities
│   │   ├── optimizer.py             # AdamW + warmup scheduler
│   │   ├── distributed.py           # DDP setup
│   │   └── fsdp.py                  # FSDP support
│   └── utils/                       # Utilities
│       ├── logging.py               # WandB logger
│       ├── metrics.py               # LPIPS, MSE, PSNR metrics
│       └── visualization.py         # Reconstruction visualization
├── train_tokenizer.py               # Tokenizer training script
├── collect_rollouts.py              # Environment rollout collection
└── requirements.txt                 # Dependencies
```

## Key Hyperparameters (from paper)

| Component | Parameter | Value |
|-----------|-----------|-------|
| Tokenizer | Parameters | 400M (TBD) |
| Dynamics | Parameters | 1.6B (TBD) |
| Dynamics | Spatial tokens | 256 |
| Dynamics | Context length | 192 frames |
| Dynamics | Sampling steps | K=4 |
| Dynamics | Context noise | 0.1 |
| Training | Batch length | 64 (short) / 256 (long) |
| Training | Temporal attention freq | Every 4 layers |
| Agent | Discount | 0.997 |
| Agent | MTP length | 8 |

## References

- Paper: [Training Agents Inside of Scalable World Models](https://arxiv.org/abs/2509.24527)
- Project Page: [danijar.com/project/dreamer4](https://danijar.com/project/dreamer4)

## Citation

```bibtex
@article{hafner2025dreamer4,
  title={Training Agents Inside of Scalable World Models},
  author={Hafner, Danijar and Yan, Wilson and Lillicrap, Timothy},
  journal={arXiv preprint arXiv:2509.24527},
  year={2025}
}
```

## License

This is an unofficial reimplementation for research (and fun) purposes only, I guess. Kudos to authors of the paper
