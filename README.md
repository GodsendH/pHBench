# Venus-DREAM

A Deep Retrieval-Enhanced Meta-learning Framework for Enzyme Optimum pH Prediction

## Introduction

This is a meta-learning framework for predicting protein pH using MAML and Reptile algorithms.

## Key Features

- **Meta-Learning Implementations**:
  - **MAML** (Model-Agnostic Meta-Learning)
  - **Reptile** 
  
- **Sequence Retrieval Strategies**:
  - **Similarity-based** retrieval using ESM2 embeddings
  - **Random** sequence selection
  - **Fixed random** selection
  - **Scaled retrieval** with different dataset sizes



## Installation

To install the project, follow these steps:

```bash
# Clone the repository
git clone https://github.com/zhangliang-sys/PRO-DREAM.git
cd PRO-DREAM

# Install requirements
pip install -r requirements.txt


```

## Usage

### Data Preparation

Build the homology-controlled PHOPT datasets with MMseqs2. Identity and
coverage accept either fractions or percentages. Exact-sequence groups whose
pHopt range exceeds `0.25` are excluded; the remaining duplicate groups use
their median pHopt label.

```bash
conda activate phbench
python scripts/build_homology_datasets.py \
    --identity 100 50 30 20 \
    --coverage 80 \
    --label-conflict-threshold 0.25
```

The generated datasets are stored under `data/datasets/<dataset>/`:

| Dataset | Train | Validation | Test | Homology clusters |
| --- | ---: | ---: | ---: | ---: |
| `identity100` | 7,046 | 749 | 1,950 | 9,734 |
| `identity50` | 7,031 | 765 | 1,949 | 6,189 |
| `identity30` | 6,936 | 846 | 1,963 | 3,917 |
| `identity20` | 6,853 | 913 | 1,979 | 3,252 |

Each directory includes the split FASTA files, `records.tsv`, `clusters.tsv`,
`duplicate_conflicts.tsv`, and `metadata.json`. The builder uses all-vs-all
MMseqs2 search followed by connected components and verifies train-validation,
train-test, and validation-test separation before publishing a dataset.

Prepare sequence retrieval data using different strategies:

```bash
python retrieval.py \
    --dataset identity20 \
    --model_name facebook/esm2_t33_650M_UR50D \
    --strategy opt_retrieval \
    --topk 5
```

Available dataset names are `phopt`, `identity100`, `identity50`, `identity30`,
and `identity20`. `phopt` preserves the original paths and remains the default.
Generated retrieval JSON files are isolated under
`data/processed/<dataset>/`; explicit `--opt_train`, `--opt_valid`,
`--opt_test`, `--features_dir`, and `--output_dir` still override the defaults.

### Model Training

#### MAML Training

Train the model using Model-Agnostic Meta-Learning:

```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

python maml.py \
    --dataset identity20 \
    --mode train \
    --num_epochs 50 \
    --retrieval_strategy opt_retrieval \
    --topk 5 \
    --pretrained \
    --meta_lr 0.0001 \
    --inner_lr 0.0005 \
    --inner_steps 5 \
    --validate_every 200 \
    --patience 5 \
    --seed 0
```

#### Reptile Training

Train the model on one GPU using the Reptile algorithm:

```bash
python reptile.py \
    --dataset identity20 \
    --mode train \
    --num_epochs 50 \
    --retrieval_strategy opt_retrieval \
    --topk 5 \
    --pretrained \
    --meta_lr 1 \
    --inner_lr 0.001 \
    --inner_steps 5 \
    --meta_batch_size 5 \
    --validate_every 200 \
    --patience 5 \
    --seed 0
```

For distributed Reptile training, activate the same `phbench` environment on
every process and launch one process per GPU. Inner-loop task adaptation is
local to each GPU; one flattened RLAT parameter delta is reduced per global
meta-batch with NCCL.

Single-node example with two GPUs:

```bash
conda activate phbench
export LD_LIBRARY_PATH=/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=4

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    reptile.py \
    --dataset identity20 \
    --mode train \
    --num_epochs 50 \
    --retrieval_strategy opt_retrieval \
    --topk 5 \
    --pretrained \
    --meta_lr 1 \
    --inner_lr 0.001 \
    --inner_steps 5 \
    --meta_batch_size 5 \
    --validate_every 200 \
    --patience 5 \
    --seed 0
```

Multi-node launch uses the usual torchrun rendezvous arguments:

```bash
torchrun \
    --nnodes="$NUM_NODES" \
    --nproc_per_node="$GPUS_PER_NODE" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    reptile.py [training arguments]
```

`--meta_batch_size` is the global number of tasks per Reptile update and must
be at least the total process count. Keeping it at 5 preserves the previous
batch size and is best suited to two GPUs. For four GPUs, a value divisible by
four, such as 8, improves utilization but changes the optimization trajectory.

The distributed implementation uses standard Reptile meta-batching: every
task adapts from the same base parameters, task deltas are averaged globally,
and one outer update is applied. This replaces the old sequential behavior
where each task updated the base model before the next task.

The frozen ESM1v representations are cached lazily as float32 tensors in
`data/features/esm1v_t33_650M_UR90S_1`. The first encounter of a sequence
creates its cache file; later inner-loop steps, validation runs, and epochs
reuse it. Use `--embedding_cache_dir` to choose another location,
`--embedding_memory_cache_size` to control the in-memory LRU size, or
`--disable_embedding_cache` to run the original uncached forward path.

Run names include `inner_steps`, `support_batch_size`, and `seed` so that
experiments with different adaptation settings do not overwrite each other.
Distributed Reptile run names also include global meta-batch size and world
size. Use `--checkpoint_path` to test a checkpoint with a different number of
GPUs from the training run.


### Command Line Arguments

- **Mode Options**:
  - `--mode`: train or test
  - `--pretrained`: Use pretrained EpHod RLAT weights; ESM1v is always pretrained
  - `--dataset`: `phopt`, `identity100`, `identity50`, `identity30`, or `identity20`

- **Retrieval Strategy**:
  - `--retrieval_strategy`:
    - `opt_retrieval`: Similarity-based
    - `opt_random`: Random selection
    - `opt_fixed_random`: Fixed random selection
    - `opt_retrieval_scaled_0.2`: 20% scaled retrieval
    - `opt_retrieval_scaled_0.6`: 60% scaled retrieval

- **Training Parameters**:
  - `--topk`: Number of sequences to retrieve
  - `--meta_lr`: Outer loop learning rate
  - `--inner_lr`: Inner loop learning rate
  - `--num_epochs`: Number of training epochs
  - `--meta_batch_size`: Global Reptile task count per outer update
  - `--eval_batch_size`: Per-process validation loader batch size
  - `--num_workers`: Data loader worker count per process
  - `--distributed_backend`: Distributed backend, normally `nccl`
  - `--checkpoint_path`: Explicit checkpoint path for test mode
  - `--seed`: Random seed for task and support-set shuffling
  - `--embedding_cache_dir`: Directory for persistent float32 ESM1v embeddings
  - `--embedding_memory_cache_size`: Maximum number of CPU embeddings kept in memory
  - `--disable_embedding_cache`: Disable persistent and in-memory embedding reuse



## Citation

Please cite our work if you use this code in your research:

```bibtex
@article{zhang2024deep,
    title={A Deep Retrieval-Enhanced Meta-learning Framework for Enzyme Optimum pH Prediction},
    author={Liang Zhang, Kuan Luo, Ziyi Zhou, Yuanxi Yu, Fan Jiang, Banghao Wu, Mingchen Li, and Liang Hong},
    journal={Under Review},   
    year={2024},

}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## pH-GeoFuse

`phgeofuse` is the structure-aware successor implemented alongside the original
Venus-DREAM entry points. It combines cached SaProt-650M residue embeddings, a
residue EGNN, pH-conditioned ionization features, and homology-aware SaProt and
Foldseek retrieval. The original MAML/Reptile workflow remains unchanged.

Upgrade the existing environment without replacing its PyTorch/CUDA stack:

```bash
bash scripts/install_phgeofuse.sh phbench
conda activate phbench
python -m phgeofuse.doctor --config configs/phgeofuse_phopt.yaml
```

Prepare structures and graph features online, then cache frozen SaProt features:

```bash
python -m phgeofuse.prepare --config configs/phgeofuse_phopt.yaml \
  --dataset identity20 --online
torchrun --standalone --nproc_per_node=1 \
  -m phgeofuse.encode --config configs/phgeofuse_phopt.yaml \
  --dataset identity20
```

The preparation command first validates an AlphaFold DB structure against the
exact PHOPT FASTA sequence. Missing or mismatched entries use ESMFold. Use
`--offline` on machines without network access; it fails with a complete list of
missing artifacts instead of attempting downloads.

Train on one or more GPUs with the same command shape. `global_batch_size` is
kept constant by automatically changing gradient accumulation with world size.

```bash
torchrun --standalone --nproc_per_node=4 \
  -m phgeofuse.train --config configs/phgeofuse_phopt.yaml \
  --dataset identity20

torchrun --standalone --nproc_per_node=4 \
  -m phgeofuse.evaluate --config configs/phgeofuse_phopt.yaml \
  --dataset identity20 \
  --checkpoint artifacts/phgeofuse/datasets/identity20/runs/phgeofuse_phopt_mse_frozen_seed42/best.pt
```

For homology-controlled datasets, PhGeoFuse stores manifests, retrieval caches,
predictions, and runs below `artifacts/phgeofuse/datasets/<dataset>/`; structure,
graph, and SaProt embedding caches remain shared by sequence hash. Use the same
`--dataset` value for preparation, encoding, training, and evaluation. Omit the
option to retain the original PHOPT workflow and artifact paths.

Evaluate the official EpHod low-homology benchmark with the dedicated split:

```bash
CUDA_VISIBLE_DEVICES=0 python -m phgeofuse.evaluate \
  --config configs/phgeofuse_phopt_tuned_v1.yaml \
  --checkpoint artifacts/phgeofuse/runs/phgeofuse_phopt_tuned_mse_v1_frozen_seed42/best.pt \
  --split test_low_identity
```

The final EpHod paper and its Zenodo metadata define this subset as 999 of the
1,971 test sequences whose maximum identity to the training set is below 20%.
The often-cited value 499 is not the size in the final release. The checked-in
`data/phopt_testing_low_identity.fasta` is generated from the official
`Test <20% to Train` field in Zenodo record 14252615. Rebuild it with:

```bash
python scripts/build_ephod_low_identity_subset.py
```

Subset metrics include requested/evaluated counts, coverage, and unavailable
IDs so a partially prepared structure manifest cannot silently masquerade as
the complete 999-sequence benchmark.

For multi-node execution, replace `--standalone` with the normal `torchrun`
`--nnodes`, `--node_rank`, `--master_addr`, and `--master_port` arguments. All
nodes must see the same manifest, structures, feature cache, and model cache.

Set `model.mode: lora` to fine-tune SaProt attention projections with LoRA. The
default `frozen` mode is substantially cheaper and reads fp16 residue embeddings
created by `phgeofuse.encode`.

Available controlled ablations are `saprot_only`, `geometry`, `ph_conditioned`,
`saprot_retrieval`, and `full`:

```bash
torchrun --standalone --nproc_per_node=4 -m phgeofuse.train \
  --config configs/phgeofuse_phopt.yaml --ablation geometry
```

Predict directly from an unlabeled FASTA. The checkpoint embeds its resolved
configuration; `--config` remains available when cache paths differ on another
machine.

```bash
python -m phgeofuse.predict --fasta proteins.fasta \
  --checkpoint artifacts/phgeofuse/runs/phgeofuse_phopt_frozen_seed42/best.pt \
  --offline
```

Run lightweight tests and the two-process CPU distributed smoke test with:

```bash
python -m unittest tests.test_phgeofuse
torchrun --standalone --nproc_per_node=2 tests/phgeofuse_ddp_smoke.py
torchrun --standalone --nproc_per_node=2 tests/phgeofuse_training_ddp_smoke.py
```
