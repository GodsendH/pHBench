# Venus-DREAM

Venus-DREAM is a retrieval-enhanced meta-learning framework for enzyme optimum
pH prediction. It provides ESM2-based sequence retrieval, MAML and Reptile
training, and progressively lower-homology benchmark splits.

Run every command in this document from the repository root. All paths in the
code are relative to that directory.

## Installation

Python 3.10 is recommended because the pinned PyTorch and learn2learn versions
in `requirements.txt` target that generation of Python.

```bash
conda create -n phbench python=3.10
conda activate phbench
pip install -r requirements.txt
```

The cluster must provide a CUDA driver compatible with the installed PyTorch
build. Retrieval requires the Hugging Face ESM2 model
`facebook/esm2_t33_650M_UR50D`. Training uses the ESM1v and RLAT components in
`baseline/EpHod`; their model weights must either be cached already or be
downloadable on the first run.

MMseqs2 is only required when rebuilding the low-homology datasets:

```bash
conda install -c conda-forge -c bioconda mmseqs2
```

## End-to-End Workflow

For any dataset, prepare retrieval JSON files before starting MAML or Reptile:

```text
FASTA files
  -> retrieval.py
  -> retrieval_{train,valid,test}.json
  -> maml.py or reptile.py
  -> checkpoint, TensorBoard logs, metrics, and predictions CSV
```

A minimal PHOPT run is:

```bash
python retrieval.py --dataset phopt --strategy opt_retrieval --topk 5

mkdir -p predictions
python reptile.py \
    --dataset phopt \
    --mode train \
    --retrieval_strategy opt_retrieval \
    --topk 5 \
    --pretrained
```

The `--dataset`, retrieval strategy, and `--topk` values used for training must
match a retrieval directory that has already been prepared.

## Datasets

The same `--dataset` argument is accepted by `retrieval.py`, `maml.py`, and
`reptile.py`.

| Dataset | Input FASTA files | Dataset root |
| --- | --- | --- |
| `phopt` | `data/phopt_{training,validation,testing}.fasta` | `data` |
| `dedup` | `dedup_data/dedup/{train,val,test}.fasta` | `dedup_data` |
| `homology100` | `homology_data/identity100/{train,valid,test}.fasta` | `homology_data/identity100` |
| `homology90` | `homology_data/identity90/{train,valid,test}.fasta` | `homology_data/identity90` |
| `homology70` | `homology_data/identity70/{train,valid,test}.fasta` | `homology_data/identity70` |
| `homology50` | `homology_data/identity50/{train,valid,test}.fasta` | `homology_data/identity50` |
| `homology30` | `homology_data/identity30/{train,valid,test}.fasta` | `homology_data/identity30` |

Each dataset keeps its own ESM feature caches and processed retrieval files, so
runs on `phopt`, `dedup`, and the homology ladder do not overwrite one another.

## Build the Homology Ladder

The repository includes the generated `homology_data` directory. Rebuilding is
only necessary when changing the source FASTA files, identity thresholds, or
split parameters.

`build_homology_ladder.py` performs global exact-sequence cleaning, removes
exact-sequence groups with conflicting pH labels, runs all-vs-all MMseqs2, forms
connected components at each identity threshold, and assigns whole components
to train, validation, or test. This prevents threshold-level homology edges
from crossing split boundaries.

To build a separate copy for inspection:

```bash
python build_homology_ladder.py \
    --train data/phopt_training.fasta \
    --valid data/phopt_validation.fasta \
    --test data/phopt_testing.fasta \
    --output-root homology_data_rebuilt \
    --thresholds 100 90 70 50 30 \
    --coverage 0.8 \
    --sensitivity 7.5 \
    --ph-bin-width 0.5 \
    --split-attempts 64 \
    --seed 0 \
    --threads "${SLURM_CPUS_PER_TASK:-1}"
```

The builder intentionally refuses to overwrite an existing output directory.
To rebuild the datasets selected by `--dataset homology*`, first archive the
existing `homology_data` directory and then use `--output-root homology_data`.

Each identity directory contains:

- `train.fasta`, `valid.fasta`, and `test.fasta`
- `assignments.tsv` and `clusters.tsv`
- `manifest.json` with split statistics, hashes, MMseqs2 parameters, and
  cross-split validation results

The root manifest is `homology_data/manifest.json`; removed label conflicts are
recorded in `homology_data/removed_conflicts.tsv`.

Do not pass `--pretrained` for the homology ladder experiments. The EpHod RLAT
checkpoint was trained using the original PHOPT split, while the ladder
reassigns globally clustered sequences across new splits. Reusing that
supervised RLAT checkpoint would leak split information. ESM1v remains
pretrained and frozen even when `--pretrained` is omitted.

## Prepare Retrieval Data

Standard similarity retrieval for any registered dataset:

```bash
python retrieval.py \
    --dataset dedup \
    --strategy opt_retrieval \
    --topk 5 \
    --batch_size 32 \
    --model_name facebook/esm2_t33_650M_UR50D
```

For an offline cluster with an existing Hugging Face cache:

```bash
export HF_MODEL_CACHE=/path/to/huggingface/hub

python retrieval.py \
    --dataset dedup \
    --strategy opt_retrieval \
    --topk 5 \
    --batch_size 32 \
    --model_cache_dir "$HF_MODEL_CACHE" \
    --local_files_only
```

Available retrieval strategies are:

| Strategy | Meaning | MAML | Reptile |
| --- | --- | --- | --- |
| `opt_retrieval` | Top-k ESM2 cosine similarity | yes | yes |
| `opt_random` | Random training sequences | yes | yes |
| `opt_fixed_random` | The same fixed random support set | yes | yes |
| `opt_retrieval_scaled_0.2` | Similarity retrieval using 20% of train | no | yes |
| `opt_retrieval_scaled_0.6` | Similarity retrieval using 60% of train | no | yes |

Outputs are written to:

```text
<dataset-root>/features/opt_{train,valid,test}_features.pkl
<dataset-root>/processed/top<K>/esm2_<strategy>/retrieval_train.json
<dataset-root>/processed/top<K>/esm2_<strategy>/retrieval_valid.json
<dataset-root>/processed/top<K>/esm2_<strategy>/retrieval_test.json
```

Existing feature pickle files are reused. Delete or move them only when the
corresponding FASTA contents or ESM2 model have changed; the code does not
validate cache provenance automatically.

Custom FASTA and output paths can be supplied with `--opt_train`, `--opt_valid`,
`--opt_test`, `--features_dir`, and `--output_dir`.

## MAML

MAML is a single-process trainer. Limit visible GPUs explicitly when one GPU is
intended. The following example trains on PHOPT and tests automatically after
training:

```bash
mkdir -p predictions
CUDA_VISIBLE_DEVICES=0 python maml.py \
    --dataset phopt \
    --mode train \
    --retrieval_strategy opt_retrieval \
    --topk 5 \
    --pretrained \
    --num_epochs 50 \
    --meta_lr 0.0001 \
    --inner_lr 0.0005 \
    --inner_steps 5 \
    --support_batch_size 1 \
    --validate_every 200 \
    --patience 5 \
    --seed 0
```

For a homology dataset, change `--dataset` and omit `--pretrained`:

```bash
mkdir -p predictions
CUDA_VISIBLE_DEVICES=0 python maml.py \
    --dataset homology50 \
    --mode train \
    --retrieval_strategy opt_retrieval \
    --topk 5 \
    --num_epochs 50 \
    --meta_lr 0.0001 \
    --inner_lr 0.0005 \
    --inner_steps 5 \
    --support_batch_size 1 \
    --validate_every 200 \
    --patience 5 \
    --seed 0
```

MAML test mode reconstructs the checkpoint directory from the run arguments.
Repeat every argument that affects the training run name and change only the
mode:

```bash
mkdir -p predictions
CUDA_VISIBLE_DEVICES=0 python maml.py \
    --dataset phopt \
    --mode test \
    --retrieval_strategy opt_retrieval \
    --topk 5 \
    --pretrained \
    --num_epochs 50 \
    --meta_lr 0.0001 \
    --inner_lr 0.0005 \
    --inner_steps 5 \
    --support_batch_size 1 \
    --validate_every 200 \
    --patience 5 \
    --seed 0
```

MAML does not currently expose `--checkpoint_path`. `--save_dir` must therefore
be the same root used for training, and all run-name parameters must match.

## Reptile

### Single GPU

```bash
CUDA_VISIBLE_DEVICES=0 python reptile.py \
    --dataset phopt \
    --mode train \
    --retrieval_strategy opt_retrieval \
    --topk 5 \
    --pretrained \
    --num_epochs 50 \
    --meta_lr 1 \
    --inner_lr 0.001 \
    --inner_steps 5 \
    --support_batch_size 1 \
    --meta_batch_size 5 \
    --eval_batch_size 10 \
    --num_workers 4 \
    --validate_every 200 \
    --patience 5 \
    --seed 0
```

Training restores the best validated state when one has been recorded, tests
it, and writes predictions. For an explicit test run, either repeat the
training arguments or supply the checkpoint directly:

```bash
CUDA_VISIBLE_DEVICES=0 python reptile.py \
    --dataset phopt \
    --mode test \
    --checkpoint_path saved_models/<training-run-name>/best_model.pth \
    --retrieval_strategy opt_retrieval \
    --topk 5 \
    --pretrained \
    --inner_lr 0.001 \
    --inner_steps 5 \
    --support_batch_size 1 \
    --seed 0
```

The dataset, retrieval source, inner-loop learning rate, adaptation steps, and
support batch size are part of evaluation behavior and should match training.

### Distributed Reptile

Reptile supports one process per GPU through `torchrun`. A single-node,
four-GPU example is:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=4

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    reptile.py \
    --dataset phopt \
    --mode train \
    --retrieval_strategy opt_retrieval \
    --topk 5 \
    --pretrained \
    --num_epochs 50 \
    --meta_lr 1 \
    --inner_lr 0.001 \
    --inner_steps 5 \
    --support_batch_size 1 \
    --meta_batch_size 8 \
    --eval_batch_size 10 \
    --num_workers 4 \
    --validate_every 200 \
    --patience 5 \
    --seed 0
```

Multi-node template:

```bash
torchrun \
    --nnodes="$NUM_NODES" \
    --nproc_per_node="$GPUS_PER_NODE" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    reptile.py \
    --dataset phopt \
    --mode train \
    --retrieval_strategy opt_retrieval \
    --topk 5 \
    --pretrained \
    --meta_batch_size "$GLOBAL_META_BATCH_SIZE" \
    --num_workers "$WORKERS_PER_PROCESS"
```

`--meta_batch_size` is global across all processes. It must be at least the
world size; a value divisible by the world size avoids uneven per-rank work.
`--num_workers` is per process, so the total worker count is that value times
the number of GPUs. The run name includes world size and global meta-batch
size. Use `--checkpoint_path` when testing with a different world size.

## Random-Test Control

Both trainers accept `--random_test`. This uses a separately prepared
`opt_random`, top-5 test JSON regardless of the main training strategy. Prepare
it first:

```bash
python retrieval.py --dataset phopt --strategy opt_random --topk 5
```

Then add `--random_test` to the MAML or Reptile command. Use the same dataset in
both commands.

## Outputs and Caches

Default training outputs are:

```text
saved_models/<run-name>/best_model.pth
logs/<run-name>/
predictions/predictions_<run-name>.csv
```

The first training pass lazily caches frozen ESM1v embeddings under:

```text
<dataset-root>/features/esm1v_t33_650M_UR90S_1/
```

Use `--embedding_cache_dir` to override this location,
`--embedding_memory_cache_size` to change the in-memory LRU size, or
`--disable_embedding_cache` to disable both persistent and in-memory reuse.

Important operational notes:

- `retrieval.py` uses CUDA only when `torch.cuda.is_available()` succeeds. A
  CUDA initialization warning followed by `Using device: cpu` means feature
  extraction is running on CPU, even if an A800 was requested by the job.
- `retrieval.py` only shows a batch progress bar after FASTA parsing and model
  loading. ESM2 feature extraction is the expensive part on a new dataset.
- MAML requires the `predictions` directory to exist; the commands above create
  it. Reptile creates its predictions directory automatically.
- Validation must run at least once to create `best_model.pth`. If the number
  of training updates is smaller than `--validate_every`, lower that value.
- Cached ESM2 pickle files and ESM1v tensors are dataset-specific but are not
  automatically invalidated when FASTA contents change.

## Tests

```bash
pytest -q
```

The distributed Reptile smoke test can be run separately in a suitable CUDA
environment:

```bash
torchrun --standalone --nproc_per_node=2 tests/distributed_smoke.py
```

## Citation

```bibtex
@article{zhang2024deep,
    title={A Deep Retrieval-Enhanced Meta-learning Framework for Enzyme Optimum pH Prediction},
    author={Liang Zhang, Kuan Luo, Ziyi Zhou, Yuanxi Yu, Fan Jiang, Banghao Wu, Mingchen Li, and Liang Hong},
    journal={Under Review},
    year={2024}
}
```

## License

The bundled EpHod implementation retains its own license in
`baseline/EpHod/LICENSE`.
