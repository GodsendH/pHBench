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

Prepare sequence retrieval data using different strategies:

```bash
python retrieval.py \
    --opt_train data/phopt_training.fasta \
    --opt_test data/phopt_testing.fasta \
    --opt_valid data/phopt_validation.fasta \
    --model_name facebook/esm2_t33_650M_UR50D \
    --features_dir data/features \
    --strategy opt_retrieval \
    --topk 5
```

### Model Training

#### MAML Training

Train the model using Model-Agnostic Meta-Learning:

```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

python maml.py \
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

Train the model using the Reptile algorithm:

```bash
python reptile.py \
    --mode train \
    --num_epochs 50 \
    --retrieval_strategy opt_retrieval \
    --topk 5 \
    --pretrained \
    --meta_lr 1 \
    --inner_lr 0.001 \
    --inner_steps 5 \
    --validate_every 200 \
    --patience 5 \
    --seed 0
```

The frozen ESM1v representations are cached lazily as float32 tensors in
`data/features/esm1v_t33_650M_UR90S_1`. The first encounter of a sequence
creates its cache file; later inner-loop steps, validation runs, and epochs
reuse it. Use `--embedding_cache_dir` to choose another location,
`--embedding_memory_cache_size` to control the in-memory LRU size, or
`--disable_embedding_cache` to run the original uncached forward path.

Run names include `inner_steps`, `support_batch_size`, and `seed` so that
experiments with different adaptation settings do not overwrite each other.


### Command Line Arguments

- **Mode Options**:
  - `--mode`: train or test
  - `--pretrained`: Use pretrained EpHod RLAT weights; ESM1v is always pretrained

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
