# scGRAPE: Graph Representation and Anchor Pair Extraction for scHi-C

**scGRAPE** is a graph-based deep learning framework for analyzing single-cell Hi-C (scHi-C) data.
It reconstructs sparse chromatin contact graphs and predicts higher-order chromatin structures,
including **chromatin loops**, **topologically associating domains (TADs)**, and **A/B compartments**,
using a unified graph representation learning strategy.

This repository provides an end-to-end pipeline from raw scHi-C contact matrices to downstream
3D genome structure inference.

<p align="center">
  <img src="figures/figure1.png" width="1000">
</p>

---

## Background

<p align="center">
  <img src="figures/figure3.png" width="1000">
</p>

- Single-cell Hi-C (scHi-C) enables the investigation of 3D genome organization at single-cell resolution, but its extreme sparsity and noise severely limit the recovery of meaningful chromatin interaction signals and the interpretation of cell-to-cell structural variability.
- To address this challenge, **scGRAPE** reformulates scHi-C contact maps as graphs and applies a Graph Autoencoder (GAE)-based representation learning framework to model sparse chromatin interaction topology.
- By integrating sequence embeddings derived from the large-scale DNA foundation model **HyenaDNA** as node features, scGRAPE incorporates sequence-level structural priors beyond raw contact intensity.
- The learned latent space consistently captures biologically meaningful variations in 3D genome architecture across cells and resolutions.
- Furthermore, scGRAPE enables unified downstream analyses—including chromatin loop prediction, TAD boundary detection, and A/B compartment inference—within a single, coherent latent representation framework.

  
--- 

## Usage

scGRAPE provides a modular pipeline for learning cell-level representations of 3D genome organization from single-cell Hi-C (scHi-C) data. Dataset-specific preprocessing is performed once, after which all downstream analyses operate on a unified graph representation.

### 1. Preprocess raw scHi-C data

Convert heterogeneous raw scHi-C formats into a standardized, chromosome-wise sparse matrix representation (.npz), and compute cell-level embeddings used for cosine alignment during training.

```code
python scripts/preprocess_lee.py --help
python scripts/preprocess_nagano.py --help
```
- Details: src/preprocessing/README.md

### 2. Construct node features
Build node features for each genomic bin using sequence- and structure-derived information.

```code
python scripts/embed_dna_hyenadna.py
python scripts/compute_ldp.py
python scripts/merge_ldp_dna.py
```
- Details: src/features/README.md

### 3. Build scHi-C graphs
Combine standardized contact matrices and node features into per-(cell, chromosome) graph objects compatible with PyTorch Geometric (.pt).

```code
python scripts/preprocess_lee.py --help
python scripts/preprocess_nagano.py --help
```

### 4. Train the model
Model training is performed in two stages: pretraining for representation learning and fine-tuning for loop prediction.

#### Pretraining
`pretrain_gae.py` trains a Graph Autoencoder (GAE-FiLM) on scHi-C graphs to learn
node and graph-level latent representations.

- Optimizes **graph reconstruction (BCE)**
- Includes **cosine alignment to PCA-based cell embeddings** to encode cell-level structural variation

```code
python pretrain_gae.py [-h]
required arguments:
    --dataset {lee,nagano,pt}       Dataset type
    --train_dir TRAIN_DIR           Directory containing training .pt graphs
    --val_dir VAL_DIR               Directory containing validation .pt graphs
    --out_dir OUT_DIR               Output directory for checkpoints and logs

optional arguments:
  --cell_emb_dir CELL_EMB_DIR      Directory with cell_embeddings.npy and cell_names.txt
  --epochs EPOCHS                  Number of training epochs (default: 150)
  --lr LR                          Learning rate (default: 1e-3)
  --hidden_dim HIDDEN_DIM          Encoder hidden dimension (default: 128)
  --z_dim Z_DIM                    Latent embedding dimension (default: 32)
```

#### Fine-tuning (Loop prediction)
python finetune_loops.py [-h]
required arguments:
  --train_dir TRAIN_DIR            Directory containing training .pt graphs
  --val_dir VAL_DIR                Directory containing validation .pt graphs
  --label_path LABEL_PATH          Dataset-specific label/metadata file
  --cell_emb_dir CELL_EMB_DIR      Directory with PCA-based cell embeddings
  --ref_loop_path REF_LOOP_PATH    Reference loop file (.pt)
  --pretrain_model PRETRAIN_MODEL  Path to pretrained model checkpoint
  --out_dir OUT_DIR                Output directory for fine-tuned models
```


