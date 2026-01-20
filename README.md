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
scGRAPE training is performed in two stages: (i) pretraining a GAE-FiLM encoder for robust representation learning from sparse scHi-C graphs, followed by (ii) task-specific fine-tuning with an explicit loop prediction head.

```code
python scripts/pretrain_gae.py
python scripts/finetune_loops.py
```

