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

- Single-cell Hi-C (scHi-C) enables the investigation of 3D genome organization at single-cell resolution, but its extreme sparsity and noise severely limit the recovery of meaningful chromatin interaction signals and the interpretation of cell-to-cell structural variability.
- To address this challenge, **scGRAPE** reformulates scHi-C contact maps as graphs and applies a Graph Autoencoder (GAE)-based representation learning framework to model sparse chromatin interaction topology.
- By integrating sequence embeddings derived from the large-scale DNA foundation model **HyenaDNA** as node features, scGRAPE incorporates sequence-level structural priors beyond raw contact intensity.
- The learned latent space consistently captures biologically meaningful variations in 3D genome architecture across cells and resolutions.
- Furthermore, scGRAPE enables unified downstream analyses—including chromatin loop prediction, TAD boundary detection, and A/B compartment inference—within a single, coherent latent representation framework.

  
--- 
