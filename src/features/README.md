# Feature Construction

This module defines feature construction pipelines used to represent
genomic bins (nodes) in scHi-C graphs.

All feature extraction steps operate on the **standardized outputs of
preprocessing** and produce chromosome-wise node features that can be
directly consumed by graph learning models.

---

## Overview

Each genomic bin is represented by a feature vector composed of:

1. **Sequence-derived features**  
   DNA embeddings extracted from the reference genome

2. **Structure-derived features**  
   Local graph statistics computed from scHi-C contact matrices

These features are computed independently and optionally merged into a
single node representation.

---

## Outputs
All feature pipelines produce chromosome-wise feature files:
```text
<cell_id>.npz
 ├─ chr1 → (n_bins, d)
 ├─ chr2 → (n_bins, d)
 └─ ...
```
Each array corresponds to node features for one chromosome.

## Implemented Features
### 1. DNA sequence embeddings (dna_embed_hyenadna.py)
All feature pipelines produce chromosome-wise feature files:


- Extracts embeddings from genomic DNA sequences using HyenaDNA

- Each genomic bin is mapped to a fixed-dimensional embedding vector

- Features are shared across all cells (genome-dependent)

**Outputs**
```text
bin_dna_features.npz
```

### 2. Local Degree Profile (LDP) features (ldp.py)

- Computes local structural features from scHi-C contact graphs

- Captures node degree statistics and neighborhood properties

- Features are computed per cell and per chromosome

**Outputs**
```text
<cell_id>.npz
```

### 3. Feature merging (merge_ldp_dna.py)

- Concatenates DNA embeddings and LDP features along the feature dimension

- Ensures consistent node ordering across feature types

- Produces unified node features for graph construction
**Outputs**
```text
<cell_id>.npz
```

## Usage
Feature construction is executed via scripts in the scripts/ directory:
```text
python scripts/embed_dna_hyenadna.py
python scripts/compute_ldp.py
python scripts/merge_ldp_dna.py
```
Refer to individual scripts for configurable options such as resolution,
feature dimensions, and output paths.

## Notes

- Feature extraction is fully decoupled from model training
- 
- All downstream graph models operate on the same feature format







