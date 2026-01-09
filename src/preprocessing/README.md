# Preprocessing

This directory contains dataset-specific preprocessing pipelines for
single-cell Hi-C (scHi-C) data.

The goal of preprocessing is to convert heterogeneous raw scHi-C formats
into a unified, chromosome-wise sparse matrix representation that can be
shared across downstream analysis and graph learning pipelines.

---

## Supported Datasets

- **Lee et al.**
  - Input format: multi-resolution Cooler files (`.mcool`)
  - Genome metadata and binning are handled internally by `cooler`

- **Nagano et al.**
  - Input format: schic2 per-cell interaction outputs
  - Requires external chromosome size information for binning

---

## What Preprocessing Does

- Loads dataset-specific raw scHi-C files
- Removes unwanted chromosomes (e.g., chrX, chrY, chrM)
- Constructs chromosome-wise contact matrices at a fixed resolution
- Applies optional cell-level quality control
- Saves results in a unified `.npz` format

Each output file corresponds to a single cell.

---

## Output Format

All preprocessing pipelines produce compressed `.npz` files with the same structure:

```text
cell_id.npz
 ├─ chr1 → scipy.sparse.csr_matrix
 ├─ chr2 → scipy.sparse.csr_matrix
 ├─ ...

---

## Usage

Preprocessing is executed via CLI scripts located in the scripts/ directory:

```text
python scripts/preprocess_lee.py --help
python scripts/preprocess_nagano.py --help

Refer to individual scripts for dataset-specific options.

---

## Note

Dataset-specific differences are intentionally handled only at the
preprocessing stage. All downstream components operate on the same
data representation.
