# Feature Construction

This module constructs node-level features for scHi-C graphs.
All feature pipelines operate on the standardized, chromosome-wise contact
matrices produced during preprocessing.

The resulting features are used as node attributes (`x`) in graph-based
learning models.

---

## Overview

Two complementary feature types are supported:

- **Structural features** derived from scHi-C contact topology
- **Sequence-based features** derived from genomic DNA

These features can be used independently or merged into a unified
node feature representation.

---

## Implemented Features

### 1. Local Degree Profile (LDP)

Local Degree Profile (LDP) features capture local graph topology around
each genomic bin, summarizing interaction density and neighborhood structure.

- Computed directly from chromosome-wise contact matrices
- Resolution-aware and chromosome-specific
- Encodes local connectivity patterns in scHi-C graphs

**Implementation**
- `ldp.py` — core LDP feature computation
- `scripts/compute_ldp.py` — CLI wrapper for batch processing

**Output**
```text
<cell_id>.npz
 ├─ chr1 → (n_bins, d_ldp)
 ├─ chr2 → (n_bins, d_ldp)
 ├─ ...
