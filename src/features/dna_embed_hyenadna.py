"""
DNA embedding extraction using HyenaDNA (Hugging Face).

This module reads per-bin DNA sequences from a JSON list (produced by dna_bins.py),
groups sequences by chromosome, computes embeddings using a foundation model, and
saves a single merged .npz file.

Input schema (per record)
-------------------------
{
  "chrom": "chr1",
  "bin": 1,
  "sequence": "ACGT..."
}

Notes
-----
HyenaDNA does not use a dedicated [CLS] token. Embeddings are extracted from a token
position in the output:
- pool="first": output.last_hidden_state[:, 0, :]
- pool="last" : output.last_hidden_state[:, -1, :]

Output
-------------------
A single compressed .npz file with chromosome keys:

merged.npz
 ├─ chr1 → (n_bins_chr1, hidden_dim)
 ├─ chr2 → (n_bins_chr2, hidden_dim)
 ├─ ...
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


# ============================================================
# Spec
# ============================================================

@dataclass(frozen=True)
class HyenaDNASpec:
    """
    Specification for HyenaDNA embedding extraction.

    Parameters
    ----------
    input_json_path : str or pathlib.Path
        Path to JSON list with per-bin sequences (from dna_bins.py).
    output_npz_path : str or pathlib.Path
        Output path for merged .npz file.
    checkpoint : str
        Hugging Face checkpoint name.
    device : str
        Torch device string (e.g., "cuda:0", "cuda:1", "cpu").
    pool : {"first", "last"}
        Token position to use for embedding:
        - "first": output.last_hidden_state[:, 0, :]
        - "last" : output.last_hidden_state[:, -1, :]
    include_chroms : sequence[str] or None
        If provided, only these chromosomes are processed.
    exclude_chroms : sequence[str]
        Chromosomes to skip (e.g., chrX, chrY, chrM).
    clear_cache_every : int
        Call torch.cuda.empty_cache() periodically on CUDA to reduce fragmentation.
    """
    input_json_path: Union[str, Path]
    output_npz_path: Union[str, Path]
    checkpoint: str = "LongSafari/hyenadna-large-1m-seqlen-hf"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    pool: str = "last"
    include_chroms: Optional[Sequence[str]] = None
    exclude_chroms: Sequence[str] = ("chrX", "chrY", "chrM")
    clear_cache_every: int = 64


# ============================================================
# Utilities
# ============================================================

def canonical_chrom_name(chrom: str) -> str:
    """Normalize chromosome names to UCSC-style (chr-prefixed)."""
    chrom = chrom.strip()
    return chrom if chrom.startswith("chr") else f"chr{chrom}"


def load_bin_records(input_json_path: Union[str, Path]) -> List[Mapping[str, object]]:
    """Load JSON list of bin records."""
    p = Path(input_json_path)
    if not p.exists():
        raise FileNotFoundError(f"Input JSON not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of records.")
    return data


def group_sequences_by_chrom(
    records: Sequence[Mapping[str, object]],
    *,
    include_chroms: Optional[Sequence[str]] = None,
    exclude_chroms: Sequence[str] = ("chrX", "chrY", "chrM"),
) -> Dict[str, List[str]]:
    """
    Group sequences by chromosome. Output chromosome names are canonicalized to 'chr*'.
    """
    include_set = set(canonical_chrom_name(c) for c in include_chroms) if include_chroms else None
    exclude_set = set(canonical_chrom_name(c) for c in exclude_chroms)

    grouped: Dict[str, List[str]] = {}
    for r in records:
        chrom = canonical_chrom_name(str(r["chrom"]))
        if chrom in exclude_set:
            continue
        if include_set is not None and chrom not in include_set:
            continue

        seq = str(r["sequence"])
        grouped.setdefault(chrom, []).append(seq)

    # stable ordering for keys (lexicographic; fine for chr1..chr22 when consistent)
    return dict(sorted(grouped.items(), key=lambda x: x[0]))


def load_model_and_tokenizer(checkpoint: str, device: str) -> Tuple[AutoTokenizer, AutoModel]:
    """Load model/tokenizer and move model to device."""
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModel.from_pretrained(checkpoint, torch_dtype=torch.float32, trust_remote_code=True)
    model.to(torch.device(device))
    model.eval()
    return tokenizer, model


def embed_sequence(
    sequence: str,
    *,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    pool: str,
) -> np.ndarray:
    """
    Embed one DNA sequence using token position pooling.

    pool : {"first", "last"}
    """
    tokens = tokenizer(sequence)["input_ids"]
    x = torch.as_tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    with torch.inference_mode():
        out = model(x)
        if pool == "first":
            emb = out.last_hidden_state[:, 0, :]
        elif pool == "last":
            emb = out.last_hidden_state[:, -1, :]
        else:
            raise ValueError(f"Unknown pool='{pool}'. Use 'first' or 'last'.")

    return emb.squeeze(0).detach().cpu().numpy()


# ============================================================
# Core
# ============================================================

def compute_embeddings(
    chr_to_seqs: Mapping[str, Sequence[str]],
    *,
    checkpoint: str,
    device: str,
    pool: str,
    clear_cache_every: int = 64,
) -> Dict[str, np.ndarray]:
    """
    Compute embeddings for all chromosomes.

    Returns
    -------
    dict[str, np.ndarray]
        Key: chromosome name (chr*)
        Value: array of shape (n_bins, hidden_dim)
    """
    torch_device = torch.device(device)
    tokenizer, model = load_model_and_tokenizer(checkpoint, device)

    results: Dict[str, np.ndarray] = {}
    for chrom, seqs in tqdm(chr_to_seqs.items(), desc="Processing chromosomes"):
        embs: List[np.ndarray] = []

        for i, seq in enumerate(tqdm(seqs, desc=f"Embedding {chrom}", leave=False)):
            embs.append(embed_sequence(seq, tokenizer=tokenizer, model=model, device=torch_device, pool=pool))

            if torch_device.type == "cuda" and (i + 1) % clear_cache_every == 0:
                torch.cuda.empty_cache()

        results[chrom] = np.stack(embs, axis=0)

    return results


def save_merged_npz(embeddings: Mapping[str, np.ndarray], output_npz_path: Union[str, Path]) -> Path:
    """
    Save merged .npz with keys=chromosomes (chr*).
    """
    out_path = Path(output_npz_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(out_path, **{k: v for k, v in embeddings.items()})
    print(f"Saved merged embeddings: {out_path}")
    return out_path


# ============================================================
# Main runner
# ============================================================

def run_hyenadna_embedding(spec: HyenaDNASpec) -> Path:
    """
    End-to-end runner: JSON -> chr-wise embeddings -> merged NPZ.

    Returns
    -------
    pathlib.Path
        Saved merged npz path.
    """
    records = load_bin_records(spec.input_json_path)
    chr_to_seqs = group_sequences_by_chrom(
        records,
        include_chroms=spec.include_chroms,
        exclude_chroms=spec.exclude_chroms,
    )

    if len(chr_to_seqs) == 0:
        raise RuntimeError("No sequences to process after chromosome filtering.")

    print(f"Loaded {len(records)} records from {spec.input_json_path}")
    print(f"Processing {len(chr_to_seqs)} chromosomes (checkpoint={spec.checkpoint}, pool={spec.pool})")

    embeddings = compute_embeddings(
        chr_to_seqs,
        checkpoint=spec.checkpoint,
        device=spec.device,
        pool=spec.pool,
        clear_cache_every=spec.clear_cache_every,
    )

    return save_merged_npz(embeddings, spec.output_npz_path)

