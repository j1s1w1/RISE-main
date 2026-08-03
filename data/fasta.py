#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FASTA loading and clean-label bookkeeping for RISE/TMNR2.

This module is extracted from the original
``esm2_bilstm_attn_baseline.py`` and
``dual_esm2_esmfold_tmnr2_shared_viewT.py`` implementations.

The public functions preserve the original record format::

    {"id": str, "seq": str, "label": int}

Clean reference FASTA files are used only for validation and label-correction
bookkeeping. They must not be used as training supervision.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

PathLike = Union[str, os.PathLike[str]]
Record = Dict[str, Any]


def normalize_sequence(sequence: Any) -> str:
    """Return an uppercase amino-acid sequence with all whitespace removed."""
    return "".join(str(sequence).strip().upper().split())


# Backward-compatible name used by the original SharedT implementation.
norm_seq = normalize_sequence


def read_fasta(fp: PathLike, label: int) -> List[Record]:
    """Read a FASTA file and attach the same binary label to every record.

    Parameters
    ----------
    fp:
        FASTA file path.
    label:
        Binary class label. AMP is conventionally 1 and non-AMP is 0.

    Returns
    -------
    list of dict
        Records in the original format: ``id``, ``seq`` and ``label``.
    """
    path = Path(fp)
    if not path.is_file():
        raise FileNotFoundError(f"FASTA file not found: {path}")

    records: List[Record] = []
    header: Optional[str] = None
    seq_parts: List[str] = []

    def flush_record() -> None:
        nonlocal header, seq_parts
        if header is None:
            return
        seq = normalize_sequence("".join(seq_parts))
        if seq:
            records.append({"id": header, "seq": seq, "label": int(label)})

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush_record()
                header = line[1:].strip()
                seq_parts = []
            else:
                seq_parts.append(line)
        flush_record()

    return records


def load_binary_split(amp_fasta: PathLike, nonamp_fasta: PathLike) -> List[Record]:
    """Load one AMP/non-AMP split while preserving the original ordering."""
    return read_fasta(amp_fasta, 1) + read_fasta(nonamp_fasta, 0)


def _print_split_summary(name: str, records: Sequence[Mapping[str, Any]]) -> None:
    amp_n = sum(int(record["label"]) for record in records)
    print(
        f"✅ {name} samples: {len(records)} | "
        f"AMP={amp_n} | nonAMP={len(records) - amp_n}"
    )


def load_train_test(args: Any) -> Tuple[List[Record], List[Record]]:
    """Load train/test FASTA files from an argparse-style namespace.

    Required attributes are ``train_amp``, ``train_nonamp``, ``test_amp`` and
    ``test_nonamp``. The signature is unchanged from the original baseline, so
    existing training code can import this function without call-site changes.
    """
    required = ("train_amp", "train_nonamp", "test_amp", "test_nonamp")
    missing = [name for name in required if not getattr(args, name, None)]
    if missing:
        raise ValueError(f"Missing FASTA arguments: {', '.join(missing)}")

    train = load_binary_split(args.train_amp, args.train_nonamp)
    test = load_binary_split(args.test_amp, args.test_nonamp)

    _print_split_summary("train", train)
    _print_split_summary("test ", test)
    return train, test


def read_clean_ref_fasta(fp: Optional[PathLike], label: int) -> Dict[str, int]:
    """Read a clean reference FASTA and return ``normalized sequence -> label``.

    Duplicate sequences keep the first assigned label, matching the original
    SharedT implementation.
    """
    mapping: Dict[str, int] = {}
    if not fp:
        return mapping

    path = Path(fp)
    if not path.is_file():
        raise FileNotFoundError(f"clean reference fasta not found: {path}")

    sequence_parts: List[str] = []

    def flush() -> None:
        if not sequence_parts:
            return
        seq = normalize_sequence("".join(sequence_parts))
        if seq:
            mapping.setdefault(seq, int(label))

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                sequence_parts.clear()
            else:
                sequence_parts.append(line)
        flush()

    return mapping


def build_clean_labels_from_reference(
    seqs: Sequence[str],
    args: Any,
    observed_labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Build clean labels for validation and correction auditing only.

    Returns
    -------
    clean_labels:
        Reference label when available, otherwise the observed input label.
    matched_mask:
        Whether each sequence was matched unambiguously in the clean reference.
    conflict_count:
        Number of sequences occurring in both AMP and non-AMP references.
    """
    observed = np.asarray(observed_labels, dtype=np.int64)
    if len(seqs) != len(observed):
        raise ValueError(
            f"Sequence/label length mismatch: {len(seqs)} vs {len(observed)}"
        )

    clean_ref_amp = getattr(args, "clean_ref_amp", None)
    clean_ref_nonamp = getattr(args, "clean_ref_nonamp", None)
    if not clean_ref_amp and not clean_ref_nonamp:
        return observed.copy(), np.zeros_like(observed, dtype=bool), 0

    amp_ref = read_clean_ref_fasta(clean_ref_amp, 1)
    non_ref = read_clean_ref_fasta(clean_ref_nonamp, 0)

    clean = observed.copy()
    matched = np.zeros_like(observed, dtype=bool)
    conflict_count = 0

    for index, sequence in enumerate(seqs):
        normalized = normalize_sequence(sequence)
        in_amp = normalized in amp_ref
        in_nonamp = normalized in non_ref

        if in_amp and in_nonamp:
            conflict_count += 1
            clean[index] = int(observed[index])
            matched[index] = False
        elif in_amp:
            clean[index] = 1
            matched[index] = True
        elif in_nonamp:
            clean[index] = 0
            matched[index] = True
        else:
            clean[index] = int(observed[index])
            matched[index] = False

    return clean.astype(np.int64), matched, conflict_count


def resolve_noise_source(args: Any) -> str:
    """Resolve whether noisy labels come from files or internal injection.

    ``file`` means the input FASTA files are already poisoned and no additional
    noise may be injected. ``internal`` means symmetric noise is generated from
    clean input labels. ``auto`` follows the original path-based detection.
    """
    source = str(getattr(args, "noise_source", "auto"))
    if source in {"file", "internal"}:
        return source
    if source != "auto":
        raise ValueError(
            f"Unsupported noise_source={source!r}; expected auto, file or internal"
        )

    train_paths = " ".join(
        [
            str(getattr(args, "train_amp", "")),
            str(getattr(args, "train_nonamp", "")),
        ]
    )

    if re.search(r"[/\\]noise[/\\]noise_[0-9.]+[/\\]rep[0-9]+", train_paths):
        return "file"
    if re.search(r"noise_[0-9.]+[/\\]rep[0-9]+", train_paths):
        return "file"

    noise_rate = float(getattr(args, "noise_rate", 0.0))
    return "internal" if noise_rate > 0 else "file"


__all__ = [
    "Record",
    "normalize_sequence",
    "norm_seq",
    "read_fasta",
    "load_binary_split",
    "load_train_test",
    "read_clean_ref_fasta",
    "build_clean_labels_from_reference",
    "resolve_noise_source",
]
