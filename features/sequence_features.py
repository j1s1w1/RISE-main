#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ESM2 per-residue sequence feature extraction for RISE.

This module contains only frozen ESM2 loading, per-token embedding extraction,
and cache I/O. It does not contain the trainable BiLSTM-attention encoder.

The cache format is intentionally unchanged from the original implementation::

    {
        "ids": List[str],
        "seqs": List[str],
        "labels": LongTensor[N],
        "embeddings": List[FloatTensor[L_i, D]],
    }
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

import torch

if __package__:
    from data.fasta import load_binary_split
else:  # Support: python rise/features/sequence_features.py ...
    import sys

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from rise.data.fasta import load_binary_split

PathLike = Union[str, os.PathLike[str]]
SequenceFeatureCache = Dict[str, Any]


def safe_torch_load(fp: PathLike, device: Union[str, torch.device] = "cpu") -> Any:
    """Load legacy or current PyTorch cache files across PyTorch versions."""
    try:
        return torch.load(fp, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(fp, map_location=device)


def load_esm2_model(
    esm_model_path: PathLike,
    device: Union[str, torch.device],
) -> Tuple[torch.nn.Module, Any]:
    """Load a local HuggingFace ESM2 model and freeze all parameters."""
    model_path = Path(esm_model_path).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"ESM2 model path not found: {model_path}")

    try:
        from transformers import EsmModel, EsmTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required. Install it with: pip install transformers"
        ) from exc

    device_str = str(device)
    print("🧠 使用 transformers 加载本地 ESM2 模型 ...")
    print(f"✅ ESM2 path: {model_path}")

    tokenizer = EsmTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
    )

    dtype = torch.float16 if device_str.startswith("cuda") else torch.float32
    model = EsmModel.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
        local_files_only=True,
    ).to(device)

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"✅ ESM2 loaded | params={parameter_count / 1e6:.1f}M | "
        f"dtype={dtype}"
    )
    return model, tokenizer


def validate_sequence_feature_cache(
    data: Mapping[str, Any],
    expected_records: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    """Validate cache keys, sample counts and residue-feature lengths."""
    required = {"ids", "seqs", "labels", "embeddings"}
    missing = sorted(required.difference(data.keys()))
    if missing:
        raise ValueError(f"Sequence feature cache is missing keys: {missing}")

    sample_count = len(data["ids"])
    lengths = {
        "ids": sample_count,
        "seqs": len(data["seqs"]),
        "labels": len(data["labels"]),
        "embeddings": len(data["embeddings"]),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Inconsistent sequence cache lengths: {lengths}")

    for index, (sequence, embedding) in enumerate(
        zip(data["seqs"], data["embeddings"])
    ):
        if not isinstance(embedding, torch.Tensor) or embedding.ndim != 2:
            raise ValueError(
                f"Embedding {index} must be a rank-2 tensor, got "
                f"{type(embedding).__name__}"
            )
        if embedding.shape[0] != len(sequence):
            raise ValueError(
                f"Embedding length mismatch at {index}: "
                f"sequence={len(sequence)}, embedding={embedding.shape[0]}"
            )

    if expected_records is not None:
        if len(expected_records) != sample_count:
            raise ValueError(
                f"Record/cache length mismatch: {len(expected_records)} vs {sample_count}"
            )
        for index, record in enumerate(expected_records):
            if str(record["id"]) != str(data["ids"][index]):
                raise ValueError(f"Record/cache ID mismatch at index {index}")
            if str(record["seq"]) != str(data["seqs"][index]):
                raise ValueError(f"Record/cache sequence mismatch at index {index}")
            if int(record["label"]) != int(data["labels"][index]):
                raise ValueError(f"Record/cache label mismatch at index {index}")


@torch.no_grad()
def extract_token_embeddings(
    records: List[Dict[str, Any]],
    esm_model_path: PathLike,
    cache_fp: PathLike,
    device: Union[str, torch.device],
    esm_batch_size: int = 4,
    repr_layer: int = 33,
) -> SequenceFeatureCache:
    """Extract frozen ESM2 per-residue embeddings and save a reusable cache.

    ``repr_layer`` is retained for command-line and cache-name compatibility.
    HuggingFace ``EsmModel`` returns the final hidden state used by the original
    implementation, so the parameter does not alter the forward call.
    """
    del repr_layer  # Compatibility parameter; the original HF code did not use it.

    cache_path = Path(cache_fp).expanduser()
    if cache_path.is_file():
        print(f"✅ token embeddings 已缓存: {cache_path}")
        cached = safe_torch_load(cache_path, "cpu")
        validate_sequence_feature_cache(cached, records)
        return cached

    if not records:
        raise ValueError("Cannot extract sequence features from an empty record list")
    if esm_batch_size < 1:
        raise ValueError(f"esm_batch_size must be >= 1, got {esm_batch_size}")

    print(
        "🚀 Extracting ESM2 token embeddings with transformers: "
        f"{len(records)} samples"
    )
    model, tokenizer = load_esm2_model(esm_model_path, device)

    ids: List[str] = []
    seqs: List[str] = []
    labels: List[int] = []
    embeddings: List[torch.Tensor] = []

    target_batch_size = int(esm_batch_size)
    current_batch_size = target_batch_size
    index = 0
    start_time = time.time()
    device_str = str(device)

    while index < len(records):
        batch_records = records[index : index + current_batch_size]
        batch_sequences = [str(record["seq"]) for record in batch_records]

        try:
            inputs = tokenizer(
                batch_sequences,
                return_tensors="pt",
                padding=True,
                truncation=False,
                add_special_tokens=True,
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}

            output = model(**inputs)
            last_hidden = output.last_hidden_state

            for batch_index, record in enumerate(batch_records):
                sequence = str(record["seq"])
                sequence_length = len(sequence)

                # Position 0 is the leading special token. Residues occupy
                # positions 1 .. sequence_length in the original implementation.
                embedding = (
                    last_hidden[batch_index, 1 : 1 + sequence_length]
                    .detach()
                    .cpu()
                    .float()
                )

                if embedding.shape[0] != sequence_length:
                    attention_mask = (
                        inputs["attention_mask"][batch_index]
                        .detach()
                        .cpu()
                        .bool()
                    )
                    valid_indices = torch.where(attention_mask)[0]
                    valid_indices = valid_indices[1 : 1 + sequence_length]
                    embedding = (
                        last_hidden[
                            batch_index,
                            valid_indices.to(last_hidden.device),
                        ]
                        .detach()
                        .cpu()
                        .float()
                    )

                if embedding.shape[0] != sequence_length:
                    raise RuntimeError(
                        f"ESM2 token length mismatch for id={record['id']!r}: "
                        f"expected {sequence_length}, got {embedding.shape[0]}"
                    )

                ids.append(str(record["id"]))
                seqs.append(sequence)
                labels.append(int(record["label"]))
                embeddings.append(embedding)

            index += len(batch_records)

            if current_batch_size < target_batch_size:
                current_batch_size = min(
                    target_batch_size,
                    current_batch_size * 2,
                )

            if index == len(records) or (index // max(current_batch_size, 1)) % 10 == 0:
                elapsed = time.time() - start_time
                print(
                    f"  ESM token embeddings: {index}/{len(records)} | "
                    f"{index / max(elapsed, 1e-9):.2f} seq/s"
                )

            del inputs, output, last_hidden
            if device_str.startswith("cuda"):
                torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            if device_str.startswith("cuda"):
                torch.cuda.empty_cache()
            if current_batch_size <= 1:
                raise
            new_batch_size = max(1, current_batch_size // 2)
            print(
                f"⚠️ CUDA OOM: esm_batch_size "
                f"{current_batch_size} -> {new_batch_size}"
            )
            current_batch_size = new_batch_size

    data: SequenceFeatureCache = {
        "ids": ids,
        "seqs": seqs,
        "labels": torch.tensor(labels, dtype=torch.long),
        "embeddings": embeddings,
    }
    validate_sequence_feature_cache(data, records)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, cache_path)
    print(f"✅ Saved token embeddings: {cache_path}")

    del model, tokenizer
    if device_str.startswith("cuda"):
        torch.cuda.empty_cache()

    return data


# Clearer public name while keeping the old import unchanged.
extract_sequence_features = extract_token_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frozen ESM2 per-residue features for AMP/non-AMP FASTA files."
    )
    parser.add_argument("--amp-fasta", required=True)
    parser.add_argument("--nonamp-fasta", required=True)
    parser.add_argument("--esm-model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--esm-batch-size", type=int, default=4)
    parser.add_argument("--repr-layer", type=int, default=33)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_binary_split(args.amp_fasta, args.nonamp_fasta)
    extract_token_embeddings(
        records=records,
        esm_model_path=args.esm_model_path,
        cache_fp=args.output,
        device=args.device,
        esm_batch_size=args.esm_batch_size,
        repr_layer=args.repr_layer,
    )


if __name__ == "__main__":
    main()
