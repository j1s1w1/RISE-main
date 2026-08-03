#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate nested, class-balanced symmetric label noise for AMP datasets.

Supported inputs
----------------
A. CSV:
   split_dir/train.csv
   split_dir/test.csv
   split_dir/val.csv            # optional

B. FASTA:
   train_amp.fasta
   train_nonamp.fasta
   test_amp.fasta
   test_nonamp.fasta
   val_amp.fasta                # optional
   val_nonamp.fasta             # optional

When validation data are missing, a fixed clean validation set is created
from the clean training set before adding label noise.

The generated training CSV uses:
- clean_target: original clean label
- target: label used for model training
- is_noisy: 1 when the training label was flipped
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


TARGET_TO_LABEL = {0: "non-AMP", 1: "AMP"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate nested class-balanced symmetric training-label noise. "
            "Validation and test labels always remain clean."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--input-format", choices=("auto", "csv", "fasta"), default="auto")
    parser.add_argument("--split-dir", type=Path, default=None)

    parser.add_argument("--label-column", default="label")
    parser.add_argument("--sequence-column", default="sequence")
    parser.add_argument("--id-column", default=None)

    parser.add_argument("--train-amp", type=Path, default=None)
    parser.add_argument("--train-nonamp", type=Path, default=None)
    parser.add_argument("--val-amp", type=Path, default=None)
    parser.add_argument("--val-nonamp", type=Path, default=None)
    parser.add_argument("--test-amp", type=Path, default=None)
    parser.add_argument("--test-nonamp", type=Path, default=None)

    parser.add_argument(
        "--create-val-if-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--val-seed", type=int, default=42)

    parser.add_argument(
        "--noise-rates",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.2, 0.3],
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46],
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-duplicate-check", action="store_true")
    return parser.parse_args()


def normalize_sequence(value: object) -> str:
    if pd.isna(value):
        return ""
    return "".join(str(value).strip().upper().split())


def normalize_label(value: object) -> Tuple[str, int]:
    if pd.isna(value):
        raise ValueError("Missing label value.")

    if isinstance(value, (int, np.integer)):
        target = int(value)
        if target in TARGET_TO_LABEL:
            return TARGET_TO_LABEL[target], target

    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        target = int(value)
        if target in TARGET_TO_LABEL:
            return TARGET_TO_LABEL[target], target

    compact = (
        str(value).strip().lower()
        .replace("_", "").replace("-", "").replace(" ", "")
    )
    if compact in {"amp", "1", "positive", "pos"}:
        return "AMP", 1
    if compact in {"nonamp", "0", "negative", "neg"}:
        return "non-AMP", 0

    raise ValueError(
        f"Unsupported label {value!r}; expected AMP/non-AMP or 1/0."
    )


def read_fasta(path: Path, label: str) -> pd.DataFrame:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"FASTA file not found: {path}")

    records: List[Dict[str, object]] = []
    header: Optional[str] = None
    parts: List[str] = []

    def flush() -> None:
        nonlocal header, parts
        if header is None:
            return
        seq = normalize_sequence("".join(parts))
        if not seq:
            raise ValueError(f"Empty FASTA sequence: {path}, header={header}")
        records.append({"id": header, "sequence": seq, "label": label})

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                header = line[1:].strip() or f"{path.stem}_{len(records):07d}"
                parts = []
            else:
                if header is None:
                    raise ValueError(f"Sequence before FASTA header: {path}")
                parts.append(line)
        flush()

    if not records:
        raise ValueError(f"No records found in FASTA: {path}")
    return pd.DataFrame(records)


def resolve_id_column(
    frame: pd.DataFrame,
    requested: Optional[str],
) -> Optional[str]:
    if requested:
        if requested not in frame.columns:
            raise ValueError(f"ID column not found: {requested}")
        return requested

    for name in ("id", "accession_version", "accession", "name"):
        if name in frame.columns:
            return name
    return None


def prepare_clean_frame(
    frame: pd.DataFrame,
    *,
    split_name: str,
    label_column: str,
    sequence_column: str,
    id_column: Optional[str],
) -> pd.DataFrame:
    out = frame.copy()

    missing = {label_column, sequence_column} - set(out.columns)
    if missing:
        raise ValueError(f"{split_name}: missing columns {sorted(missing)}")

    out[sequence_column] = out[sequence_column].map(normalize_sequence)
    if (out[sequence_column] == "").any():
        rows = out.index[out[sequence_column] == ""].tolist()[:10]
        raise ValueError(f"{split_name}: empty sequences at rows {rows}")

    labels = out[label_column].map(normalize_label)
    out["clean_label"] = labels.map(lambda pair: pair[0])
    out["clean_target"] = labels.map(lambda pair: pair[1]).astype(int)

    resolved_id = resolve_id_column(out, id_column)
    if "id" not in out.columns:
        if resolved_id is not None:
            out.insert(0, "id", out[resolved_id].astype(str))
        else:
            out.insert(
                0,
                "id",
                [f"{split_name}_{i:07d}" for i in range(len(out))],
            )
    else:
        out["id"] = out["id"].astype(str)

    # Canonical columns used by the RISE data loader.
    out["sequence"] = out[sequence_column]
    out["label"] = out["clean_label"]
    out["noisy_label"] = out["clean_label"]
    out["target"] = out["clean_target"].astype(int)
    out["is_noisy"] = 0
    out["source_row_index"] = np.arange(len(out), dtype=int)
    out["split"] = split_name
    return out.reset_index(drop=True)


def standard_fasta_paths(split_dir: Path) -> Dict[str, Path]:
    return {
        "train_amp": split_dir / "train_amp.fasta",
        "train_nonamp": split_dir / "train_nonamp.fasta",
        "val_amp": split_dir / "val_amp.fasta",
        "val_nonamp": split_dir / "val_nonamp.fasta",
        "test_amp": split_dir / "test_amp.fasta",
        "test_nonamp": split_dir / "test_nonamp.fasta",
    }


def resolve_input_format(args: argparse.Namespace) -> str:
    if args.input_format != "auto":
        return args.input_format

    explicit_fasta = any(
        value is not None
        for value in (
            args.train_amp, args.train_nonamp,
            args.val_amp, args.val_nonamp,
            args.test_amp, args.test_nonamp,
        )
    )
    if explicit_fasta:
        return "fasta"

    if args.split_dir is None:
        raise ValueError(
            "Use --split-dir, or provide the FASTA paths explicitly."
        )

    root = args.split_dir.expanduser().resolve()
    if (root / "train.csv").is_file() and (root / "test.csv").is_file():
        return "csv"

    paths = standard_fasta_paths(root)
    required = (
        paths["train_amp"], paths["train_nonamp"],
        paths["test_amp"], paths["test_nonamp"],
    )
    if all(path.is_file() for path in required):
        return "fasta"

    raise FileNotFoundError(
        f"Could not detect CSV or standard-named FASTA splits under {root}."
    )


def load_csv(
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame, Dict[str, str]]:
    if args.split_dir is None:
        raise ValueError("--split-dir is required for CSV input.")

    root = args.split_dir.expanduser().resolve()
    paths = {
        "train": root / "train.csv",
        "val": root / "val.csv",
        "test": root / "test.csv",
    }
    for key in ("train", "test"):
        if not paths[key].is_file():
            raise FileNotFoundError(f"Required CSV not found: {paths[key]}")

    def load(name: str) -> pd.DataFrame:
        return prepare_clean_frame(
            pd.read_csv(paths[name]),
            split_name=name,
            label_column=args.label_column,
            sequence_column=args.sequence_column,
            id_column=args.id_column,
        )

    train = load("train")
    val = load("val") if paths["val"].is_file() else None
    test = load("test")
    sources = {
        key: str(path)
        for key, path in paths.items()
        if path.is_file()
    }
    return train, val, test, sources


def load_fasta(
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame, Dict[str, str]]:
    defaults = (
        standard_fasta_paths(args.split_dir.expanduser().resolve())
        if args.split_dir is not None
        else {}
    )

    def choose(explicit: Optional[Path], key: str) -> Optional[Path]:
        if explicit is not None:
            return explicit.expanduser().resolve()
        candidate = defaults.get(key)
        return candidate if candidate is not None and candidate.is_file() else None

    paths: Dict[str, Optional[Path]] = {
        "train_amp": choose(args.train_amp, "train_amp"),
        "train_nonamp": choose(args.train_nonamp, "train_nonamp"),
        "val_amp": choose(args.val_amp, "val_amp"),
        "val_nonamp": choose(args.val_nonamp, "val_nonamp"),
        "test_amp": choose(args.test_amp, "test_amp"),
        "test_nonamp": choose(args.test_nonamp, "test_nonamp"),
    }

    for key in ("train_amp", "train_nonamp", "test_amp", "test_nonamp"):
        if paths[key] is None:
            raise FileNotFoundError(
                f"Missing {key}. Supply the corresponding command-line path."
            )

    if (paths["val_amp"] is None) != (paths["val_nonamp"] is None):
        raise ValueError(
            "Validation input must include both AMP and non-AMP FASTA files."
        )

    raw_train = pd.concat(
        [
            read_fasta(paths["train_amp"], "AMP"),
            read_fasta(paths["train_nonamp"], "non-AMP"),
        ],
        ignore_index=True,
    )
    raw_val = None
    if paths["val_amp"] is not None:
        raw_val = pd.concat(
            [
                read_fasta(paths["val_amp"], "AMP"),
                read_fasta(paths["val_nonamp"], "non-AMP"),
            ],
            ignore_index=True,
        )
    raw_test = pd.concat(
        [
            read_fasta(paths["test_amp"], "AMP"),
            read_fasta(paths["test_nonamp"], "non-AMP"),
        ],
        ignore_index=True,
    )

    def prep(frame: pd.DataFrame, name: str) -> pd.DataFrame:
        return prepare_clean_frame(
            frame,
            split_name=name,
            label_column="label",
            sequence_column="sequence",
            id_column="id",
        )

    train = prep(raw_train, "train")
    val = prep(raw_val, "val") if raw_val is not None else None
    test = prep(raw_test, "test")
    sources = {
        key: str(path)
        for key, path in paths.items()
        if path is not None
    }
    return train, val, test, sources


def create_validation(
    clean_train: pd.DataFrame,
    *,
    val_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"--val-ratio must be in (0,1), got {val_ratio}")

    rng = np.random.default_rng(seed)
    val_indices: List[int] = []

    for label in ("AMP", "non-AMP"):
        indices = clean_train.index[
            clean_train["clean_label"] == label
        ].to_numpy()

        if len(indices) < 2:
            raise ValueError(f"Not enough {label} samples for a validation split.")

        order = rng.permutation(indices)
        n_val = int(round(len(order) * val_ratio))
        n_val = max(1, min(n_val, len(order) - 1))
        val_indices.extend(order[:n_val].tolist())

    val_set = set(int(index) for index in val_indices)
    train_indices = [
        int(index)
        for index in clean_train.index
        if int(index) not in val_set
    ]

    train = clean_train.loc[sorted(train_indices)].copy().reset_index(drop=True)
    val = clean_train.loc[sorted(val_set)].copy().reset_index(drop=True)

    train["source_row_index"] = np.arange(len(train), dtype=int)
    train["split"] = "train"
    val["split"] = "val"
    val["target"] = val["clean_target"].astype(int)
    val["noisy_label"] = val["clean_label"]
    val["is_noisy"] = 0

    return train, val


def check_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    skip_duplicates: bool,
) -> Dict[str, Dict[str, int]]:
    result: Dict[str, Dict[str, int]] = {}

    for name, frame in (("train", train), ("val", val), ("test", test)):
        counts = (
            frame["clean_label"]
            .value_counts()
            .reindex(["AMP", "non-AMP"], fill_value=0)
            .astype(int)
            .to_dict()
        )
        result[name] = {
            "total": int(len(frame)),
            "AMP": int(counts["AMP"]),
            "non-AMP": int(counts["non-AMP"]),
        }

        if not skip_duplicates and frame["sequence"].duplicated().any():
            raise ValueError(f"{name}: duplicate sequences detected.")

    if result["train"]["AMP"] != result["train"]["non-AMP"]:
        raise ValueError(
            "The training split must be class-balanced. "
            f"Observed {result['train']}."
        )

    if not skip_duplicates:
        combined = pd.concat(
            [
                train[["sequence"]],
                val[["sequence"]],
                test[["sequence"]],
            ],
            ignore_index=True,
        )
        if combined["sequence"].duplicated().any():
            raise ValueError("Cross-split duplicate sequences detected.")

    return result


def validate_rates(values: Sequence[float]) -> List[float]:
    rates = sorted(set(float(value) for value in values))
    if not rates:
        raise ValueError("No noise rates supplied.")
    if any(rate < 0.0 or rate > 1.0 for rate in rates):
        raise ValueError(f"Noise rates must be within [0,1]: {rates}")
    return rates


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def prepare_output(path: Path, overwrite: bool) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {path}\n"
                "Add --overwrite only when replacing all generated data."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> None:
    args = parse_args()
    input_format = resolve_input_format(args)

    if input_format == "csv":
        original_train, clean_val, clean_test, sources = load_csv(args)
    else:
        original_train, clean_val, clean_test, sources = load_fasta(args)

    val_created = clean_val is None
    if clean_val is None:
        if not args.create_val_if_missing:
            raise FileNotFoundError(
                "Validation files are absent and validation creation is disabled."
            )
        clean_train, clean_val = create_validation(
            original_train,
            val_ratio=args.val_ratio,
            seed=args.val_seed,
        )
    else:
        clean_train = original_train

    counts = check_splits(
        clean_train,
        clean_val,
        clean_test,
        args.skip_duplicate_check,
    )
    rates = validate_rates(args.noise_rates)
    output = prepare_output(args.output_dir, args.overwrite)

    source_hashes = {
        key: file_sha256(Path(path))
        for key, path in sources.items()
    }

    manifest: Dict[str, object] = {
        "dataset": args.dataset,
        "input_format": input_format,
        "source_files": sources,
        "source_sha256": source_hashes,
        "validation_source": (
            "created_from_clean_train" if val_created else "provided"
        ),
        "validation_ratio": float(args.val_ratio) if val_created else None,
        "validation_seed": int(args.val_seed) if val_created else None,
        "original_clean_train_size": int(len(original_train)),
        "final_clean_train_size": counts["train"]["total"],
        "validation_size": counts["val"]["total"],
        "test_size": counts["test"]["total"],
        "split_counts": counts,
        "noise_rates": rates,
        "repetition_seeds": [int(seed) for seed in args.seeds],
    }
    write_json(output / "source_manifest.json", manifest)

    summary: List[Dict[str, object]] = []

    for rep, seed in enumerate(args.seeds, start=1):
        rng = np.random.default_rng(seed)

        permutations = {
            label: rng.permutation(
                clean_train.index[
                    clean_train["clean_label"] == label
                ].to_numpy()
            )
            for label in ("AMP", "non-AMP")
        }

        previous: set[int] = set()

        for rate in rates:
            selected: List[int] = []
            for label in ("AMP", "non-AMP"):
                order = permutations[label]
                n_flip = int(round(len(order) * rate))
                selected.extend(order[:n_flip].tolist())

            selected_set = set(int(index) for index in selected)
            if not previous.issubset(selected_set):
                raise RuntimeError(
                    f"Nested-noise check failed: rep={rep}, rate={rate}"
                )
            previous = selected_set

            train_out = clean_train.copy()
            selected_array = np.asarray(selected, dtype=int)

            if selected_array.size:
                new_targets = (
                    1
                    - train_out.loc[selected_array, "clean_target"].astype(int)
                )
                train_out.loc[selected_array, "target"] = new_targets.to_numpy()
                train_out.loc[selected_array, "noisy_label"] = (
                    new_targets.map(TARGET_TO_LABEL).to_numpy()
                )
                train_out.loc[selected_array, "is_noisy"] = 1

            train_out["target"] = train_out["target"].astype(int)
            train_out["is_noisy"] = train_out["is_noisy"].astype(int)

            amp_to_nonamp = int(
                (
                    (train_out["clean_label"] == "AMP")
                    & (train_out["noisy_label"] == "non-AMP")
                ).sum()
            )
            nonamp_to_amp = int(
                (
                    (train_out["clean_label"] == "non-AMP")
                    & (train_out["noisy_label"] == "AMP")
                ).sum()
            )
            if amp_to_nonamp != nonamp_to_amp:
                raise RuntimeError("Class-balanced flip check failed.")

            number_noisy = int(train_out["is_noisy"].sum())
            expected = (
                int(round(counts["train"]["AMP"] * rate)) * 2
            )
            if number_noisy != expected:
                raise RuntimeError(
                    f"Noise count mismatch: {number_noisy} != {expected}"
                )

            run_dir = output / f"noise_{rate:.3f}" / f"rep{rep}"
            run_dir.mkdir(parents=True, exist_ok=False)

            noise_columns = [
                "source_row_index",
                "id",
                "sequence",
                "clean_label",
                "noisy_label",
                "clean_target",
                "target",
                "is_noisy",
            ]
            train_out.loc[
                train_out["is_noisy"] == 1,
                noise_columns,
            ].to_csv(run_dir / "noise_indices.csv", index=False)

            train_out.to_csv(run_dir / "train.csv", index=False)
            clean_val.to_csv(run_dir / "val.csv", index=False)
            clean_test.to_csv(run_dir / "test.csv", index=False)

            metadata: Dict[str, object] = {
                "dataset": args.dataset,
                "noise_type": "class-balanced symmetric label noise",
                "noise_rate": float(rate),
                "rep": int(rep),
                "seed": int(seed),
                "train_size": counts["train"]["total"],
                "val_size": counts["val"]["total"],
                "test_size": counts["test"]["total"],
                "number_noisy": number_noisy,
                "amp_to_nonamp": amp_to_nonamp,
                "nonamp_to_amp": nonamp_to_amp,
                "validation_labels_clean": True,
                "test_labels_clean": True,
                "validation_source": manifest["validation_source"],
                "validation_ratio": manifest["validation_ratio"],
                "validation_seed": manifest["validation_seed"],
                "nested_noise_within_rep": True,
                "model_training_label_column": "target",
                "clean_audit_label_column": "clean_target",
            }
            write_json(run_dir / "metadata.json", metadata)
            summary.append(metadata)

            print(
                f"noise_{rate:.3f}/rep{rep}: "
                f"seed={seed}, noisy={number_noisy}, "
                f"AMP->non-AMP={amp_to_nonamp}, "
                f"non-AMP->AMP={nonamp_to_amp}"
            )

    pd.DataFrame(summary).to_csv(
        output / "noise_summary.csv",
        index=False,
    )

    print("=" * 100)
    print(f"Dataset: {args.dataset}")
    print(f"Input format: {input_format}")
    print(f"Validation source: {manifest['validation_source']}")
    print(
        f"Clean split sizes: train={counts['train']['total']}, "
        f"val={counts['val']['total']}, "
        f"test={counts['test']['total']}"
    )
    print(f"Generated noise data: {output}")
    print(f"Summary: {output / 'noise_summary.csv'}")
    print("=" * 100)


if __name__ == "__main__":
    main()
