#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate a trained RISE checkpoint on a predefined clean test split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping

import torch

from data.fasta import load_binary_split
from data.csv_dataset import load_csv_split
from features.sequence_features import extract_token_embeddings
from features.structure_features import extract_graphs

from train import (
    RISEModel,
    _load_yaml,
    assert_alignment,
    evaluate,
    resolve_experiment_paths,
)


def load_run_config(run_dir: Path) -> Dict[str, Any]:
    """Load the exact training configuration saved with the checkpoint."""
    config_path = run_dir / "run_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Training configuration not found: {config_path}"
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if "arguments" not in config:
        raise ValueError(
            f"Invalid run configuration: missing 'arguments': {config_path}"
        )

    return config


def safe_torch_load(path: Path, device: str):
    """Load a state dictionary across PyTorch versions."""
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(path, map_location=device)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate RISE on a predefined clean test split."
    )

    parser.add_argument("--config", required=True)
    parser.add_argument("--paths", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--noise", required=True, type=float)
    parser.add_argument("--rep", required=True, type=int)

    parser.add_argument(
        "--run_dir",
        default=None,
        help="Training run directory containing run_config.json and best_model.pt.",
    )

    parser.add_argument("--test_csv", default=None)
    parser.add_argument("--test_amp", default=None)
    parser.add_argument("--test_nonamp", default=None)

    parser.add_argument("--result_dir", default=None)
    parser.add_argument("--cache_dir", default=None)

    parser.add_argument("--device", default=None)
    parser.add_argument("--esm_model_path", default=None)
    parser.add_argument("--esm_batch_size", type=int, default=None)

    parser.add_argument("--pdb_dir", default=None)
    parser.add_argument("--pdb_map_csv", default=None)

    parser.add_argument(
        "--index_pdb_by_seq",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--allow_missing_pdb",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--log_every_extract", type=int, default=None)

    return parser


def build_test_namespace(
    saved: Mapping[str, Any],
    cli: argparse.Namespace,
    resolved_paths: Mapping[str, str],
) -> SimpleNamespace:
    """Reuse training-time model/feature settings for deterministic evaluation."""
    values = dict(saved["arguments"])

    runtime_overrides = {
        "device": (
            cli.device
            or resolved_paths.get("device")
            or values.get("device")
            or "cuda:0"
        ),
        "esm_model_path": (
            cli.esm_model_path
            or resolved_paths.get("esm_model_path")
            or values.get("esm_model_path")
        ),
        "pdb_dir": (
            cli.pdb_dir
            or resolved_paths.get("pdb_dir")
            or values.get("pdb_dir")
        ),
        "pdb_map_csv": (
            cli.pdb_map_csv
            if cli.pdb_map_csv is not None
            else values.get("pdb_map_csv")
        ),
        "esm_batch_size": (
            cli.esm_batch_size
            if cli.esm_batch_size is not None
            else values.get("esm_batch_size", 8)
        ),
        "eval_batch_size": (
            cli.eval_batch_size
            if cli.eval_batch_size is not None
            else values.get("eval_batch_size", 64)
        ),
        "num_workers": (
            cli.num_workers
            if cli.num_workers is not None
            else values.get("num_workers", 0)
        ),
        "index_pdb_by_seq": (
            cli.index_pdb_by_seq
            if cli.index_pdb_by_seq is not None
            else values.get("index_pdb_by_seq", False)
        ),
        "allow_missing_pdb": (
            cli.allow_missing_pdb
            if cli.allow_missing_pdb is not None
            else values.get("allow_missing_pdb", False)
        ),
        "log_every_extract": (
            cli.log_every_extract
            if cli.log_every_extract is not None
            else values.get("log_every_extract", 500)
        ),
    }

    values.update(runtime_overrides)

    required = ("esm_model_path", "pdb_dir")
    missing = [name for name in required if not values.get(name)]
    if missing:
        raise ValueError(
            "Missing test runtime path(s): " + ", ".join(missing)
        )

    return SimpleNamespace(**values)


def resolve_test_inputs(
    cli: argparse.Namespace,
    resolved_paths: Mapping[str, str],
):
    """Resolve the predefined clean test split."""
    test_csv = cli.test_csv or resolved_paths.get("test_csv")
    test_amp = cli.test_amp or resolved_paths.get("test_amp")
    test_nonamp = cli.test_nonamp or resolved_paths.get("test_nonamp")

    if test_csv:
        path = Path(test_csv).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"test_csv not found: {path}")
        return "csv", str(path), None, None

    if test_amp and test_nonamp:
        amp_path = Path(test_amp).expanduser().resolve()
        nonamp_path = Path(test_nonamp).expanduser().resolve()

        if not amp_path.is_file():
            raise FileNotFoundError(f"test_amp not found: {amp_path}")
        if not nonamp_path.is_file():
            raise FileNotFoundError(
                f"test_nonamp not found: {nonamp_path}"
            )

        return "fasta", None, str(amp_path), str(nonamp_path)

    raise ValueError(
        "Configure test_csv, or both test_amp and test_nonamp."
    )


def main() -> None:
    cli = build_parser().parse_args()

    experiment_config = _load_yaml(cli.config)
    paths_config = _load_yaml(cli.paths)

    resolved_paths = resolve_experiment_paths(
        experiment_config,
        paths_config,
        cli.dataset,
        cli.noise,
        cli.rep,
    )

    # By default, evaluate the checkpoint produced by the corresponding
    # training run.
    run_dir = Path(
        cli.run_dir or resolved_paths["result_dir"]
    ).expanduser().resolve()

    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"Training run directory not found: {run_dir}"
        )

    saved = load_run_config(run_dir)
    args = build_test_namespace(
        saved,
        cli,
        resolved_paths,
    )

    input_format, test_csv, test_amp, test_nonamp = resolve_test_inputs(
        cli,
        resolved_paths,
    )

    # Test features use a separate cache so evaluation never overwrites
    # training/validation feature caches.
    if cli.cache_dir:
        cache_dir = Path(cli.cache_dir).expanduser().resolve()
    else:
        configured_cache = resolved_paths.get("cache_dir")
        if configured_cache:
            cache_dir = (
                Path(configured_cache).expanduser().resolve() / "test"
            )
        else:
            cache_dir = run_dir / "test_cache"

    cache_dir.mkdir(parents=True, exist_ok=True)

    if cli.result_dir:
        result_dir = Path(cli.result_dir).expanduser().resolve()
    else:
        result_dir = run_dir

    result_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load the clean test labels.
    # ------------------------------------------------------------------
    if input_format == "csv":
        records = load_csv_split(
            test_csv,
            label_column="clean_target",
            expected_split="test",
        )
    else:
        records = load_binary_split(
            test_amp,
            test_nonamp,
        )

    print(
        f"Predefined clean {input_format} test records: "
        f"{len(records)}"
    )

    # ------------------------------------------------------------------
    # Feature extraction.
    # ------------------------------------------------------------------
    sequence_data = extract_token_embeddings(
        records=records,
        esm_model_path=args.esm_model_path,
        cache_fp=str(cache_dir / "test_esm2_token.pt"),
        device=args.device,
        esm_batch_size=args.esm_batch_size,
    )

    graph_data = extract_graphs(
        records,
        str(cache_dir / "test_struct_graph.pt"),
        args,
        "test",
    )

    assert_alignment(
        sequence_data,
        graph_data,
        "test",
    )

    sequence_input_dim = int(
        sequence_data["embeddings"][0].shape[1]
    )
    graph_input_dim = int(
        graph_data["graphs"][0]["x"].shape[1]
    )

    saved_sequence_dim = int(
        saved["sequence_input_dim"]
    )
    saved_graph_dim = int(
        saved["graph_input_dim"]
    )

    if sequence_input_dim != saved_sequence_dim:
        raise RuntimeError(
            "Sequence feature dimension differs from the training "
            f"checkpoint: test={sequence_input_dim}, "
            f"checkpoint={saved_sequence_dim}."
        )

    if graph_input_dim != saved_graph_dim:
        raise RuntimeError(
            "Graph feature dimension differs from the training "
            f"checkpoint: test={graph_input_dim}, "
            f"checkpoint={saved_graph_dim}."
        )

    # ------------------------------------------------------------------
    # Rebuild exactly the same prediction backbone used in training.
    # Supervision refinement is not called during testing.
    # ------------------------------------------------------------------
    device = torch.device(args.device)

    model = RISEModel(
        sequence_input_dim,
        graph_input_dim,
        args,
    ).to(device)

    checkpoint_path = run_dir / "best_model.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    state_dict = safe_torch_load(
        checkpoint_path,
        args.device,
    )

    if not isinstance(state_dict, Mapping):
        raise RuntimeError(
            f"Unexpected checkpoint format: {checkpoint_path}"
        )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    # evaluate() uses the untransformed fused evidence:
    # sequence evidence + structural evidence -> fused prediction.
    # No candidate identification, reliability calibration, or
    # supervision refinement is performed here.
    metrics, prediction_rows = evaluate(
        model,
        sequence_data,
        graph_data,
        args,
    )

    metrics.update(
        {
            "evaluation_split": "test",
            "validation_mode": "predefined",
            "model": "RISE",
            "dataset": resolved_paths["dataset_name"],
            "noise": float(cli.noise),
            "rep": int(cli.rep),
            "checkpoint": str(checkpoint_path),
            "input_format": input_format,
            "test_csv": (
                test_csv
                if input_format == "csv"
                else None
            ),
            "test_amp": (
                test_amp
                if input_format == "fasta"
                else None
            ),
            "test_nonamp": (
                test_nonamp
                if input_format == "fasta"
                else None
            ),
        }
    )

    metrics_path = result_dir / "test_metrics.json"
    predictions_path = result_dir / "test_predictions.csv"

    metrics_path.write_text(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    with predictions_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "seq",
                "true_label",
                "prob_amp",
                "pred_label",
            ],
        )
        writer.writeheader()
        writer.writerows(prediction_rows)

    print()
    print("=" * 72)
    print("RISE Test Results")
    print("=" * 72)
    print(
        f"Dataset     : {resolved_paths['dataset_name']}"
    )
    print(
        f"Noise       : {float(cli.noise):.3f}"
    )
    print(
        f"Repetition  : {int(cli.rep)}"
    )
    print(
        f"Test samples: {len(records)}"
    )
    print()
    print(
        f"ACC = {metrics['Accuracy']:.4f}"
    )
    print(
        f"AUC = {metrics['AUC']:.4f}"
    )
    print(
        f"MCC = {metrics['MCC']:.4f}"
    )
    print()
    print(f"Saved metrics     : {metrics_path}")
    print(f"Saved predictions : {predictions_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
