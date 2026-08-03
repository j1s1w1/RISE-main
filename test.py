#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate a trained RISE checkpoint on the predefined test split."""

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
from features.sequence_features import extract_token_embeddings, safe_torch_load
from features.structure_features import extract_graphs
from train import (
    DualModalTMNR2,
    _load_yaml,
    assert_alignment,
    evaluate,
    resolve_experiment_paths,
)


def load_run_config(run_dir: Path) -> Dict[str, Any]:
    config_path = run_dir / "run_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Training configuration not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if "arguments" not in config:
        raise ValueError(f"Invalid run configuration: {config_path}")
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate RISE on a predefined test split."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--paths", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--noise", required=True, type=float)
    parser.add_argument("--rep", required=True, type=int)

    parser.add_argument("--run_dir", default=None)
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
    parser.add_argument("--index_pdb_by_seq", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--allow_missing_pdb", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--log_every_extract", type=int, default=None)
    return parser


def build_test_namespace(
    saved: Mapping[str, Any],
    cli: argparse.Namespace,
    resolved_paths: Mapping[str, str],
) -> SimpleNamespace:
    values = dict(saved["arguments"])
    overrides = {
        "device": cli.device or resolved_paths.get("device") or values.get("device", "cuda:0"),
        "esm_model_path": cli.esm_model_path or resolved_paths.get("esm_model_path") or values.get("esm_model_path"),
        "pdb_dir": cli.pdb_dir or resolved_paths.get("pdb_dir") or values.get("pdb_dir"),
        "pdb_map_csv": cli.pdb_map_csv if cli.pdb_map_csv is not None else values.get("pdb_map_csv"),
        "esm_batch_size": cli.esm_batch_size if cli.esm_batch_size is not None else values.get("esm_batch_size", 8),
        "eval_batch_size": cli.eval_batch_size if cli.eval_batch_size is not None else values.get("eval_batch_size", 64),
        "num_workers": cli.num_workers if cli.num_workers is not None else values.get("num_workers", 0),
        "index_pdb_by_seq": cli.index_pdb_by_seq if cli.index_pdb_by_seq is not None else values.get("index_pdb_by_seq", False),
        "allow_missing_pdb": cli.allow_missing_pdb if cli.allow_missing_pdb is not None else values.get("allow_missing_pdb", False),
        "log_every_extract": cli.log_every_extract if cli.log_every_extract is not None else values.get("log_every_extract", 500),
    }
    values.update(overrides)
    missing = [name for name in ("esm_model_path", "pdb_dir") if not values.get(name)]
    if missing:
        raise ValueError(f"Missing test runtime path(s): {', '.join(missing)}")
    return SimpleNamespace(**values)


def main() -> None:
    cli = build_parser().parse_args()
    experiment_config = _load_yaml(cli.config)
    paths_config = _load_yaml(cli.paths)
    resolved = resolve_experiment_paths(
        experiment_config, paths_config, cli.dataset, cli.noise, cli.rep
    )

    run_dir = Path(cli.run_dir or resolved["result_dir"]).expanduser().resolve()
    saved = load_run_config(run_dir)
    args = build_test_namespace(saved, cli, resolved)

    test_csv = cli.test_csv or resolved.get("test_csv")
    test_amp = cli.test_amp or resolved.get("test_amp")
    test_nonamp = cli.test_nonamp or resolved.get("test_nonamp")

    if test_csv:
        test_csv_path = Path(test_csv).expanduser()
        if not test_csv_path.is_file():
            raise FileNotFoundError(f"test_csv not found: {test_csv_path}")
        test_csv = str(test_csv_path.resolve())
        input_format = "csv"
    elif test_amp and test_nonamp:
        for name, value in (("test_amp", test_amp), ("test_nonamp", test_nonamp)):
            if not Path(value).expanduser().is_file():
                raise FileNotFoundError(f"{name} not found: {value}")
        test_amp = str(Path(test_amp).expanduser().resolve())
        test_nonamp = str(Path(test_nonamp).expanduser().resolve())
        input_format = "fasta"
    else:
        raise ValueError(
            "Configure test_csv, or both test_amp and test_nonamp."
        )

    result_dir = Path(cli.result_dir).expanduser().resolve() if cli.result_dir else run_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = (
        Path(cli.cache_dir).expanduser().resolve()
        if cli.cache_dir
        else Path(resolved.get("cache_dir", result_dir / "test_cache")).expanduser().resolve() / "test"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    if input_format == "csv":
        records = load_csv_split(
            test_csv,
            label_column="clean_target",
            expected_split="test",
        )
    else:
        records = load_binary_split(test_amp, test_nonamp)
    print(f"✅ predefined {input_format} test records: {len(records)}")
    sequence_data = extract_token_embeddings(
        records=records,
        esm_model_path=args.esm_model_path,
        cache_fp=str(cache_dir / "test_esm2_token.pt"),
        device=args.device,
        esm_batch_size=args.esm_batch_size,
    )
    graph_data = extract_graphs(
        records, str(cache_dir / "test_struct_graph.pt"), args, "test"
    )
    assert_alignment(sequence_data, graph_data, "test")

    sequence_input_dim = int(sequence_data["embeddings"][0].shape[1])
    graph_input_dim = int(graph_data["graphs"][0]["x"].shape[1])
    if sequence_input_dim != int(saved["sequence_input_dim"]):
        raise RuntimeError("Sequence feature dimension differs from the checkpoint.")
    if graph_input_dim != int(saved["graph_input_dim"]):
        raise RuntimeError("Graph feature dimension differs from the checkpoint.")

    device = torch.device(args.device)
    model = DualModalTMNR2(sequence_input_dim, graph_input_dim, args).to(device)
    checkpoint_path = run_dir / "best_model.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model.load_state_dict(safe_torch_load(checkpoint_path, args.device), strict=True)

    metrics, rows = evaluate(model, sequence_data, graph_data, args)
    metrics.update({
        "evaluation_split": "test",
        "validation_mode": "predefined",
        "model": "RISE",
        "algorithm": "EC-RML-Safe",
        "dataset": resolved["dataset_name"],
        "noise": float(cli.noise),
        "rep": int(cli.rep),
        "checkpoint": str(checkpoint_path),
        "input_format": input_format,
        "test_csv": test_csv if input_format == "csv" else None,
        "test_amp": test_amp if input_format == "fasta" else None,
        "test_nonamp": test_nonamp if input_format == "fasta" else None,
    })

    metrics_path = result_dir / "test_metrics.json"
    predictions_path = result_dir / "test_predictions.csv"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "seq", "true_label", "prob_amp", "pred_label"])
        writer.writeheader()
        writer.writerows(rows)

    print("\n📊 Test metrics")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"✅ saved metrics: {metrics_path}")
    print(f"✅ saved predictions: {predictions_path}")


if __name__ == "__main__":
    main()
