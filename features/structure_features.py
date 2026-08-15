#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ESMFold PDB parsing and residue-graph construction for RISE.

This module performs structure-side preprocessing only. For each peptide, it
converts a precomputed ESMFold PDB file into a residue graph containing node
features and ``edge_index``.

The first node-feature column is always normalized residue-level pLDDT unless
``disable_plddt_feature`` is explicitly enabled. RISE uses this column as the
observable residue-level structural-confidence profile when estimating
sample-level structural reliability.

The trainable structural encoder and reliability-conditioned supervision
refinement are implemented in ``train.py``.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import torch

if __package__:
    from data.fasta import load_binary_split, normalize_sequence
else:
    import sys
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from data.fasta import load_binary_split, normalize_sequence


PathLike = Union[str, os.PathLike[str]]

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O", "ASX": "B", "GLX": "Z", "UNK": "X",
}
AA_ORDER = list("ACDEFGHIKLMNPQRSTVWYXBZUO")
AA_TO_IDX = {aa: index for index, aa in enumerate(AA_ORDER)}
norm_seq = normalize_sequence


def sanitize_id(value: Any) -> str:
    """Convert a record identifier to a filesystem-safe form."""
    return re.sub(r"[^\w.\-]+", "_", str(value).strip())


def safe_torch_load(
    fp: PathLike,
    device: Union[str, torch.device] = "cpu",
) -> Any:
    """Load a cached feature object across supported PyTorch versions."""
    try:
        return torch.load(
            fp,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(fp, map_location=device)


def parse_pdb_ca(pdb_fp: PathLike):
    """Parse unique C-alpha atoms, coordinates and B-factor pLDDT values."""
    residues: List[str] = []
    coordinates: List[List[float]] = []
    plddts: List[float] = []
    seen = set()

    path = Path(pdb_fp)
    if not path.is_file():
        raise FileNotFoundError(f"PDB file not found: {path}")

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            if line[12:16].strip() != "CA":
                continue

            residue_name = line[17:20].strip().upper()
            chain = line[21].strip()
            residue_number = line[22:26].strip()
            insertion_code = line[26].strip()
            residue_key = (chain, residue_number, insertion_code)

            if residue_key in seen:
                continue
            seen.add(residue_key)

            try:
                xyz = [
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ]
            except (TypeError, ValueError):
                continue

            try:
                b_factor = float(line[60:66])
            except (TypeError, ValueError):
                b_factor = 0.0

            residues.append(AA3_TO_1.get(residue_name, "X"))
            coordinates.append(xyz)
            plddts.append(b_factor)

    if not coordinates:
        raise ValueError(f"No CA atoms found: {path}")

    return (
        residues,
        np.asarray(coordinates, dtype=np.float32),
        np.asarray(plddts, dtype=np.float32),
    )


def normalize_plddt(plddt: Any, scale: str = "auto") -> np.ndarray:
    """Normalize pLDDT values from either [0,100] or [0,1] into [0,1]."""
    values = np.asarray(plddt, dtype=np.float32)

    if values.size == 0:
        return values.astype(np.float32)

    if scale == "auto":
        output = (
            np.clip(values, 0, 100) / 100.0
            if np.nanmax(values) > 1.5
            else np.clip(values, 0, 1)
        )
    elif scale == "100":
        output = np.clip(values, 0, 100) / 100.0
    elif scale == "1":
        output = np.clip(values, 0, 1)
    else:
        raise ValueError(f"Unsupported plddt_scale: {scale}")

    return output.astype(np.float32)


def build_pdb_index(
    pdb_dir: PathLike,
    use_seq_index: bool = True,
) -> Dict[str, Dict[str, str]]:
    """Index PDB/ENT files by filename stem and optionally by sequence."""
    directory = Path(pdb_dir).expanduser()

    if not directory.is_dir():
        raise FileNotFoundError(f"PDB dir not found: {directory}")

    file_paths: List[str] = []
    for extension in ("*.pdb", "*.ent"):
        file_paths.extend(
            glob.glob(str(directory / "**" / extension), recursive=True)
        )

    file_paths = sorted(file_paths)
    by_id: Dict[str, str] = {}
    by_sequence: Dict[str, str] = {}

    print(f"Indexing PDB files: {len(file_paths)} from {directory}")

    for index, file_path in enumerate(file_paths):
        stem = Path(file_path).stem
        by_id.setdefault(stem, file_path)
        by_id.setdefault(sanitize_id(stem), file_path)

        if use_seq_index:
            try:
                residues, _, _ = parse_pdb_ca(file_path)
                sequence = normalize_sequence("".join(residues))
                if sequence:
                    by_sequence.setdefault(sequence, file_path)
            except Exception:
                pass

        if (index + 1) % 2000 == 0:
            print(f"  indexed {index + 1}/{len(file_paths)} PDB files")

    print(
        "PDB index ready | "
        f"by_id={len(by_id)} | by_sequence={len(by_sequence)}"
    )

    return {"by_id": by_id, "by_seq": by_sequence}


def read_pdb_map_csv(fp: Optional[PathLike]) -> Dict[str, str]:
    """Read an optional explicit ID/sequence-to-PDB mapping CSV."""
    if not fp:
        return {}

    path = Path(fp).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"PDB map CSV not found: {path}")

    mapping: Dict[str, str] = {}

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return mapping

        lower_to_original = {
            column.lower().strip(): column
            for column in reader.fieldnames
        }

        id_column = lower_to_original.get("id")
        sequence_column = (
            lower_to_original.get("seq")
            or lower_to_original.get("sequence")
        )
        path_column = (
            lower_to_original.get("pdb_path")
            or lower_to_original.get("path")
            or lower_to_original.get("pdb")
        )

        if path_column is None:
            raise ValueError(
                "pdb_map_csv must contain a pdb_path/path/pdb column"
            )

        for row in reader:
            pdb_path = str(row.get(path_column, "")).strip()
            if not pdb_path:
                continue

            if id_column and row.get(id_column):
                mapping[
                    "id:" + str(row[id_column]).strip()
                ] = pdb_path

            if sequence_column and row.get(sequence_column):
                mapping[
                    "seq:" + normalize_sequence(row[sequence_column])
                ] = pdb_path

    return mapping


def find_pdb(
    record: Mapping[str, Any],
    idx: int,
    pdb_index: Mapping[str, Mapping[str, str]],
    pdb_map: Mapping[str, str],
) -> Optional[str]:
    """Resolve the structure file corresponding to one peptide record."""
    record_id = str(record["id"]).strip()
    sequence = normalize_sequence(record["seq"])

    for key in (f"id:{record_id}", f"seq:{sequence}"):
        mapped = pdb_map.get(key)
        if mapped and os.path.exists(mapped):
            return mapped

    by_id = pdb_index.get("by_id", {})
    by_sequence = pdb_index.get("by_seq", {})

    candidates = [
        record_id,
        sanitize_id(record_id),
        str(idx),
        f"{idx:05d}",
        f"seq_{idx}",
        f"sample_{idx}",
    ]

    for candidate in candidates:
        resolved = by_id.get(candidate)
        if resolved and os.path.exists(resolved):
            return resolved

    resolved = by_sequence.get(sequence)
    if resolved and os.path.exists(resolved):
        return resolved

    return None


def local_angle_cos(coords: np.ndarray) -> np.ndarray:
    """Calculate the local C-alpha backbone-angle cosine per residue."""
    residue_count = int(coords.shape[0])
    output = np.zeros(residue_count, dtype=np.float32)

    if residue_count < 3:
        return output

    for index in range(1, residue_count - 1):
        previous_vector = coords[index - 1] - coords[index]
        next_vector = coords[index + 1] - coords[index]
        denominator = (
            np.linalg.norm(previous_vector)
            * np.linalg.norm(next_vector)
            + 1e-8
        )
        output[index] = float(
            np.dot(previous_vector, next_vector) / denominator
        )

    output[0] = output[1]
    output[-1] = output[-2]
    return output


def graph_from_pdb(
    pdb_fp: PathLike,
    input_seq: str,
    args: Any,
) -> Dict[str, Any]:
    """Build one RISE residue graph from an ESMFold PDB file."""
    residues, coordinates, plddt = parse_pdb_ca(pdb_fp)
    residue_count = int(coordinates.shape[0])

    coordinate_difference = (
        coordinates[:, None, :] - coordinates[None, :, :]
    )
    distance = np.sqrt(
        (coordinate_difference ** 2).sum(axis=-1) + 1e-8
    ).astype(np.float32)

    # Spatial contact information.
    contact_threshold = float(args.contact_threshold)
    contact = (
        distance < contact_threshold
    ) & (~np.eye(residue_count, dtype=bool))
    contact_degree = contact.sum(axis=1).astype(np.float32)

    # Spatial nearest-neighbor information.
    graph_knn_k = int(args.graph_knn_k)
    effective_k = min(
        max(1, graph_knn_k),
        max(1, residue_count - 1),
    )

    all_neighbors: List[np.ndarray] = []
    distance_statistics: List[List[float]] = []

    for index in range(residue_count):
        order = np.argsort(distance[index])
        order = order[order != index]
        neighbors = (
            order[:effective_k]
            if len(order) > 0
            else np.array([index], dtype=np.int64)
        )
        all_neighbors.append(neighbors)

        neighbor_distance = distance[index, neighbors]
        distance_statistics.append(
            [
                float(neighbor_distance.mean()),
                float(neighbor_distance.min()),
                float(neighbor_distance.max()),
                float(neighbor_distance.std()),
            ]
        )

    distance_statistics_array = np.asarray(
        distance_statistics,
        dtype=np.float32,
    )

    # IMPORTANT:
    # x[:, 0] is normalized residue pLDDT. train.py reads this exact column
    # when computing the structural-confidence profile c_i.
    plddt_normalized_1d = normalize_plddt(
        plddt,
        args.plddt_scale,
    )
    plddt_normalized = (
        plddt_normalized_1d
        .reshape(-1, 1)
        .astype(np.float32)
    )

    relative_position = (
        np.arange(residue_count, dtype=np.float32)
        / max(1, residue_count - 1)
    ).reshape(-1, 1)

    degree_normalized = (
        contact_degree
        / max(1, residue_count - 1)
    ).reshape(-1, 1)

    distance_statistics_normalized = (
        distance_statistics_array / 20.0
    )
    angle = local_angle_cos(coordinates).reshape(-1, 1)

    if bool(args.disable_plddt_feature):
        plddt_feature = np.zeros_like(
            plddt_normalized,
            dtype=np.float32,
        )
    else:
        plddt_feature = plddt_normalized

    # Keep this feature order unchanged:
    # 0: normalized pLDDT
    # 1: relative residue position
    # 2: normalized contact degree
    # 3-6: spatial-neighbor distance statistics
    # 7: local C-alpha angle cosine
    node_feature_parts = [
        plddt_feature,
        relative_position,
        degree_normalized,
        distance_statistics_normalized,
        angle,
    ]

    if bool(args.use_aa_onehot):
        amino_acid_onehot = np.zeros(
            (residue_count, len(AA_ORDER)),
            dtype=np.float32,
        )
        normalized_input_sequence = normalize_sequence(input_seq)

        for index, amino_acid in enumerate(residues):
            resolved_amino_acid = amino_acid

            if amino_acid == "X":
                resolved_amino_acid = (
                    normalized_input_sequence[index]
                    if index < len(normalized_input_sequence)
                    else "X"
                )

            amino_acid_onehot[
                index,
                AA_TO_IDX.get(
                    resolved_amino_acid,
                    AA_TO_IDX["X"],
                ),
            ] = 1.0

        node_feature_parts.append(amino_acid_onehot)

    node_features = np.concatenate(
        node_feature_parts,
        axis=1,
    ).astype(np.float32)

    # Edge construction:
    # 1. sequential peptide-backbone edges;
    # 2. spatial-contact edges;
    # 3. spatial k-nearest-neighbor edges.
    edges = set()

    def add_edge(source: int, target: int) -> None:
        if source != target:
            edges.add((int(source), int(target)))

    # Sequential edges.
    for index in range(residue_count - 1):
        add_edge(index, index + 1)
        add_edge(index + 1, index)

    # Spatial contact edges.
    contact_sources, contact_targets = np.where(contact)
    for source, target in zip(
        contact_sources.tolist(),
        contact_targets.tolist(),
    ):
        add_edge(source, target)

    # Spatial k-nearest-neighbor edges.
    for source, neighbors in enumerate(all_neighbors):
        for target in neighbors.tolist():
            add_edge(source, target)
            add_edge(target, source)

    edge_index = (
        np.asarray(sorted(edges), dtype=np.int64).T
        if edges
        else np.zeros((2, 0), dtype=np.int64)
    )

    return {
        "x": torch.tensor(
            node_features,
            dtype=torch.float32,
        ),
        "edge_index": torch.tensor(
            edge_index,
            dtype=torch.long,
        ),
        "pdb_path": str(pdb_fp),
        "plddt_mean": float(plddt_normalized.mean()),
        "plddt_min": float(plddt_normalized.min()),
        "length_struct": int(residue_count),
    }


build_residue_graph = graph_from_pdb


def fallback_graph(
    seq: str,
    args: Any,
) -> Dict[str, Any]:
    """Construct a low-confidence chain graph if missing structures are allowed."""
    sequence = normalize_sequence(seq)
    residue_count = max(1, len(sequence))
    node_features: List[List[float]] = []

    for index in range(residue_count):
        features = [
            0.0,
            index / max(1, residue_count - 1),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

        if bool(args.use_aa_onehot):
            onehot = [0.0] * len(AA_ORDER)
            amino_acid = (
                sequence[index]
                if index < len(sequence)
                else "X"
            )
            onehot[
                AA_TO_IDX.get(
                    amino_acid,
                    AA_TO_IDX["X"],
                )
            ] = 1.0
            features.extend(onehot)

        node_features.append(features)

    edges = []
    for index in range(residue_count - 1):
        edges.append((index, index + 1))
        edges.append((index + 1, index))

    edge_index = (
        torch.tensor(edges, dtype=torch.long).T
        if edges
        else torch.zeros((2, 0), dtype=torch.long)
    )

    return {
        "x": torch.tensor(
            node_features,
            dtype=torch.float32,
        ),
        "edge_index": edge_index,
        "pdb_path": "",
        "plddt_mean": 0.0,
        "plddt_min": 0.0,
        "length_struct": int(residue_count),
    }


def validate_structure_feature_cache(
    data: Mapping[str, Any],
    expected_records: Optional[
        Sequence[Mapping[str, Any]]
    ] = None,
) -> None:
    """Validate graph-cache fields, tensors and sample alignment."""
    required = {"ids", "seqs", "labels", "graphs"}
    missing = sorted(required.difference(data.keys()))

    if missing:
        raise ValueError(
            f"Structure feature cache is missing keys: {missing}"
        )

    sample_count = len(data["ids"])
    lengths = {
        "ids": sample_count,
        "seqs": len(data["seqs"]),
        "labels": len(data["labels"]),
        "graphs": len(data["graphs"]),
    }

    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"Inconsistent structure cache lengths: {lengths}"
        )

    for index, graph in enumerate(data["graphs"]):
        if not isinstance(graph, Mapping):
            raise ValueError(
                f"Graph {index} is not a mapping"
            )

        if "x" not in graph or "edge_index" not in graph:
            raise ValueError(
                f"Graph {index} lacks x or edge_index"
            )

        node_features = graph["x"]
        edge_index = graph["edge_index"]

        if (
            not isinstance(node_features, torch.Tensor)
            or node_features.ndim != 2
        ):
            raise ValueError(
                f"Graph {index} x must be a rank-2 tensor"
            )

        if node_features.shape[1] < 1:
            raise ValueError(
                f"Graph {index} has no node features"
            )

        if (
            not isinstance(edge_index, torch.Tensor)
            or edge_index.ndim != 2
        ):
            raise ValueError(
                f"Graph {index} edge_index must be rank 2"
            )

        if edge_index.shape[0] != 2:
            raise ValueError(
                f"Graph {index} edge_index must have shape [2, E]"
            )

    if expected_records is None:
        return

    if len(expected_records) != sample_count:
        raise ValueError(
            "Record/cache length mismatch: "
            f"{len(expected_records)} vs {sample_count}"
        )

    for index, record in enumerate(expected_records):
        if str(record["id"]) != str(data["ids"][index]):
            raise ValueError(
                f"Record/cache ID mismatch at index {index}"
            )

        if normalize_sequence(
            record["seq"]
        ) != normalize_sequence(
            data["seqs"][index]
        ):
            raise ValueError(
                f"Record/cache sequence mismatch at index {index}"
            )

        if int(record["label"]) != int(data["labels"][index]):
            raise ValueError(
                f"Record/cache label mismatch at index {index}"
            )


def extract_graphs(
    records: Sequence[Mapping[str, Any]],
    cache_fp: PathLike,
    args: Any,
    split_name: str,
) -> Dict[str, Any]:
    """Extract or load residue graphs for the RISE structure branch."""
    cache_path = Path(cache_fp).expanduser()

    if cache_path.is_file():
        print(f"Structure graph cache exists: {cache_path}")
        cached = safe_torch_load(cache_path, "cpu")
        validate_structure_feature_cache(cached, records)
        return cached

    if not records:
        raise ValueError(
            "Cannot extract structure features from an empty record list"
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    pdb_index = build_pdb_index(
        args.pdb_dir,
        use_seq_index=bool(args.index_pdb_by_seq),
    )
    pdb_map = read_pdb_map_csv(args.pdb_map_csv)

    ids: List[str] = []
    seqs: List[str] = []
    labels: List[int] = []
    graphs: List[Dict[str, Any]] = []
    missing: List[List[Any]] = []

    print(
        f"Extracting structure graphs for {split_name}: "
        f"{len(records)} samples"
    )

    for index, record in enumerate(records):
        pdb_path = find_pdb(
            record,
            index,
            pdb_index,
            pdb_map,
        )

        if pdb_path is None:
            missing.append(
                [
                    index,
                    record["id"],
                    record["seq"],
                    "pdb_not_found",
                ]
            )

            if not bool(args.allow_missing_pdb):
                raise FileNotFoundError(
                    "PDB not found for "
                    f"index={index}, id={record['id']}. "
                    "Enable sequence indexing or provide pdb_map_csv "
                    "if the PDB file name does not match the record ID."
                )

            graph = fallback_graph(
                str(record["seq"]),
                args,
            )
        else:
            try:
                graph = graph_from_pdb(
                    pdb_path,
                    str(record["seq"]),
                    args,
                )
            except Exception as exc:
                missing.append(
                    [
                        index,
                        record["id"],
                        record["seq"],
                        repr(exc),
                    ]
                )

                if not bool(args.allow_missing_pdb):
                    raise

                graph = fallback_graph(
                    str(record["seq"]),
                    args,
                )

        ids.append(str(record["id"]))
        seqs.append(str(record["seq"]))
        labels.append(int(record["label"]))
        graphs.append(graph)

        log_every = max(
            1,
            int(args.log_every_extract),
        )
        if len(graphs) % log_every == 0:
            print(
                f"  parsed {len(graphs)} / {len(records)} graphs"
            )

    if missing:
        missing_path = (
            cache_path.parent
            / f"missing_pdb_{split_name}.csv"
        )

        with missing_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["index", "id", "seq", "error"]
            )
            writer.writerows(missing)

        print(
            f"Missing/failed PDB records: {len(missing)} | "
            f"saved: {missing_path}"
        )

    data = {
        "ids": ids,
        "seqs": seqs,
        "labels": torch.tensor(
            labels,
            dtype=torch.long,
        ),
        "graphs": graphs,
    }

    validate_structure_feature_cache(data, records)
    torch.save(data, cache_path)

    print(
        f"Saved structure graph cache: {cache_path}"
    )

    return data


extract_structure_features = extract_graphs


def get_mean_graph_features(
    graph_data: Mapping[str, Any],
) -> np.ndarray:
    """Mean-pool node features into one descriptor per peptide.

    The resulting descriptors are used to construct the joint
    sequence-structure neighborhood for candidate identification.
    """
    graphs = graph_data.get("graphs")

    if graphs is None:
        raise KeyError(
            'graph_data does not contain the required "graphs" field.'
        )

    if len(graphs) == 0:
        raise ValueError(
            "Cannot calculate graph features from an empty graph list."
        )

    pooled_features = []

    for index, graph in enumerate(graphs):
        if not isinstance(graph, Mapping):
            raise TypeError(
                f"Graph {index} must be a mapping, "
                f"got {type(graph).__name__}."
            )

        node_features = graph.get("x")

        if not isinstance(node_features, torch.Tensor):
            raise TypeError(
                f'Graph {index} does not contain a valid tensor field "x".'
            )

        if node_features.ndim != 2:
            raise ValueError(
                f'Graph {index} field "x" must be two-dimensional, '
                f"got shape {tuple(node_features.shape)}."
            )

        if node_features.shape[0] == 0:
            raise ValueError(
                f"Graph {index} contains no residue nodes."
            )

        pooled = (
            node_features
            .detach()
            .cpu()
            .float()
            .mean(dim=0)
            .numpy()
        )
        pooled_features.append(pooled)

    return np.stack(
        pooled_features,
        axis=0,
    ).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert ESMFold PDB files into "
            "RISE residue-graph feature caches."
        )
    )

    parser.add_argument("--amp-fasta", required=True)
    parser.add_argument("--nonamp-fasta", required=True)
    parser.add_argument("--pdb-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split-name", default="data")
    parser.add_argument("--pdb-map-csv", default=None)
    parser.add_argument(
        "--index-pdb-by-seq",
        action="store_true",
    )
    parser.add_argument(
        "--allow-missing-pdb",
        action="store_true",
    )
    parser.add_argument(
        "--contact-threshold",
        type=float,
        default=8.0,
    )
    parser.add_argument(
        "--graph-knn-k",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--plddt-scale",
        choices=["auto", "100", "1"],
        default="auto",
    )
    parser.add_argument(
        "--disable-plddt-feature",
        action="store_true",
    )
    parser.add_argument(
        "--use-aa-onehot",
        action="store_true",
    )
    parser.add_argument(
        "--log-every-extract",
        type=int,
        default=500,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    records = load_binary_split(
        args.amp_fasta,
        args.nonamp_fasta,
    )

    extract_graphs(
        records=records,
        cache_fp=args.output,
        args=args,
        split_name=args.split_name,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "AA3_TO_1",
    "AA_ORDER",
    "AA_TO_IDX",
    "sanitize_id",
    "safe_torch_load",
    "parse_pdb_ca",
    "normalize_plddt",
    "build_pdb_index",
    "read_pdb_map_csv",
    "find_pdb",
    "local_angle_cos",
    "graph_from_pdb",
    "build_residue_graph",
    "fallback_graph",
    "validate_structure_feature_cache",
    "extract_graphs",
    "extract_structure_features",
    "get_mean_graph_features",
]
