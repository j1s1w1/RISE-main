#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CSV split loader for the RISE benchmark datasets."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

from data.fasta import norm_seq


def _binary_value(value: Any, *, column: str, row_number: int) -> int:
    text = str(value).strip()
    if text not in {"0", "1"}:
        raise ValueError(
            f"{column} must contain only 0/1 values; "
            f"got {value!r} at CSV row {row_number}."
        )
    return int(text)


def load_csv_split(
    csv_path: str,
    *,
    label_column: str,
    expected_split: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load one predefined RISE CSV split.

    Parameters
    ----------
    csv_path:
        Path to ``train.csv``, ``val.csv`` or ``test.csv``.
    label_column:
        Column used as the model label. Use ``target`` for training and
        ``clean_target`` for validation/testing.
    expected_split:
        Optional expected value of the CSV ``split`` column.

    Returns
    -------
    list of dict
        Records compatible with the existing sequence and structure feature
        extractors. Each record contains ``id``, ``seq``, ``label``,
        ``clean_target`` and ``is_noisy``.
    """
    path = Path(csv_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"CSV split not found: {path}")

    records: List[Dict[str, Any]] = []
    seen_ids = set()

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {path}")

        required = {"id", "sequence", label_column, "clean_target"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"{path} is missing required column(s): {sorted(missing)}"
            )

        for row_number, row in enumerate(reader, start=2):
            sample_id = str(row.get("id", "")).strip()
            sequence = norm_seq(str(row.get("sequence", "")))

            if not sample_id:
                raise ValueError(f"Empty id at CSV row {row_number}: {path}")
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate id {sample_id!r} in {path}")
            if not sequence:
                raise ValueError(
                    f"Empty sequence for id {sample_id!r} at CSV row {row_number}."
                )

            label = _binary_value(
                row.get(label_column),
                column=label_column,
                row_number=row_number,
            )
            clean_target = _binary_value(
                row.get("clean_target"),
                column="clean_target",
                row_number=row_number,
            )

            raw_is_noisy = str(row.get("is_noisy", "")).strip()
            if raw_is_noisy == "":
                is_noisy = int(label != clean_target)
            else:
                is_noisy = _binary_value(
                    raw_is_noisy,
                    column="is_noisy",
                    row_number=row_number,
                )

            if label_column == "target" and is_noisy != int(label != clean_target):
                raise ValueError(
                    f"Inconsistent target/clean_target/is_noisy values for "
                    f"id {sample_id!r} in {path}."
                )

            if expected_split is not None and "split" in row:
                split_value = str(row.get("split", "")).strip().lower()
                if split_value and split_value != expected_split.lower():
                    raise ValueError(
                        f"Expected split={expected_split!r}, got "
                        f"{split_value!r} for id {sample_id!r} in {path}."
                    )

            records.append(
                {
                    "id": sample_id,
                    "seq": sequence,
                    "label": label,
                    "clean_target": clean_target,
                    "is_noisy": is_noisy,
                }
            )
            seen_ids.add(sample_id)

    if not records:
        raise ValueError(f"No records found in CSV split: {path}")

    return records
