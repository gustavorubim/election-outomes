from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_parquet(df: pl.DataFrame, path: Path) -> Path:
    ensure_parent(path)
    df.write_parquet(path)
    return path


def read_parquet(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path)


def write_json(payload: dict[str, Any], path: Path) -> Path:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def write_text(text: str, path: Path) -> Path:
    ensure_parent(path)
    path.write_text(text, encoding="utf-8")
    return path
