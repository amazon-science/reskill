"""Shared helpers for ReSkill data preparation scripts."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd


SYSTEM_PROMPT = "You are a helpful and harmless assistant."


def chat_prompt(user_content: str, system_content: str | None = SYSTEM_PROMPT) -> list[dict[str, str]]:
    messages = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": user_content})
    return messages


def write_parquet(rows: list[dict[str, Any]], path: str) -> pd.DataFrame:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    print(f"Wrote {len(df)} rows to {path}")
    return df


def make_dummy_rows(data_source: str, split: str, count: int) -> list[dict[str, Any]]:
    return [
        {
            "data_source": data_source,
            "prompt": chat_prompt(""),
            "ability": "agent",
            "extra_info": {"split": split, "index": idx},
        }
        for idx in range(count)
    ]


def write_dummy_parquets(
    output_dir: str,
    data_source: str,
    train_size: int,
    val_size: int,
    test_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = write_parquet(
        make_dummy_rows(data_source=data_source, split="train", count=train_size),
        os.path.join(output_dir, "train.parquet"),
    )
    val_df = write_parquet(
        make_dummy_rows(data_source=data_source, split="val", count=val_size),
        os.path.join(output_dir, "val.parquet"),
    )
    test_df = write_parquet(
        make_dummy_rows(data_source=data_source, split="test", count=test_size),
        os.path.join(output_dir, "test.parquet"),
    )
    return train_df, val_df, test_df
