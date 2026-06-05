"""Prepare Search-R1 style parquet files and retriever assets for ReSkill search training.

Usage
-----
Training data only (default):
    python scripts/data_prep/prepare_search.py --output_dir data/search

Training data + retriever assets (index, corpus, model):
    python scripts/data_prep/prepare_search.py \\
        --output_dir data/search \\
        --retriever_dir /path/to/searchR1 \\
        --retriever_model_dir /path/to/e5-base-v2

The retriever assets are large (~55 GB total) and require internet access.
"""

from __future__ import annotations

import argparse
import gzip
import os
import shutil
from typing import Any

import pandas as pd

try:
    from scripts.data_prep.common import SYSTEM_PROMPT, chat_prompt, write_parquet
except ModuleNotFoundError:
    from common import SYSTEM_PROMPT, chat_prompt, write_parquet


def make_search_row(
    question: str,
    ground_truth: Any,
    source: str,
    split: str,
    index: int,
) -> dict[str, Any]:
    env_kwargs = {
        "question": question,
        "ground_truth": ground_truth,
        "data_source": source,
    }
    return {
        "data_source": source,
        "prompt": chat_prompt(question, SYSTEM_PROMPT),
        "ability": "agent",
        "reward_model": {"ground_truth": ground_truth, "style": "rule"},
        "extra_info": {
            "split": split,
            "index": index,
            "need_tools_kwargs": True,
            "question": question,
            "tools_kwargs": {
                "search": {
                    "create_kwargs": env_kwargs,
                },
            },
        },
        "metadata": None,
        "env_kwargs": env_kwargs,
    }


def _sample_records() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    train = [
        {
            "question": "Which city is the Eiffel Tower located in?",
            "ground_truth": ["Paris"],
            "data_source": "nq",
        },
        {
            "question": "What company created the Python package manager pip?",
            "ground_truth": ["The Python Packaging Authority", "PyPA"],
            "data_source": "hotpotqa",
        },
    ]
    val = [
        {
            "question": "Which programming language uses pip as a package installer?",
            "ground_truth": ["Python"],
            "data_source": "nq",
        },
        {
            "question": "Which landmark is in Paris and made of iron lattice?",
            "ground_truth": ["Eiffel Tower"],
            "data_source": "hotpotqa",
        },
        {
            "question": "Which scientist discovered polonium?",
            "ground_truth": ["Marie Curie"],
            "data_source": "popqa",
        },
        {
            "question": "What is the largest planet in the Solar System?",
            "ground_truth": ["Jupiter"],
            "data_source": "triviaqa",
        },
        {
            "question": "What country is the author of Hamlet from?",
            "ground_truth": ["England"],
            "data_source": "2wikimultihopqa",
        },
        {
            "question": "What field did the inventor of the telephone work in?",
            "ground_truth": ["communication", "invention"],
            "data_source": "musique",
        },
        {
            "question": "Is Mount Everest located in the Andes?",
            "ground_truth": ["No"],
            "data_source": "bamboogle",
        },
    ]
    test = [
        {
            "question": "What is the capital of France?",
            "ground_truth": ["Paris"],
            "data_source": "nq",
        },
        {
            "question": "Which city hosted the landmark known as the Eiffel Tower?",
            "ground_truth": ["Paris"],
            "data_source": "hotpotqa",
        },
        {
            "question": "What occupation is Marie Curie known for?",
            "ground_truth": ["physicist", "chemist"],
            "data_source": "popqa",
        },
        {
            "question": "Who wrote Pride and Prejudice?",
            "ground_truth": ["Jane Austen"],
            "data_source": "triviaqa",
        },
        {
            "question": "What country contains the city where the Louvre is located?",
            "ground_truth": ["France"],
            "data_source": "2wikimultihopqa",
        },
        {
            "question": "What is the profession of the person who developed relativity?",
            "ground_truth": ["physicist"],
            "data_source": "musique",
        },
        {
            "question": "Is the capital of Australia Sydney?",
            "ground_truth": ["No"],
            "data_source": "bamboogle",
        },
    ]
    return train, val, test


def _record_from_row(row: pd.Series) -> dict[str, Any]:
    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict) and "ground_truth" in reward_model:
        ground_truth = reward_model["ground_truth"]
    else:
        ground_truth = row.get("golden_answers", [])
    return {
        "question": row.get("question", ""),
        "ground_truth": ground_truth,
        "data_source": row.get("data_source", ""),
    }


def _split_by_domain(
    df: pd.DataFrame,
    val_samples_per_domain: int,
    test_samples_per_domain: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    val_subsets = []
    test_subsets = []
    for domain in sorted(df["data_source"].dropna().unique()):
        domain_df = df[df["data_source"] == domain].sample(frac=1.0, random_state=seed)
        total = len(domain_df)
        if total >= val_samples_per_domain + test_samples_per_domain:
            val_subsets.append(domain_df.iloc[:val_samples_per_domain])
            test_subsets.append(
                domain_df.iloc[
                    val_samples_per_domain:val_samples_per_domain + test_samples_per_domain
                ]
            )
        else:
            val_subsets.append(domain_df.iloc[:min(val_samples_per_domain, total)])
            test_subsets.append(domain_df.iloc[:min(test_samples_per_domain, total)])

    empty = df.iloc[:0]
    val_df = pd.concat(val_subsets, ignore_index=True) if val_subsets else empty
    test_df = pd.concat(test_subsets, ignore_index=True) if test_subsets else empty
    return val_df, test_df


def _load_hf_records(
    hf_repo_id: str,
    train_size: int,
    val_samples_per_domain: int,
    eval_samples_per_domain: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    from huggingface_hub import hf_hub_download

    train_path = hf_hub_download(
        repo_id=hf_repo_id,
        filename="train.parquet",
        repo_type="dataset",
    )
    test_path = hf_hub_download(
        repo_id=hf_repo_id,
        filename="test.parquet",
        repo_type="dataset",
    )

    train_raw = pd.read_parquet(train_path).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    train_records = [_record_from_row(row) for _, row in train_raw.iloc[:train_size].iterrows()]

    test_raw = pd.read_parquet(test_path)
    val_sample, test_sample = _split_by_domain(
        test_raw,
        val_samples_per_domain=val_samples_per_domain,
        test_samples_per_domain=eval_samples_per_domain,
        seed=seed,
    )
    val_records = [_record_from_row(row) for _, row in val_sample.iterrows()]
    test_records = [_record_from_row(row) for _, row in test_sample.iterrows()]
    return train_records, val_records, test_records


def prepare_search_data(
    output_dir: str,
    sample_data: bool = False,
    hf_repo_id: str = "PeterJinGo/nq_hotpotqa_train",
    train_size: int = 3000,
    val_samples_per_domain: int = 300,
    eval_samples_per_domain: int = 300,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if sample_data:
        train_records, val_records, test_records = _sample_records()
    else:
        train_records, val_records, test_records = _load_hf_records(
            hf_repo_id=hf_repo_id,
            train_size=train_size,
            val_samples_per_domain=val_samples_per_domain,
            eval_samples_per_domain=eval_samples_per_domain,
            seed=seed,
        )

    train_rows = [
        make_search_row(
            question=record["question"],
            ground_truth=record["ground_truth"],
            source=record["data_source"],
            split="train",
            index=idx,
        )
        for idx, record in enumerate(train_records[:train_size])
    ]
    val_rows = [
        make_search_row(
            question=record["question"],
            ground_truth=record["ground_truth"],
            source=record["data_source"],
            split="val",
            index=idx,
        )
        for idx, record in enumerate(val_records)
    ]
    test_rows = [
        make_search_row(
            question=record["question"],
            ground_truth=record["ground_truth"],
            source=record["data_source"],
            split="test",
            index=idx,
        )
        for idx, record in enumerate(test_records)
    ]

    train_df = write_parquet(train_rows, os.path.join(output_dir, "train.parquet"))
    val_df = write_parquet(val_rows, os.path.join(output_dir, "val.parquet"))
    test_df = write_parquet(test_rows, os.path.join(output_dir, "test.parquet"))
    return train_df, val_df, test_df


def download_retriever_assets(retriever_dir: str, retriever_model_dir: str | None) -> None:
    """Download FAISS index, Wikipedia corpus, and retriever model to retriever_dir."""
    from huggingface_hub import snapshot_download
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError("huggingface_hub is required: pip install huggingface_hub") from e

    os.makedirs(retriever_dir, exist_ok=True)

    # FAISS index — stored as two split parts on HuggingFace
    print("Downloading e5 FAISS index parts (part_aa + part_ab, ~43 GB total)...")
    part_paths = []
    for part in ["part_aa", "part_ab"]:
        path = hf_hub_download(
            repo_id="PeterJinGo/wiki-18-e5-index",
            filename=part,
            repo_type="dataset",
            local_dir=retriever_dir,
        )
        part_paths.append(path)
        print(f"  {part} -> {path}")

    index_path = os.path.join(retriever_dir, "e5_Flat.index")
    print(f"Assembling {index_path} ...")
    with open(index_path, "wb") as out:
        for part_path in part_paths:
            with open(part_path, "rb") as inp:
                shutil.copyfileobj(inp, out)
    print(f"  e5_Flat.index written ({os.path.getsize(index_path) / 1e9:.1f} GB)")

    # Wikipedia 2018 corpus
    print("Downloading wiki-18 corpus (~5 GB gz / ~33 GB uncompressed)...")
    gz_path = hf_hub_download(
        repo_id="PeterJinGo/wiki-18-corpus",
        filename="wiki-18.jsonl.gz",
        repo_type="dataset",
        local_dir=retriever_dir,
    )
    jsonl_path = os.path.join(retriever_dir, "wiki-18.jsonl")
    print(f"Decompressing {jsonl_path} ...")
    with gzip.open(gz_path, "rb") as f_in, open(jsonl_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    print(f"  wiki-18.jsonl written ({os.path.getsize(jsonl_path) / 1e9:.1f} GB)")

    # Retriever model
    if retriever_model_dir:
        print(f"Downloading intfloat/e5-base-v2 -> {retriever_model_dir} ...")
        snapshot_download(
            repo_id="intfloat/e5-base-v2",
            local_dir=retriever_model_dir,
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model*"],
        )
        print("  e5-base-v2 done")

    print("\nRetriever assets ready:")
    print(f"  Index : {index_path}")
    print(f"  Corpus: {jsonl_path}")
    if retriever_model_dir:
        print(f"  Model : {retriever_model_dir}")
    print("\nStart the retrieval server before training:")
    model_arg = retriever_model_dir or "intfloat/e5-base-v2"
    print(
        f"  conda activate retriever\n"
        f"  python verl/examples/sglang_multiturn/search_r1_like/local_dense_retriever/retrieval_server.py \\\n"
        f"      --index_path {index_path} \\\n"
        f"      --corpus_path {jsonl_path} \\\n"
        f"      --retriever_name e5 \\\n"
        f"      --retriever_model {model_arg} \\\n"
        f"      --faiss_gpu --port 8000 &\n"
        f"  sleep 120"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare training data and (optionally) retriever assets for the search environment."
    )
    # Training data args
    parser.add_argument("--output_dir", default="data/search",
                        help="Directory for train/val/test parquet files.")
    parser.add_argument("--hf_repo_id", default="PeterJinGo/nq_hotpotqa_train")
    parser.add_argument("--train_size", type=int, default=3000)
    parser.add_argument("--val_samples_per_domain", type=int, default=300)
    parser.add_argument("--eval_samples_per_domain", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_data", action="store_true",
                        help="Use built-in sample data instead of downloading from HuggingFace.")
    # Retriever asset args
    parser.add_argument("--retriever_dir", default=None,
                        help="If set, download the FAISS index and wiki-18 corpus here (~55 GB). "
                             "Requires internet access.")
    parser.add_argument("--retriever_model_dir", default=None,
                        help="If set, also download intfloat/e5-base-v2 to this directory.")
    args = parser.parse_args()

    prepare_search_data(
        output_dir=args.output_dir,
        sample_data=args.sample_data,
        hf_repo_id=args.hf_repo_id,
        train_size=args.train_size,
        val_samples_per_domain=args.val_samples_per_domain,
        eval_samples_per_domain=args.eval_samples_per_domain,
        seed=args.seed,
    )

    if args.retriever_dir:
        download_retriever_assets(
            retriever_dir=args.retriever_dir,
            retriever_model_dir=args.retriever_model_dir,
        )


if __name__ == "__main__":
    main()
