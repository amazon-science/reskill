"""Prepare ScienceWorld dummy parquet files for ReSkill training."""

from __future__ import annotations

import argparse

try:
    from scripts.data_prep.common import write_dummy_parquets
except ModuleNotFoundError:
    from common import write_dummy_parquets


def check_scienceworld_ready() -> None:
    from scienceworld import ScienceWorldEnv

    env = ScienceWorldEnv("")
    tasks = env.getTaskNames()
    print(f"ScienceWorld installed with {len(tasks)} task types:")
    for task in tasks:
        print(f"  - {task}")
    env.close()


def prepare_scienceworld_data(
    output_dir: str,
    train_size: int = 16,
    val_size: int = 32,
    test_size: int = 32,
    check_runtime: bool = True,
):
    if check_runtime:
        check_scienceworld_ready()

    print("ScienceWorld uses the 'electricity' split (2 task types):")
    print("  - identifying power components")
    print("  - testing conductivity")
    print(f"Training parquet rows: {train_size}")
    print(f"Validation parquet rows: {val_size}")
    print(f"Final eval parquet rows: {test_size}")
    print("Runtime train tasks: power-component, test-conductivity")
    print("Runtime val/eval tasks: power-component-renewable-vs-nonrenewable-energy, test-conductivity-of-unknown-substances")
    print("Task pool is built dynamically at runtime.")
    return write_dummy_parquets(output_dir, "scienceworld", train_size, val_size, test_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/scienceworld")
    parser.add_argument("--train_size", type=int, default=16)
    parser.add_argument("--val_size", type=int, default=32)
    parser.add_argument("--test_size", type=int, default=32)
    parser.add_argument("--skip_runtime_check", action="store_true")
    args = parser.parse_args()

    prepare_scienceworld_data(
        output_dir=args.output_dir,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        check_runtime=not args.skip_runtime_check,
    )


if __name__ == "__main__":
    main()
