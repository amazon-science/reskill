"""Prepare ALFWorld dummy parquet files for ReSkill training."""

from __future__ import annotations

import argparse

try:
    from scripts.data_prep.common import write_dummy_parquets
except ModuleNotFoundError:
    from common import write_dummy_parquets


def check_alfworld_ready() -> None:
    import alfworld.agents.environment as environment

    print("ALFWorld data directory:", environment.ALFWORLD_DATA)
    print("ALFWorld is ready.")


def prepare_alfworld_data(
    output_dir: str,
    train_size: int = 3800,
    val_size: int = 64,
    test_size: int = 274,
    check_runtime: bool = True,
):
    if check_runtime:
        check_alfworld_ready()

    print("ALFWorld uses ~3,800 training games and 140 seen + 134 unseen eval games.")
    print("Game files are loaded at runtime from the alfworld package.")
    return write_dummy_parquets(output_dir, "alfworld", train_size, val_size, test_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/alfworld")
    parser.add_argument("--train_size", type=int, default=3800)
    parser.add_argument("--val_size", type=int, default=64)
    parser.add_argument("--test_size", type=int, default=274)
    parser.add_argument("--skip_runtime_check", action="store_true")
    args = parser.parse_args()

    prepare_alfworld_data(
        output_dir=args.output_dir,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        check_runtime=not args.skip_runtime_check,
    )


if __name__ == "__main__":
    main()
