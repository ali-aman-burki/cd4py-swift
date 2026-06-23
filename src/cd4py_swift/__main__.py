"""
cd4py-swift entry point.

Subcommands:
  cd4py-swift          — find duplicates, write cluster file (mirrors original CD4Py)
  cd4py-swift dataset  — full pipeline: find duplicates, copy clean dataset
"""

import argparse
import sys
from cd4py_swift.cli import cmd_swift, cmd_dataset


def main():
    # Top-level parser — catches `cd4py-swift dataset ...`
    # and falls back to the base command for everything else.
    parser = argparse.ArgumentParser(
        prog="cd4py-swift",
        description="cd4py-swift — fast, GPU-accelerated Python code deduplication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Subcommands:\n"
            "  dataset    End-to-end pipeline: deduplicate and copy a clean dataset\n\n"
            "Run without a subcommand to find duplicates and write a cluster file\n"
            "(mirrors the original CD4Py CLI).\n\n"
            "Examples:\n"
            "  cd4py-swift --input ./projects --output-dupes dupes.jsonl.gz "
            "--output-tokens ./tokens --gpu\n"
            "  cd4py-swift dataset --input ./projects --output ./clean --gpu --workers 8"
        ),
    )

    subparsers = parser.add_subparsers(dest="subcommand")

    # Register `dataset` subcommand
    cmd_dataset.build_parser(subparsers)

    # Everything else: parse known args to detect whether a subcommand was given
    # If not, hand off to the base swift parser
    args, remaining = parser.parse_known_args()

    if args.subcommand == "dataset":
        # Re-parse with the full dataset parser to get all flags
        dataset_parser = cmd_dataset.build_parser()
        cmd_dataset.run(dataset_parser.parse_args(sys.argv[2:]))

    else:
        # No subcommand — treat all args as the base `cd4py-swift` command
        swift_parser = cmd_swift.build_parser()
        cmd_swift.run(swift_parser.parse_args(sys.argv[1:]))


if __name__ == "__main__":
    main()
