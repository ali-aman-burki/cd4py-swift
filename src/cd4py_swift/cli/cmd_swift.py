"""
cd4py-swift — mirrors the original CD4Py CLI with GPU and parallelism flags added.

Tokenizes a Python corpus, finds duplicate clusters, and writes them to a
.jsonl.gz file. What you do with that file is up to you.
"""

import argparse
import logging
import os
import tempfile
from rich.console import Console
from rich.panel import Panel
from cd4py_swift.core.dedup import deduplicate_py_data

console = Console()


def build_parser(subparsers=None):
    """
    Build the argument parser for the base `cd4py-swift` command.
    If subparsers is given, registers as a sub-command; otherwise returns a
    standalone ArgumentParser (used when invoked without a subcommand).
    """
    kwargs = dict(
        description="CD4Py-Swift — fast near-duplicate detection for Python code corpora",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    if subparsers is not None:
        p = subparsers.add_parser("run", **kwargs)
    else:
        p = argparse.ArgumentParser(**kwargs)

    p.add_argument("--input",         required=True,
                   help="Path to Python projects folder (each subdirectory = one project)")
    p.add_argument("--output-dupes",  required=True,
                   help="Output path for detected duplicate clusters (.jsonl.gz)")
    p.add_argument("--output-tokens", required=True,
                   help="Output folder for tokenized files")
    p.add_argument("--d",   type=int,   default=2048, help="TF-IDF vector dimension")
    p.add_argument("--th",  type=float, default=0.95, help="Similarity threshold (0–1)")
    p.add_argument("--k",   type=int,   default=10,   help="Number of nearest neighbours")
    p.add_argument("--tr",  type=int,   default=20,   help="Annoy tree count (CPU KNN only)")
    p.add_argument("--gpu", action="store_true",
                   help="Enable GPU acceleration (FAISS-GPU for KNN, PyTorch for TF-IDF)")
    p.add_argument("--workers",    type=int, default=4,
                   help="CPU thread-pool size for parallel stages")
    p.add_argument("--batch-size", type=int, default=8192,
                   help="Documents per GPU batch for TF-IDF (GPU mode only)")
    p.add_argument("--log", default="cd4py_swift.log",
                   help="Log file for warnings and errors")
    return p


def run(args: argparse.Namespace):
    log_path  = os.path.abspath(args.log)
    input_dir = os.path.abspath(args.input)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8")],
    )
    for noisy in ("joblib", "sklearn", "numba", "faiss"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if not os.path.isdir(input_dir):
        raise SystemExit(f"ERROR: --input does not exist or is not a directory: {input_dir}")

    tokens_dir = os.path.abspath(args.output_tokens)
    dupes_path = os.path.abspath(args.output_dupes)
    os.makedirs(tokens_dir, exist_ok=True)
    os.makedirs(os.path.dirname(dupes_path) or ".", exist_ok=True)

    console.print(Panel(
        f"[cyan]Input[/cyan]         : {input_dir}\n"
        f"[cyan]Output dupes[/cyan]  : {dupes_path}\n"
        f"[cyan]Output tokens[/cyan] : {tokens_dir}\n"
        f"[cyan]Log[/cyan]           : {log_path}\n"
        f"[cyan]GPU[/cyan]           : {'enabled' if args.gpu else 'disabled'}\n"
        f"[cyan]Workers[/cyan]       : {args.workers}"
        + (f"\n[cyan]Batch size[/cyan]    : {args.batch_size:,}" if args.gpu else ""),
        title="[bold blue]cd4py-swift[/bold blue]",
        expand=False,
    ))

    deduplicate_py_data(
        py_projects_path=input_dir,
        tokenized_files_path=tokens_dir,
        detected_duplicate_f_path=dupes_path,
        dim_tfidf_vec=args.d,
        t=args.th,
        no_knn=args.k,
        knn_tree_size=args.tr,
        use_gpu=args.gpu,
        n_workers=args.workers,
        gpu_batch_size=args.batch_size,
    )

    console.print(Panel(
        f"Duplicate clusters written to [bold]{dupes_path}[/bold]\n"
        f"See [bold]{log_path}[/bold] for details.",
        title="[bold green]Done[/bold green]",
        expand=False,
    ))
