"""
cd4py-swift dataset — end-to-end deduplication pipeline.

Runs CD4Py-Swift on an input dataset and produces a clean copy:
  - Files that failed tokenization or had too few tokens are excluded.
  - For each cluster of duplicates, the largest file (by byte size) is kept
    and the rest are excluded.
  - Empty directories in the output are pruned.
"""

import argparse
import logging
import os
import shutil
import tempfile
from dpu_utils.utils.dataloading import load_jsonl_gz
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, MofNCompleteColumn, TimeElapsedColumn
from rich.table import Table
from cd4py_swift.core.dedup import deduplicate_py_data

console = Console()


def build_parser(subparsers=None):
    kwargs = dict(
        description="cd4py-swift dataset — deduplicate a Python dataset and write a clean copy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    if subparsers is not None:
        p = subparsers.add_parser("dataset", **kwargs)
    else:
        p = argparse.ArgumentParser(**kwargs)

    p.add_argument("--input",  required=True,
                   help="Path to the input dataset (folder of project subfolders)")
    p.add_argument("--output", required=True,
                   help="Destination folder for the deduplicated dataset (must not exist)")
    p.add_argument("--tokens", default=None,
                   help="Folder for tokenized files [default: temp dir, deleted after run]")
    p.add_argument("--dupes",  default=None,
                   help="Path to write duplicate clusters .jsonl.gz [default: temp file]")
    p.add_argument("--log",    default="dedup.log",
                   help="Log file for warnings and errors")
    p.add_argument("--d",   type=int,   default=2048, help="TF-IDF vector dimension")
    p.add_argument("--th",  type=float, default=0.95, help="Similarity threshold (0–1)")
    p.add_argument("--k",   type=int,   default=10,   help="Number of nearest neighbours")
    p.add_argument("--tr",  type=int,   default=20,   help="Annoy tree count (CPU KNN only)")
    p.add_argument("--gpu", action="store_true",
                   help="Enable GPU acceleration (FAISS-GPU for KNN, PyTorch for TF-IDF)")
    p.add_argument("--workers",    type=int, default=12,
                   help="CPU thread-pool size for parallel stages")
    p.add_argument("--batch-size", type=int, default=8192,
                   help="Documents per GPU batch for TF-IDF (GPU mode only)")
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_logging(log_path: str):
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8")],
    )
    for noisy in ("joblib", "sklearn", "numba", "faiss"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _pick_largest(cluster: list[str]) -> str:
    return max(cluster, key=lambda f: os.path.getsize(f) if os.path.isfile(f) else 0)


def _build_files_to_drop(dupes_path: str) -> tuple[set[str], int, int]:
    """
    Load duplicate clusters; return (files_to_drop, n_clusters, n_in_clusters).
    Keeps the largest file in each cluster.
    """
    clusters          = list(load_jsonl_gz(dupes_path))
    n_clusters        = len(clusters)
    n_in_clusters     = sum(len(c) for c in clusters)
    to_drop: set[str] = set()

    for cluster in clusters:
        keeper = _pick_largest(cluster)
        for f in cluster:
            if f != keeper:
                to_drop.add(f)

    logging.info("Clusters: %d, files in clusters: %d, to drop: %d",
                 n_clusters, n_in_clusters, len(to_drop))
    return to_drop, n_clusters, n_in_clusters


def _collect_py_files(directory: str) -> list[str]:
    py_files = []
    for root, dirs, files in os.walk(directory):
        dirs.sort()
        for fname in sorted(files):
            if fname.endswith(".py"):
                py_files.append(os.path.join(root, fname))
    return py_files


def _copy_deduplicated(input_dir: str, output_dir: str,
                       to_drop: set[str],
                       processed_files: set[str],
                       n_workers: int = 12) -> tuple[int, int, int]:
    """
    Copy files from input_dir to output_dir, applying two filters:
      1. Only copy files CD4Py processed (in processed_files)
      2. Skip files in to_drop (duplicate losers)

    Copies are done in parallel across n_workers threads (I/O bound).
    Returns (copied, skipped_dupes, skipped_unprocessed).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    drop_norm      = {os.path.normpath(f) for f in to_drop}
    processed_norm = {os.path.normpath(f) for f in processed_files}
    py_files       = _collect_py_files(input_dir)

    # Counters shared across threads
    counters = {"copied": 0, "skipped_dupes": 0, "skipped_unprocessed": 0}
    lock = threading.Lock()

    def _process(src: str):
        norm = os.path.normpath(src)
        if norm not in processed_norm:
            with lock:
                counters["skipped_unprocessed"] += 1
            logging.debug("Skipped (not processed): %s", src)
        elif norm in drop_norm:
            with lock:
                counters["skipped_dupes"] += 1
            logging.debug("Skipped (duplicate): %s", src)
        else:
            rel = os.path.relpath(src, input_dir)
            dst = os.path.join(output_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            with lock:
                counters["copied"] += 1

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Copying files ({n_workers} workers)", total=len(py_files))
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_process, src): src for src in py_files}
            for fut in as_completed(futures):
                fut.result()
                progress.advance(task)

    return counters["copied"], counters["skipped_dupes"], counters["skipped_unprocessed"]


def _remove_empty_dirs(directory: str) -> int:
    """Remove empty directories bottom-up. Returns count removed."""
    all_dirs = []
    for root, dirs, _ in os.walk(directory, topdown=False):
        dirs.sort()
        for d in dirs:
            all_dirs.append(os.path.join(root, d))

    removed = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Pruning empty dirs", total=len(all_dirs))
        for dirpath in all_dirs:
            try:
                os.rmdir(dirpath)
                removed += 1
            except OSError:
                pass
            progress.advance(task)

    return removed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace):
    input_dir  = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)
    log_path   = os.path.abspath(args.log)

    if not os.path.isdir(input_dir):
        raise SystemExit(f"ERROR: --input does not exist or is not a directory: {input_dir}")
    if os.path.exists(output_dir):
        raise SystemExit(f"ERROR: --output already exists (won't overwrite): {output_dir}")

    _setup_logging(log_path)

    _tmp_tokens = _tmp_dupes_dir = None
    if args.tokens:
        tokens_dir = os.path.abspath(args.tokens)
        os.makedirs(tokens_dir, exist_ok=True)
    else:
        _tmp_tokens = tempfile.mkdtemp(prefix="cd4py_tokens_")
        tokens_dir  = _tmp_tokens

    if args.dupes:
        dupes_path = os.path.abspath(args.dupes)
    else:
        _tmp_dupes_dir = tempfile.mkdtemp(prefix="cd4py_dupes_")
        dupes_path     = os.path.join(_tmp_dupes_dir, "duplicates.jsonl.gz")

    try:
        console.print(Panel(
            f"[cyan]Input[/cyan]      : {input_dir}\n"
            f"[cyan]Output[/cyan]     : {output_dir}\n"
            f"[cyan]Tokens[/cyan]     : {tokens_dir}\n"
            f"[cyan]Dupes[/cyan]      : {dupes_path}\n"
            f"[cyan]Log[/cyan]        : {log_path}\n"
            f"[cyan]GPU[/cyan]        : {'enabled' if args.gpu else 'disabled'}\n"
            f"[cyan]Workers[/cyan]    : {args.workers}"
            + (f"\n[cyan]Batch size[/cyan] : {args.batch_size:,}" if args.gpu else ""),
            title="[bold blue]cd4py-swift dataset[/bold blue]",
            expand=False,
        ))

        processed_files = deduplicate_py_data(
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

        to_drop, n_clusters, n_in_clusters = _build_files_to_drop(dupes_path)

        copied, skipped_dupes, skipped_unprocessed = _copy_deduplicated(
            input_dir, output_dir, to_drop, processed_files, n_workers=args.workers
        )

        removed_dirs = _remove_empty_dirs(output_dir)

        table = Table(title="Summary", show_header=False, min_width=52)
        table.add_column(style="bold cyan")
        table.add_column(justify="right")
        table.add_row("Duplicate clusters found",          f"{n_clusters:,}")
        table.add_row("Files in clusters",                 f"{n_in_clusters:,}")
        table.add_row("Files copied (kept)",               f"[green]{copied:,}[/green]")
        table.add_row("Skipped — duplicates",              f"[red]{skipped_dupes:,}[/red]")
        table.add_row("Skipped — not processed by CD4Py",  f"[yellow]{skipped_unprocessed:,}[/yellow]")
        table.add_row("Empty dirs pruned",                 f"{removed_dirs:,}")
        table.add_row("Output",                            output_dir)
        table.add_row("Log",                               log_path)
        console.print(table)

    finally:
        if _tmp_tokens and os.path.exists(_tmp_tokens):
            shutil.rmtree(_tmp_tokens, ignore_errors=True)
        if _tmp_dupes_dir and os.path.exists(_tmp_dupes_dir):
            shutil.rmtree(_tmp_dupes_dir, ignore_errors=True)
