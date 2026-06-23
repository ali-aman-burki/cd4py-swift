"""
Deduplication orchestration.

Covers:
  - Loading tokenized files from disk
  - Preprocessing (identifier filtering, keyword removal, sub-token splitting)
  - Finding duplicate pairs and resolving transitive clusters
  - The main deduplicate_py_data() entry point
"""

from typing import List, Tuple, Dict, Set
from dpu_utils.utils.dataloading import load_jsonl_gz, save_jsonl_gz
from dpu_utils.codeutils import get_language_keywords, split_identifier_into_parts
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, MofNCompleteColumn, TimeElapsedColumn
from cd4py_swift import log_step
from cd4py_swift.core.tokenizer import tokenize_all_project_folders
from cd4py_swift.core.vectorizer import vectorize_tokenized_files
from cd4py_swift.core.knn import build_knn_index, find_knn
import logging
import math
import os
import re
import shutil
import tempfile
import time
import numpy as np
import pandas as pd


NO_IDENTIFIER_TOKENS = 20
IDENTIFIER_REGEX = re.compile('[_a-zA-Z][_a-zA-Z0-9]*')


def _make_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        transient=False,
    )


# ---------------------------------------------------------------------------
# Load & preprocess
# ---------------------------------------------------------------------------

def get_tokenized_py_files(tokenized_files_path: str) -> List[Tuple[str, List[str]]]:
    """Load all .jsonl.gz token files from tokenized_files_path."""
    token_files = [f for f in os.listdir(tokenized_files_path) if f.endswith(".jsonl.gz")]
    result: List[Tuple[str, List[str]]] = []

    with _make_progress() as progress:
        task = progress.add_task("Loading token files", total=len(token_files))
        for f in token_files:
            for d in load_jsonl_gz(os.path.join(tokenized_files_path, f)):
                if d['tokens']:
                    result.append((d['filename'], d['tokens']))
            progress.advance(task)

    return result


def preprocess_tokenized_files(tokenized_py_files: List[Tuple[str, List[str]]]) -> Tuple[pd.DataFrame,
                                                                                          List[List[str]]]:
    """
    Preprocess tokenized files:
    1. Drop files with fewer than NO_IDENTIFIER_TOKENS tokens
    2. Keep only identifier tokens; remove Python keywords
    3. Split camelCase/snake_case identifiers into sub-tokens

    Returns (df_with_filename_column, list_of_token_lists).
    """
    df = pd.DataFrame(tokenized_py_files, columns=['filename', 'tokens'])
    before = df.shape[0]
    df = df[df['tokens'].map(lambda x: len(x) > NO_IDENTIFIER_TOKENS)].copy()
    dropped = before - df.shape[0]
    if dropped:
        logging.info("Preprocessing: dropped %d files with fewer than %d tokens", dropped, NO_IDENTIFIER_TOKENS)
    logging.info("Preprocessing: %d files retained, %d total tokens",
                 df.shape[0], sum(df['tokens'].apply(len)))

    py_keywords = get_language_keywords('python')
    all_tokens  = df['tokens'].tolist()
    n           = len(all_tokens)

    with _make_progress() as progress:
        task = progress.add_task("Filtering identifier tokens", total=n)
        filtered = []
        for f_tks in all_tokens:
            filtered.append([t for t in f_tks if IDENTIFIER_REGEX.match(t) and t not in py_keywords])
            progress.advance(task)

    with _make_progress() as progress:
        task = progress.add_task("Splitting identifiers", total=n)
        split = []
        for src in filtered:
            split.append([i for t in src for i in split_identifier_into_parts(t)])
            progress.advance(task)

    return df, split


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

def _find_dupes_chunk(filenames: List[str], threshold: float, k: int,
                      knn_idx_chunk: List[List[int]],
                      knn_dist_chunk: List[List[float]],
                      all_filenames: List[str],
                      start: int) -> Dict[str, List[Tuple[str, float]]]:
    """Process one chunk of files and return a partial clone_sets dict."""
    partial: Dict[str, List[Tuple[str, float]]] = {}
    for fname, nbr_idx, nbr_dist in zip(filenames, knn_idx_chunk, knn_dist_chunk):
        partial[fname] = []
        for j in range(1, k):
            if nbr_dist[j] > threshold:
                neighbour = all_filenames[nbr_idx[j]]
                if fname != neighbour:
                    partial[fname].append((neighbour, nbr_dist[j]))
            else:
                break
    return partial


def find_duplicate_sets(df_tokenized_files: pd.DataFrame,
                        t: float, k: int,
                        files_knn_idx: List[List[int]],
                        files_knn_dist: List[List[float]],
                        n_workers: int = 4) -> Dict[str, List[Tuple[str, float]]]:
    """
    Build a per-file mapping of near-duplicate candidates.
    Parallelised across n_workers threads — each handles an independent slice.
    """
    n             = df_tokenized_files.shape[0]
    all_filenames = df_tokenized_files['filename'].tolist()
    chunk_size    = max(1, math.ceil(n / n_workers))

    chunks = [
        (all_filenames[i:i + chunk_size],
         files_knn_idx[i:i + chunk_size],
         files_knn_dist[i:i + chunk_size],
         i)
        for i in range(0, n, chunk_size)
    ]

    clone_sets: Dict[str, List[Tuple[str, float]]] = {}
    with _make_progress() as progress:
        task = progress.add_task(f"Finding duplicates ({n_workers} workers)", total=n)
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    _find_dupes_chunk, fnames, t, k, idx_c, dist_c, all_filenames, start
                ): len(fnames)
                for fnames, idx_c, dist_c, start in chunks
            }
            for fut in as_completed(futures):
                clone_sets.update(fut.result())
                progress.advance(task, futures[fut])

    return clone_sets


def find_transitive_duplicate_sets(duplicate_files_set: Dict[str, List[Tuple[str, float]]]) -> Tuple[List[Set[str]],
                                                                                                      List[str]]:
    """
    Apply transitive closure: if A≈B and B≈C then {A,B,C} is one cluster.
    Inherently sequential graph traversal — unchanged from original CD4Py.
    """
    duplicate_files_closure: List[Set[str]] = []
    files_clone_idx: Dict[str, int]         = {}
    documents_to_visit = set(duplicate_files_set.keys())

    with _make_progress() as progress:
        task = progress.add_task("Resolving transitive clusters", total=len(documents_to_visit))

        while documents_to_visit:
            current_idx = documents_to_visit.pop()
            progress.advance(task)
            current_idx_closure = {current_idx}
            visit_queue         = []

            for f in duplicate_files_set[current_idx]:
                if f[0] in files_clone_idx:
                    duplicate_files_closure[files_clone_idx[f[0]]].add(current_idx)
                    files_clone_idx[current_idx] = files_clone_idx[f[0]]
                    visit_queue = []
                    break
                else:
                    visit_queue.append(f[0])

            if visit_queue:
                while visit_queue:
                    other_idx = visit_queue.pop()
                    current_idx_closure.add(other_idx)
                    if other_idx in documents_to_visit:
                        documents_to_visit.discard(other_idx)
                        progress.advance(task)  # account for nodes consumed in inner loop
                    visit_queue.extend(
                        nxt[0] for nxt in duplicate_files_set[other_idx]
                        if nxt[0] in documents_to_visit
                    )
                duplicate_files_closure.append(set(current_idx_closure))
                for f in current_idx_closure:
                    files_clone_idx[f] = len(duplicate_files_closure) - 1

    return duplicate_files_closure, [f for c in duplicate_files_closure for f in c]


def report_stats(no_src_files: int, no_duplicate_files: int,
                 clusters: List[Set[str]]):
    """Log duplicate statistics."""
    logging.info("Duplicated files: %d (%.2f%%)", no_duplicate_files,
                 no_duplicate_files / no_src_files * 100.0)
    logging.info("Clusters: %d", len(clusters))
    logging.info("Avg files/cluster: %.2f", np.mean([len(c) for c in clusters]))
    logging.info("Median files/cluster: %.2f", np.median([len(c) for c in clusters]))
    logging.info("Duplication ratio: %.2f%%",
                 (no_duplicate_files - len(clusters)) / no_src_files * 100.0)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def deduplicate_py_data(py_projects_path: str,
                        tokenized_files_path: str,
                        detected_duplicate_f_path: str,
                        dim_tfidf_vec: int,
                        t: float,
                        no_knn: int,
                        knn_tree_size: int,
                        work_dir: str = None,
                        use_gpu: bool = False,
                        n_workers: int = 4,
                        gpu_batch_size: int = 8192) -> Set[str]:
    """
    Full deduplication pipeline.

    Returns the set of absolute file paths that were successfully processed
    (tokenized and passed the minimum-token filter). Only these files should
    appear in the output dataset — anything else was silently dropped by CD4Py.

    Args:
        py_projects_path:          Folder of Python project subdirectories.
        tokenized_files_path:      Where to write tokenized .jsonl.gz files.
        detected_duplicate_f_path: Where to write duplicate cluster file.
        dim_tfidf_vec:             TF-IDF vector dimension.
        t:                         Similarity threshold (0–1).
        no_knn:                    Number of nearest neighbours.
        knn_tree_size:             annoy tree count (ignored for FAISS).
        work_dir:                  Temp dir for memmap/annoy files (auto-cleaned).
        use_gpu:                   Enable GPU acceleration.
        n_workers:                 Thread-pool size for parallel CPU stages.
        gpu_batch_size:            Documents per GPU batch (TF-IDF GPU path).
    """
    start_t = time.time()

    if use_gpu:
        from cd4py_swift.core.vectorizer import _TORCH
        from cd4py_swift.core.knn import _FAISS
        logging.info("TF-IDF backend : %s", "PyTorch/CUDA" if _TORCH else "sklearn CPU (no CUDA)")
        logging.info("KNN backend    : %s", "FAISS-GPU"    if _FAISS  else "annoy CPU (no faiss-gpu)")

    _own_work_dir = work_dir is None
    if _own_work_dir:
        work_dir = tempfile.mkdtemp(prefix='cd4py_work_',
                                    dir=os.path.dirname(tokenized_files_path))
    else:
        os.makedirs(work_dir, exist_ok=True)

    try:
        log_step("Tokenizing Python source code files")
        tokenize_all_project_folders(py_projects_path, tokenized_files_path)

        log_step("Loading tokenized files")
        all_tokenized = get_tokenized_py_files(tokenized_files_path)

        log_step("Preprocessing")
        df, all_tokens = preprocess_tokenized_files(all_tokenized)

        # Files CD4Py actually processed — used by the dataset command to filter output
        processed_files: Set[str] = set(df['filename'].tolist())

        log_step(f"Vectorizing ({'PyTorch GPU' if use_gpu else f'sklearn CPU, {n_workers} workers'})")
        vectors = vectorize_tokenized_files(
            all_tokens, dim_tfidf_vec,
            work_dir=work_dir, use_gpu=use_gpu,
            n_workers=n_workers, gpu_batch_size=gpu_batch_size,
        )

        log_step(f"Building KNN index ({'FAISS-GPU' if use_gpu else 'annoy CPU'})")
        knn_tuple = build_knn_index(vectors, dim_tfidf_vec, knn_tree_size,
                                    work_dir=work_dir, use_gpu=use_gpu)

        log_step(f"Searching KNN ({'FAISS-GPU' if use_gpu else f'annoy, {n_workers} workers'})")
        knn_idx, knn_dist = find_knn(vectors, knn_tuple, no_knn, n_workers=n_workers)

        log_step(f"Finding duplicate sets ({n_workers} workers)")
        dup_sets = find_duplicate_sets(df, t, no_knn, knn_idx, knn_dist, n_workers=n_workers)

        log_step("Resolving transitive clusters")
        dup_closure, dup_files = find_transitive_duplicate_sets(dup_sets)
        assert len(dup_files) == sum(len(c) for c in dup_closure)

        log_step("Saving duplicate clusters")
        report_stats(df.shape[0], len(dup_files), dup_closure)
        save_jsonl_gz([list(c) for c in dup_closure], detected_duplicate_f_path)

        logging.info("Finished in %.2f minutes", (time.time() - start_t) / 60.0)
        return processed_files

    finally:
        if _own_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)
        else:
            for fname in ('tfidf_vectors.mmap', 'knn.annoy'):
                fpath = os.path.join(work_dir, fname)
                if os.path.exists(fpath):
                    os.remove(fpath)
