"""
TF-IDF vectorization of preprocessed token lists.

GPU path  : sklearn fits the vocabulary (one CPU pass), then the per-document
            transform runs on GPU via PyTorch batched tensor ops.
CPU path  : sklearn fit + parallel transform across n_workers threads, each
            writing to non-overlapping rows of a disk-backed np.memmap.

Both paths write into a np.memmap so the full matrix is never held in RAM.
"""

from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, MofNCompleteColumn, TimeElapsedColumn
from cd4py_swift import dummy_preprocessor
import logging
import math
import numpy as np
import os


# ---------------------------------------------------------------------------
# GPU probes
# ---------------------------------------------------------------------------

def _probe_torch():
    try:
        import torch
        if torch.cuda.is_available():
            return torch
        return None
    except ImportError:
        return None

_TORCH = _probe_torch()


def _make_progress(description: str, total) -> tuple:
    """Return a started (Progress, task_id) pair."""
    p = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        transient=False,
    )
    p.start()
    task = p.add_task(description, total=total)
    return p, task


# ---------------------------------------------------------------------------
# GPU transform
# ---------------------------------------------------------------------------

def _torch_tfidf_transform(vocab: dict, idf_weights: np.ndarray,
                            docs: List[List[str]], dim: int,
                            device) -> np.ndarray:
    """
    TF-IDF transform for a batch of documents on GPU using PyTorch.
    Mathematically identical to sklearn's output (float32 rounding aside).

    Steps per document:
      1. Count raw term frequencies for vocab tokens
      2. Elementwise-multiply by precomputed IDF weights
      3. L2-normalise
    """
    torch = _TORCH
    idf_gpu = torch.tensor(idf_weights, dtype=torch.float32, device=device)
    result  = np.zeros((len(docs), dim), dtype=np.float32)

    for i, doc in enumerate(docs):
        if not doc:
            continue
        tf: dict[int, float] = {}
        for token in doc:
            idx = vocab.get(token)
            if idx is not None:
                tf[idx] = tf.get(idx, 0.0) + 1.0
        if not tf:
            continue

        indices = torch.tensor(list(tf.keys()),   dtype=torch.long,    device=device)
        values  = torch.tensor(list(tf.values()), dtype=torch.float32, device=device)

        vec = torch.zeros(dim, dtype=torch.float32, device=device)
        vec.scatter_(0, indices, values)
        vec  = vec * idf_gpu
        norm = vec.norm()
        if norm > 0:
            vec = vec / norm

        result[i] = vec.cpu().numpy()

    return result


# ---------------------------------------------------------------------------
# CPU parallel transform
# ---------------------------------------------------------------------------

def _transform_chunk_cpu(tfidf: TfidfVectorizer, docs: List[List[str]],
                          dim: int, vectors: np.memmap, start: int):
    """Transform a contiguous chunk of docs with sklearn, write into memmap."""
    for offset, doc in enumerate(docs):
        vectors[start + offset] = tfidf.transform([doc]).toarray().astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def vectorize_tokenized_files(preprocessed_file_tokens: List[List[str]],
                               dim_tfidf_vec: int,
                               work_dir: str,
                               use_gpu: bool = False,
                               n_workers: int = 4,
                               gpu_batch_size: int = 8192) -> np.memmap:
    """
    Vectorize preprocessed token lists into a TF-IDF matrix stored on disk.

    Args:
        preprocessed_file_tokens: list of token lists, one per file.
        dim_tfidf_vec:            number of TF-IDF features (vector dimension).
        work_dir:                 directory for the memmap file.
        use_gpu:                  use PyTorch GPU transform if available.
        n_workers:                thread-pool size for CPU parallel path.
        gpu_batch_size:           documents per GPU batch.

    Returns:
        np.memmap of shape (n_files, dim_tfidf_vec), dtype float32.
        File is at <work_dir>/tfidf_vectors.mmap — caller is responsible for cleanup.
    """
    n         = len(preprocessed_file_tokens)
    mmap_path = os.path.join(work_dir, 'tfidf_vectors.mmap')
    vectors   = np.memmap(mmap_path, dtype='float32', mode='w+', shape=(n, dim_tfidf_vec))

    # Vocabulary fit always runs on CPU — it's a single pass and is fast regardless
    tfidf = TfidfVectorizer(
        analyzer='word',
        tokenizer=dummy_preprocessor,
        preprocessor=dummy_preprocessor,
        token_pattern=None,
        max_features=dim_tfidf_vec,
    )
    tfidf.fit(preprocessed_file_tokens)

    if use_gpu and _TORCH is not None:
        device    = _TORCH.device('cuda')
        vocab     = tfidf.vocabulary_
        idf_w     = tfidf.idf_.astype(np.float32)
        n_batches = math.ceil(n / gpu_batch_size)

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Vectorizing (GPU)", total=n_batches)
            for start in range(0, n, gpu_batch_size):
                chunk = preprocessed_file_tokens[start:start + gpu_batch_size]
                vectors[start:start + len(chunk)] = _torch_tfidf_transform(
                    vocab, idf_w, chunk, dim_tfidf_vec, device
                )
                progress.advance(task)

        vectors.flush()
        return vectors

    if use_gpu and _TORCH is None:
        logging.warning("PyTorch/CUDA not available — falling back to CPU TF-IDF")

    # CPU parallel path
    chunk_size = max(1, math.ceil(n / n_workers))
    chunks     = [
        (preprocessed_file_tokens[i:i + chunk_size], i)
        for i in range(0, n, chunk_size)
    ]

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task(f"Vectorizing (CPU, {n_workers} workers)", total=n)
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_transform_chunk_cpu, tfidf, docs, dim_tfidf_vec, vectors, start): start
                for docs, start in chunks
            }
            for fut in as_completed(futures):
                fut.result()
                start = futures[fut]
                progress.advance(task, min(chunk_size, n - start))

    vectors.flush()
    return vectors
