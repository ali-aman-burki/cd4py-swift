"""
K-nearest-neighbour index build and search.

GPU path  : FAISS flat inner-product index on GPU (exact search).
CPU path  : annoy approximate index, built then saved+mmap'd to disk so the
            index file is memory-mapped rather than held in RAM alongside the
            vector memmap.

Both search paths are parallelised where possible.
"""

from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, MofNCompleteColumn, TimeElapsedColumn
import logging
import math
import numpy as np
import os
import annoy


# ---------------------------------------------------------------------------
# GPU probe
# ---------------------------------------------------------------------------

def _probe_faiss_gpu():
    try:
        import faiss
        if faiss.get_num_gpus() > 0:
            return faiss
        return None
    except ImportError:
        return None

_FAISS = _probe_faiss_gpu()


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
# Index build
# ---------------------------------------------------------------------------

def _build_faiss_index(vectors: np.memmap, vec_dim: int):
    """
    Copy vectors to a contiguous float32 array, L2-normalise, then build a
    FAISS flat IP index on GPU.  IP on unit vectors == cosine similarity.
    Returns (gpu_index, normalised_data).
    """
    faiss = _FAISS
    data  = np.array(vectors, dtype='float32')
    faiss.normalize_L2(data)
    res       = faiss.StandardGpuResources()
    cpu_index = faiss.IndexFlatIP(vec_dim)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    gpu_index.add(data)
    return gpu_index, data


def build_knn_index(vectorized_files: np.memmap, vec_dim: int,
                    knn_tree_size: int = 20,
                    work_dir: str = None,
                    use_gpu: bool = False) -> tuple:
    """
    Build the nearest-neighbour index.

    Returns a tuple (backend, index, data_or_None):
      - backend : 'faiss' or 'annoy'
      - index   : the built index object
      - data    : normalised numpy array (FAISS only, needed for search)
    """
    if use_gpu and _FAISS is not None:
        with _make_progress() as progress:
            task = progress.add_task("Building FAISS index (GPU)", total=None)
            gpu_index, data = _build_faiss_index(vectorized_files, vec_dim)
            progress.update(task, total=1, completed=1)
        return ('faiss', gpu_index, data)

    if use_gpu and _FAISS is None:
        logging.warning("faiss-gpu not found — falling back to annoy CPU")

    n         = vectorized_files.shape[0]
    annoy_idx = annoy.AnnoyIndex(vec_dim, 'dot')

    with _make_progress() as progress:
        task = progress.add_task("Building annoy index (CPU)", total=n)
        for i, v in enumerate(vectorized_files):
            annoy_idx.add_item(i, v.tolist())
            progress.advance(task)

    with _make_progress() as progress:
        task = progress.add_task("Building annoy trees", total=None)
        annoy_idx.build(knn_tree_size)
        progress.update(task, total=1, completed=1)

    # Save then reload to mmap — avoids holding the full built index in RAM
    index_path = os.path.join(work_dir, 'knn.annoy')
    annoy_idx.save(index_path)
    annoy_mmap = annoy.AnnoyIndex(vec_dim, 'dot')
    annoy_mmap.load(index_path)
    return ('annoy', annoy_mmap, None)


# ---------------------------------------------------------------------------
# KNN search
# ---------------------------------------------------------------------------

def _search_chunk_annoy(index, vectors: np.memmap,
                        row_indices: List[int],
                        k: int) -> List[Tuple[int, List[int], List[float]]]:
    """Search a slice of rows against an annoy index."""
    results = []
    for i in row_indices:
        idx, dist = index.get_nns_by_vector(vectors[i].tolist(), k, include_distances=True)
        results.append((i, idx, dist))
    return results


def find_knn(vectorized_files: np.memmap,
             knn_index_tuple: tuple,
             k: int,
             n_workers: int = 4) -> Tuple[List[List[int]], List[List[float]]]:
    """
    Find k-nearest neighbours for every file.

    FAISS : single batched GPU call.
    annoy : parallel search across n_workers threads (annoy releases the GIL).

    Returns (knn_indices, knn_distances) each of shape (n_files, k).
    """
    backend, index, data = knn_index_tuple
    n        = vectorized_files.shape[0]
    knn_idx  = [None] * n
    knn_dist = [None] * n

    if backend == 'faiss':
        with _make_progress() as progress:
            task = progress.add_task("Searching FAISS index (GPU)", total=None)
            _FAISS.normalize_L2(data)
            distances, indices = index.search(data, k)
            progress.update(task, total=1, completed=1)
        for i in range(n):
            knn_idx[i]  = indices[i].tolist()
            knn_dist[i] = distances[i].tolist()
        return knn_idx, knn_dist

    # annoy — split rows across threads
    row_indices = list(range(n))
    chunk_size  = max(1, math.ceil(n / n_workers))
    chunks      = [row_indices[i:i + chunk_size] for i in range(0, n, chunk_size)]

    with _make_progress() as progress:
        task = progress.add_task(f"Searching annoy index ({n_workers} workers)", total=n)
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_search_chunk_annoy, index, vectorized_files, chunk, k): chunk
                for chunk in chunks
            }
            for fut in as_completed(futures):
                for row_i, idx, dist in fut.result():
                    knn_idx[row_i]  = idx
                    knn_dist[row_i] = dist
                progress.advance(task, len(futures[fut]))

    return knn_idx, knn_dist
