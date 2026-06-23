# cd4py-swift

Code duplication in ML training data negatively affects model performance — models trained on duplicated
source code generalise poorly to unseen code, and evaluation metrics become inflated when test samples
appear verbatim in the training set.

**cd4py-swift** is a fork of [CD4Py](https://github.com/saltudelft/CD4Py) that solves this problem by
detecting and removing near-duplicate and exact-duplicate Python source files from a dataset. The fork
was created to make the deduplication pipeline significantly faster: all CPU-bound stages are
parallelised across multiple threads, and GPU acceleration (FAISS for KNN search, PyTorch for TF-IDF
transformation) is used when available.

---

## Requirements

- Python **3.11**
- CUDA-capable GPU *(optional, for `--gpu`)*

### Dependencies

```
pip install dpu-utils scikit-learn pandas numpy annoy tqdm rich
```

For GPU acceleration (optional):
```bash
# KNN on GPU (required for --gpu KNN)
conda install -c pytorch faiss-gpu

# TF-IDF on GPU — PyTorch with CUDA support must already be installed
# (no extra install needed if torch+cuda is present)
```

---

## Commands

cd4py-swift exposes two commands.

### `cd4py-swift`

Mirrors the original CD4Py CLI. Tokenizes a Python corpus, finds duplicate clusters, and writes them
to a `.jsonl.gz` file. You then decide what to do with that file yourself.

```
usage: cd4py-swift [-h] --input INPUT --output-dupes OUTPUT_DUPES --output-tokens OUTPUT_TOKENS
                   [--d D] [--th TH] [--k K] [--tr TR]
                   [--gpu] [--workers WORKERS] [--batch-size BATCH_SIZE]

arguments:
  --input            Path to Python projects folder (each subdirectory = one project)
  --output-dupes     Output path for detected duplicate clusters (.jsonl.gz)
  --output-tokens    Output folder for tokenized files

optional:
  --d                TF-IDF vector dimension [default: 2048]
  --th               Similarity threshold [default: 0.95]
  --k                Number of nearest neighbours [default: 10]
  --tr               Annoy index tree count [default: 20]
  --gpu              Enable GPU acceleration (FAISS + PyTorch)
  --workers          CPU thread-pool size [default: 4]
  --batch-size       Docs per GPU batch for TF-IDF [default: 8192]
```

**Example:**
```bash
cd4py-swift \
  --input      /path/to/python_projects \
  --output-dupes  py_duplicates.jsonl.gz \
  --output-tokens /tmp/tokens \
  --gpu --workers 8
```

---

### `cd4py-swift dataset`

End-to-end pipeline: runs deduplication and produces a clean copy of the dataset with duplicates
removed. For each cluster of duplicates, the largest file is kept and the rest are excluded.
Additionally, any file that CD4Py could not process (tokenization failures, too few tokens) is also
excluded from the output, ensuring the output only contains files the tool could actually analyse.

```
usage: cd4py-swift dataset [-h] --input INPUT --output OUTPUT
                           [--tokens TOKENS] [--dupes DUPES] [--log LOG]
                           [--d D] [--th TH] [--k K] [--tr TR]
                           [--gpu] [--workers WORKERS] [--batch-size BATCH_SIZE]

arguments:
  --input    Path to the input dataset (folder of project subfolders)
  --output   Destination folder for the deduplicated dataset (must not exist)

optional:
  --tokens   Folder to store tokenized files [default: temp dir, deleted after run]
  --dupes    Path to write duplicate clusters .jsonl.gz [default: temp file]
  --log      Log file path for warnings and errors [default: dedup.log]
  --d        TF-IDF vector dimension [default: 2048]
  --th       Similarity threshold [default: 0.95]
  --k        Number of nearest neighbours [default: 10]
  --tr       Annoy index tree count [default: 20]
  --gpu      Enable GPU acceleration (FAISS + PyTorch)
  --workers  CPU thread-pool size [default: 4]
  --batch-size  Docs per GPU batch for TF-IDF [default: 8192]
```

**Example:**
```bash
cd4py-swift dataset \
  --input  /path/to/dataset \
  --output /path/to/deduped \
  --tokens /tmp/cd4py_tokens \
  --dupes  /tmp/cd4py_dupes.jsonl.gz \
  --gpu --workers 8 --batch-size 262144
```

**Expected input layout:**
```
dataset/
├── project_A/
│   └── **/*.py
├── project_B/
│   └── **/*.py
```

---

## How it works

1. **Tokenize** — all `.py` files are tokenized using Python's `tokenize` stdlib module, in parallel across projects
2. **Preprocess** — only identifier tokens are kept; Python keywords and short files are discarded
3. **Vectorize** — tokens are converted to TF-IDF vectors (GPU-accelerated via PyTorch if `--gpu`)
4. **KNN search** — FAISS (GPU) or annoy (CPU) finds the `k` nearest neighbours for each file
5. **Cluster** — candidate pairs above the similarity threshold are grouped into duplicate clusters, with transitive closure applied
6. **Output** — `cd4py-swift dataset` copies the clean dataset; `cd4py-swift` writes the raw cluster file for custom handling

---

## Differences from CD4Py

| Feature | CD4Py | cd4py-swift |
|---|---|---|
| GPU TF-IDF | ✗ | ✓ PyTorch |
| GPU KNN | ✗ | ✓ FAISS |
| Parallel tokenization | ✓ (joblib) | ✓ (joblib + rich progress) |
| Parallel KNN search | ✗ | ✓ thread pool |
| Parallel duplicate detection | ✗ | ✓ thread pool |
| Memory-mapped vectors | ✗ | ✓ (avoids OOM on large datasets) |
| Rich progress bars | ✗ | ✓ |
| Errors/warnings to log file | ✗ | ✓ |
| Dataset copy pipeline | ✗ | ✓ (`dataset` subcommand) |
| Excludes unprocessable files | ✗ | ✓ |

---

## License

MIT — see [LICENSE](LICENSE).
