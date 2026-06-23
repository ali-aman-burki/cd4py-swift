"""cd4py-swift — fast, GPU-accelerated near-duplicate detection for Python code datasets."""

__version__ = "0.1.0"

# Passthrough helpers used by core modules (mirrors original cd4py __init__)
from dpu_utils.codeutils import get_language_keywords  # noqa: F401


def log_step(msg: str):
    """Print a clearly delimited pipeline stage header."""
    from rich.console import Console
    Console().print(f"\n[bold magenta]──── {msg} ────[/bold magenta]")


def dummy_preprocessor(x):
    """No-op preprocessor/tokenizer for sklearn's TfidfVectorizer."""
    return x
