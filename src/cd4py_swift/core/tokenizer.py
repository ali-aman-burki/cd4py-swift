# Originally from CD4Py (MIT-licensed portion) by Davide Giovanelli & Maliheh Izadi
# https://github.com/saltudelft/CD4Py
# Modified: rich progress bars, logging instead of print for errors.

from tokenize import tokenize, NAME, STRING
from typing import Iterator
from dpu_utils.utils import save_jsonl_gz
from joblib import Parallel, delayed
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, MofNCompleteColumn, TimeElapsedColumn
import keyword
import logging
import os
import glob


def tokenize_file(filepath: str, all_tokens: bool = False) -> dict:
    """
    Tokenize a single Python source file.
    Returns {'filename': filepath, 'tokens': [...]}.
    Failed files are logged as warnings and return an empty token list.
    """
    tokens = []
    try:
        with open(filepath, 'rb') as f:
            for toknum, tokval, _, _, _ in tokenize(f.readline):
                if all_tokens or toknum in {NAME, STRING}:
                    if not keyword.iskeyword(tokval):
                        tokens.append(tokval)
    except Exception as e:
        logging.warning('Error tokenizing %s: %s', filepath, e)
    return dict(filename=filepath, tokens=tokens)


def _tokenize_directory(directory: str, output_folder: str, only_ids: bool = False):
    """Tokenize all .py files under directory and write a .jsonl.gz to output_folder."""
    def _generate():
        for file in glob.iglob(os.path.join(directory, '**', '*.py'), recursive=True):
            if not os.path.isdir(file):
                yield tokenize_file(file, only_ids)

    directory_name = os.path.basename(directory)
    save_jsonl_gz(_generate(), os.path.join(output_folder, directory_name + '-tokens.jsonl.gz'))


def tokenize_all_project_folders(directory: str, output_folder: str,
                                  n_jobs: int = -1, only_ids: bool = False):
    """
    Tokenize each project subdirectory of directory in parallel (one job per subdir).
    Progress is shown via a rich progress bar.
    """
    os.makedirs(output_folder, exist_ok=True)
    all_dirs = [
        (os.path.join(directory, d), output_folder, only_ids)
        for d in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, d))
    ]

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        transient=False,
    ) as progress:
        task = progress.add_task("Tokenizing projects", total=len(all_dirs))

        def _run(args):
            _tokenize_directory(*args)
            progress.advance(task)

        Parallel(n_jobs=n_jobs, prefer='threads')(
            delayed(_run)(d) for d in all_dirs
        )
