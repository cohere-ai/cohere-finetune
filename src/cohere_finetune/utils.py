import io
import json
import logging
import numpy as np
import os
import pandas as pd
import pathlib
import pickle
import yaml
from consts import ENVIRONMENT_MODE_KEY
from typing import Any


# Create the logger for cohere_finetune
logging.basicConfig(
    level=logging.INFO if os.environ.get(ENVIRONMENT_MODE_KEY, "PROD") == "DEV" else logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("cohere_finetune")


def get_ext(path: str) -> str:
    """Get the file extension from the path of a file."""
    return pathlib.Path(path).suffix


def get_a_file_of_given_type_from_dir(dir_path: str, file_ext: str) -> str:
    """
    Return the path of the file of a given type from a directory.

    If there are multiple files of the given type, return the first one we found
    If there is no files of the given type, return ""
    """
    files = list(pathlib.Path(dir_path).glob(f"*{file_ext}"))
    if len(files) > 0:
        return str(files[0])
    else:
        return ""


def load_file(path: str, **kwargs) -> Any:
    """
    Load the content of a file, where it can automatically handle various types of files.

    Use this function only when you want to load the content in a "normal" / "common" way;
    if you want to load it in some special way, you need to write your own codes for that.
    """
    if not os.path.exists(path):
        raise FileNotFoundError

    ext = get_ext(path)

    if ext in {".xls", ".xlsx"}:
        x = pd.read_excel(path, index_col=None)
    elif ext in {".csv"}:
        x = pd.read_csv(path, sep=",", index_col=False, encoding="utf-8")
    elif ext in {".tsv"}:
        x = pd.read_csv(path, sep="\t", index_col=False, encoding="utf-8")
    elif ext in {".json"}:
        if kwargs.get("json_file_type", "dict") == "dataframe":
            x = pd.read_json(path, orient="records", typ="frame")
        else:
            with open(path, "r") as f:
                x = json.load(f)
    elif ext in {".jsonl"}:
        with open(path, "r") as f:
            x = [json.loads(line, parse_float=str, parse_int=str) for line in f]
    elif ext in {".txt"}:
        if kwargs.get("txt_file_type", "str") == "list":
            with open(path, "r") as f:
                x = f.read().splitlines()
        else:
            with open(path, "r") as f:
                x = f.read()
    elif ext in {".pickle"}:
        with open(path, "rb") as f:
            x = pickle.load(f)
    elif ext in {".npy"}:
        x = np.load(path)
    elif ext in {".yaml"}:
        with open(path, "r") as f:
            x = yaml.safe_load(f)
    else:
        raise NotImplementedError

    return x


def save_file(x: Any, path: str, overwrite_ok: bool = False) -> None:
    """
    Save an object to a file, where it can automatically handle various types of files.

    Use this function only when you want to save the object in a "normal" / "common" way;
    if you want to save it in some special way, you need to write your own codes for that.
    """
    if not overwrite_ok and os.path.exists(path):
        raise FileExistsError

    if not os.path.isdir(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=False)

    ext = get_ext(path)

    if ext in {".xls", ".xlsx"}:
        assert isinstance(x, pd.DataFrame)
        x.to_excel(path, index=False)
    elif ext in {".csv"}:
        assert isinstance(x, pd.DataFrame)
        x.to_csv(path, sep=",", index=False, encoding="utf-8")
    elif ext in {".tsv"}:
        assert isinstance(x, pd.DataFrame)
        x.to_csv(path, sep="\t", index=False, encoding="utf-8")
    elif ext in {".json"}:
        if isinstance(x, pd.DataFrame):
            x.reset_index(drop=True).to_json(path, orient="records")
        else:
            assert isinstance(x, dict)
            with open(path, "w") as f:
                json.dump(x, f)
    elif ext in {".jsonl"}:
        assert isinstance(x, list)
        with open(path, "w") as f:
            for entry in x:
                assert isinstance(entry, dict)
                json.dump(entry, f)
                f.write("\n")
    elif ext in {".txt"}:
        if isinstance(x, list):
            with open(path, "w") as f:
                for entry in x:
                    assert isinstance(entry, str)
                    f.write(entry)
                    f.write("\n")
        else:
            assert isinstance(x, str)
            with open(path, "w") as f:
                f.write(x)
    elif ext in {".pickle"}:
        with open(path, "wb") as f:
            pickle.dump(x, f)
    elif ext in {".npy"}:
        assert isinstance(x, np.ndarray)
        np.save(path, x)
    elif ext in {".yaml"}:
        assert isinstance(x, dict)
        with open(path, "w") as f:
            yaml.dump(x, f, default_flow_style=False)
    else:
        raise NotImplementedError


def get_lines(data_path: str) -> tuple[list[str], int]:
    """Read the lines from a file, where a line will be dropped if we can't decode it."""
    with open(data_path, "rb") as file_bytes:
        lines = []
        n_dropped_lines = 0
        while line := file_bytes.readline():
            try:
                lines.append(line.decode("utf-8"))
            except UnicodeDecodeError:
                n_dropped_lines += 1
                continue
    return lines, n_dropped_lines


def load_and_prepare_csv(data_path: str, column_types: dict) -> tuple[pd.DataFrame, int]:
    """Load a CSV file as a Pandas dataframe, and do some basic data cleaning."""
    assert get_ext(data_path) == ".csv"

    lines, n_dropped_lines = get_lines(data_path)
    df = pd.read_csv(io.StringIO("".join(lines)), keep_default_na=False)

    df = df.astype({col: dtype for col, dtype in column_types.items() if col in df.columns})
    df = df.loc[:, [col for col in column_types if col in df.columns]]
    df = df.drop_duplicates(ignore_index=True)

    return df, n_dropped_lines
