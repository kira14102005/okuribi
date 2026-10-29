import json
from pathlib import Path


def json_reader(path: Path) -> dict:
    try:
        with Path.open(path, "r") as file:
            loaded_data = json.load(file)

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Path: {path}") from e

    return loaded_data


def json_writer(path: Path, data: dict):
    try:
        with Path.open(path, "w+") as file:
            json.dump(data, file, indent=4)
    except Exception as e:
        raise Exception(f"Path: {path}") from e
