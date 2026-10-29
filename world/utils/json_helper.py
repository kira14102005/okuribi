import argparse
import json
from pathlib import Path


def json_reader(path: Path, key: str | None = None) -> dict:
    try:
        with Path.open(path, "r") as file:
            loaded_data = json.load(file)

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Path: {path}") from e

    keyed_data = loaded_data if not key else loaded_data.get(key, None)
    return keyed_data


def json_writer(path: Path, data: dict):
    try:
        with Path.open(path, "w+") as file:
            json.dump(data, file, indent=4)
    except Exception as e:
        raise Exception(f"Path: {path}") from e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JSON Helper")
    parser.add_argument("--read", type=str, help="Path to the JSON file to read")
    parser.add_argument("--write", type=str, help="Path to the JSON file to write")
    parser.add_argument(
        "--key",
        type=str,
        help="Key to extract from the JSON file (optional)",
    )
    args = parser.parse_args()
    data = json_reader(Path(args.read), key=args.key)
    json_writer(Path(args.write), data)
    print(f"Data read from {args.read} and written to {args.write}")
