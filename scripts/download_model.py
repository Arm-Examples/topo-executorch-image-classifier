import os
import shutil
import tempfile
import subprocess
import json
from pathlib import Path
from urllib import request

import yaml

MAX_CONCURRENT_FILES = 4
MAX_CONN_PER_SERVER = 16
CONN_PER_FILE = 16
MIN_SPLIT_SIZE = "1M"


def require_env(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        raise ValueError(f"{var_name} environment variable is not set")
    return value


def download_file(
    endpoint: str, repo_id: str, file_path: str, out_dir: Path, token: str | None
) -> Path:
    url = f"{endpoint}/{repo_id}/resolve/main/{file_path}"
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    out_file_path = out_dir / file_path
    out_file_path.parent.mkdir(parents=True, exist_ok=True)
    with request.urlopen(request.Request(url, headers=headers)) as response:
        with open(out_file_path, "wb") as out_file:
            shutil.copyfileobj(response, out_file)
    return out_file_path


def list_files(endpoint: str, repo_id: str, token: str | None) -> list[str]:
    url = f"{endpoint}/api/models/{repo_id}/tree/main?limit=1000&recursive=true"
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with request.urlopen(request.Request(url, headers=headers)) as response:
        data = json.load(response)
    return [item["path"] for item in data if item["type"] == "file"]


def filter_files(
    repo_files: list[str], required: list[str], optional: list[str]
) -> list[str]:
    files = [*required]
    for file in required:
        if file not in repo_files:
            raise ValueError(
                f"error: required file {file} is missing from the repository"
            )

    for file in optional:
        if file not in repo_files:
            print(f"warning: optional file {file} is missing from the repository")
        else:
            files.append(file)

    return files


def download_batch(
    endpoint: str, repo_id: str, files: list[str], out_dir: Path, token: str | None
) -> None:
    if shutil.which("aria2c") is None:
        raise RuntimeError("aria2c is required but was not found on PATH")

    input_file = tempfile.NamedTemporaryFile()
    for file in files:
        input_file.write(f"{endpoint}/{repo_id}/resolve/main/{file}\n".encode())
        input_file.write(f"  out={file}\n".encode())
    input_file.flush()

    cmd = [
        "aria2c",
        "-x",
        str(MAX_CONN_PER_SERVER),
        "-s",
        str(CONN_PER_FILE),
        "-k",
        MIN_SPLIT_SIZE,
        "-j",
        str(MAX_CONCURRENT_FILES),
        "-i",
        input_file.name,
        "-d",
        str(out_dir),
    ]

    if token:
        cmd.extend(["--header", f"Authorization: Bearer {token}"])

    subprocess.run(cmd, check=True)


def main() -> None:
    repo_id = require_env("HF_REPO_ID")
    endpoint = require_env("HF_ENDPOINT").rstrip("/")
    token = os.getenv("HF_TOKEN")

    out_dir = Path("./model")
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list_files(endpoint, repo_id, token)
    if "metadata.yaml" not in files:
        raise ValueError("error: metadata.yaml is missing from the repository")

    metadata = download_file(endpoint, repo_id, "metadata.yaml", out_dir, token)
    try:
        model = yaml.safe_load(metadata.read_text())["filename"]
    except (KeyError, TypeError, yaml.YAMLError):
        raise ValueError("error: metadata.yaml must define filename")

    filtered_files = filter_files(
        files,
        required=[model, "config.yaml"],
        optional=["imagenet_classes.json", "sample_input.jpg"],
    )
    download_batch(endpoint, repo_id, filtered_files, out_dir, token)


if __name__ == "__main__":
    main()
