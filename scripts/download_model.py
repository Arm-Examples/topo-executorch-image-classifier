import os
import sys
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download


def main() -> None:
    repo = os.getenv("MODEL") or sys.exit(
        "error: MODEL must be a Hugging Face repo ID, for example 'Arm/vit-base-int8-xnnpack-executorch'"
    )

    def download(filename: str, required: bool = True) -> str | None:
        try:
            return hf_hub_download(repo, filename, local_dir="model")
        except Exception as error:
            message = f"could not download {'required ' if required else ''}'{filename}' from '{repo}': {error}"
            if required:
                raise SystemExit(f"error: {message}") from None
            print(f"warning: {message}", file=sys.stderr)
            return None

    metadata = download("metadata.yaml")
    assert metadata
    try:
        model = yaml.safe_load(Path(metadata).read_text())["filename"]
    except (KeyError, TypeError, yaml.YAMLError):
        sys.exit("error: metadata.yaml must define filename")
    if not isinstance(model, str) or not model.endswith(".pte"):
        sys.exit("error: filename in metadata.yaml must name a .pte file")

    download(model)
    download("config.yaml")
    download("imagenet_classes.json", required=False)
    download("sample_input.jpg", required=False)


if __name__ == "__main__":
    main()
