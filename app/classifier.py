"""Config-driven image classification for ExecuTorch models."""

import json
from pathlib import Path
from time import perf_counter

import torch
import yaml
from executorch.runtime import Runtime
from PIL import Image
from torchvision import transforms


_TRANSFORMS = {
    "resize": transforms.Resize,
    "center_crop": transforms.CenterCrop,
    "to_tensor": lambda _: transforms.ToTensor(),
    "normalize": lambda options: transforms.Normalize(**options),
}


def _load_yaml(path: Path) -> dict:
    with path.open() as file:
        return yaml.safe_load(file)


class Preprocessor:
    """Convert an input image into a batched model input tensor."""

    def __init__(self, steps: list):
        pipeline = []
        for step in steps:
            name, options = (
                (step, None) if isinstance(step, str) else next(iter(step.items()))
            )
            try:
                factory = _TRANSFORMS[name]
            except KeyError:
                raise ValueError(
                    f"unsupported preprocessing operation: {name}"
                ) from None
            pipeline.append(factory(options))
        self.pipeline = transforms.Compose(pipeline)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.pipeline(image.convert("RGB")).unsqueeze(0)


class Postprocessor:
    """Convert model output scores into labelled top-k predictions."""

    def __init__(self, options: dict, labels: dict | list):
        self.apply_softmax = options["softmax"]
        self.top_k = options["top_k"]
        self.labels = labels

    def _label(self, index: int) -> str:
        if isinstance(self.labels, dict):
            return self.labels.get(str(index), f"class_{index}")
        return self.labels[index] if index < len(self.labels) else f"class_{index}"

    def __call__(self, output) -> dict[str, float]:
        scores = torch.as_tensor(output, dtype=torch.float32).reshape(-1)
        if self.apply_softmax:
            scores = torch.softmax(scores, dim=-1)
        values, indices = scores.topk(min(self.top_k, scores.numel()))
        return {
            self._label(index): value
            for value, index in zip(values.tolist(), indices.tolist(), strict=True)
        }


class Classifier:
    """Load one model directory and expose reusable classification operations."""

    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        config = _load_yaml(model_dir / "config.yaml")
        self._preprocessor = Preprocessor(config["input"]["preprocessing"])

        labels_path = model_dir / "imagenet_classes.json"
        if labels_path.is_file():
            with labels_path.open() as file:
                labels = json.load(file)
        else:
            labels = {}
        self._postprocessor = Postprocessor(
            config["output"]["postprocessing"], labels
        )
        self.top_k = self._postprocessor.top_k

        sample = model_dir / "sample_input.jpg"
        self.sample_image = str(sample) if sample.is_file() else None

        metadata = _load_yaml(model_dir / "metadata.yaml")
        filename = metadata["filename"]
        model_path = (model_dir / filename).resolve()
        program = Runtime.get().load_program(str(model_path))
        self.method = program.load_method("forward")

    def classify(
        self, image: Image.Image | None, runs: int = 20, warmup: int = 3
    ) -> tuple[dict[str, float], str]:
        """Classify an image and return predictions with a latency summary."""
        if image is None:
            return {}, "_Upload an image to classify._"

        started = perf_counter()
        input_tensor = self._preprocessor(image)
        preprocess_ms = (perf_counter() - started) * 1000

        for _ in range(max(0, warmup)):
            self.method.execute([input_tensor])

        latencies = []
        for _ in range(max(1, runs)):
            started = perf_counter()
            output = self.method.execute([input_tensor])[0]
            latencies.append((perf_counter() - started) * 1000)

        latency = torch.tensor(latencies, dtype=torch.float64)
        p50_ms = float(torch.quantile(latency, 0.5))
        return self._postprocessor(output), (
            f"**{p50_ms:.1f} ms** median inference "
            f"&nbsp;·&nbsp; **{1000.0 / p50_ms if p50_ms > 0 else 0.0:.0f}** img/s\n\n"
            f"<sub>p90 {float(torch.quantile(latency, 0.9)):.1f} ms · "
            f"best {float(latency.min()):.1f} ms · "
            f"preprocess {preprocess_ms:.1f} ms · {len(latencies)} runs, "
            "measured on this device (CPU)</sub>"
        )
