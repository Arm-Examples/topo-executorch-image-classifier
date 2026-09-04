# Image Classifier (ExecuTorch + XNNPACK)

> This project is a [Topo Project](https://github.com/arm/topo/blob/main/docs/introduction/glossary.md#topo-project) and follows the [Topo Project Specification](https://github.com/arm/topo/tree/main/docs/project-specification).

This Project provides an on-device evaluation harness for Arm-optimized ExecuTorch image classification models that use the XNNPACK backend. Select a compatible `.pte` model, deploy the Project to an Arm Target, and upload images through the web interface. The dashboard reports prediction results and performance for the Target.

The Project demonstrates:

- A multi-stage Docker build that downloads a Hugging Face model and embeds it in the image. The Target does not need the Hugging Face token or network access.
- A configuration-driven CPU inference runner that reads configuration from the selected Hugging Face model repository.
- Per-image latency benchmarking that reports the median and p90 after warmup runs.

## Model compatibility

This Project supports only ExecuTorch image classification models that use the XNNPACK backend and are hosted on Hugging Face. Each model repository must contain `config.yaml` and `metadata.yaml`. These files identify the `.pte` file and configure preprocessing and postprocessing. A repository can also provide human-readable output labels in `imagenet_classes.json`. If it does not and `metadata.yaml` identifies `calibration.dataset_name` as ImageNet-1K, the runtime uses the categories from Torchvision's `SqueezeNet1_1_Weights.IMAGENET1K_V1` metadata. Other models without labels use generic class indices.

## Build-time parameters

The `HF_REPO_ID` and `HF_ENDPOINT` Project parameters are passed to Docker as build arguments and resolved at build time.

| Parameter     | Required | Description                                           | Default                                |
| ------------- | -------- | ----------------------------------------------------- | -------------------------------------- |
| `HF_REPO_ID`  | No       | ExecuTorch + XNNPACK model repository on Hugging Face | `Arm/vit-base-int8-xnnpack-executorch` |
| `HF_ENDPOINT` | No       | Hugging Face API endpoint                             | `https://huggingface.co`               |

## Usage

Install Topo by following the instructions in the [Topo repository](https://github.com/arm/topo).

### Clone the Project

The clone step will prompt you for values for the `HF_REPO_ID` and `HF_ENDPOINT` parameters. Leave either input empty to select its default.

```bash
topo clone https://github.com/Arm-Examples/topo-executorch-image-classifier.git
```

### Build and deploy the Project

Set a Hugging Face read token on the Host, and deploy the Project to the Target:

```bash
cd topo-executorch-image-classifier
topo deploy --target <user@hostname>
```

Topo builds the image on the Host and transfers the finished image to the Target over SSH. The Target does not need network access to download the model.

> **Note:** To download a private model at build time, set `HF_TOKEN` on the Host before running `topo deploy`. The token must have read access to the repository. The build mounts it as a secret and does not store it in the image or transfer it to the Target. Public repositories do not require a token.

### Open the web interface

After deployment, open `http://<target-ip>:7860` in a browser. Upload an image to see the top predicted classes, confidence scores, and inference latency for the Target.
