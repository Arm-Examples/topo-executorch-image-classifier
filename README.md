# Image Classifier (ExecuTorch)

> This project is a [Topo](https://github.com/arm/topo) template and follows the [Topo Project Specification](https://github.com/arm/topo/tree/main/docs/project-specification).

An on-device evaluation harness for Arm-optimised image-classification models. Supply an ExecuTorch `.pte` model, deploy it to your own Arm device, upload your own images through a web UI and evaluate its performance on your real hardware.

It demonstrates:

- A multi-stage Docker build that bakes a Hugging Face model into the image at build time (the image ships self-contained, the device needs no token or network).
- A config-driven CPU inference runner that sources config from the selected Hugging Face model repository to enable seamless model switching.
- Per-image latency benchmarking (median / p90 over repeated runs after warmup) surfaced in an interactive web dashboard.

## Model Compatibility

Only Executorch image classification models on Hugging Face are supported by this project. Furthermore, model repositories must contain a `config.yaml` and `metadata.yaml` that define which `.pte` file to use, pre/post-processing steps and more. Optionally, repositories can also specify human-readable output labels in `imagenet_classes.json`.

Working examples include:

- `Arm/deit-tiny-int8-xnnpack-executorch`
- `Arm/googlenet-int8-xnnpack-executorch-graviton-g4`
- `Arm/googlenet-int8-xnnpack-executorch-raspberrypi5`
- `Arm/inception-v3-int8-xnnpack-executorch-graviton-g4`
- `Arm/inception-v3-int8-xnnpack-executorch-raspberrypi5`
- `Arm/mobilenet-v3-small-int8-xnnpack-executorch`
- `Arm/resnet-18-int8-xnnpack-executorch`
- `Arm/resnet-50-int8-xnnpack-executorch`
- `Arm/squeezenet-1-1-int8-xnnpack-executorch`
- `Arm/swin-tiny-int8-xnnpack-executorch`
- `Arm/vit-base-int8-xnnpack-executorch`

## Build-Time Parameters

The model identity is a Docker build argument (`x-topo.args` in `compose.yaml`), resolved at build time. There is no default—you must supply the Hugging Face repository ID. Its `metadata.yaml` identifies the `.pte` file to download.

| Parameter | Required | Description                   | Example                                |
| --------- | -------- | ----------------------------- | -------------------------------------- |
| `MODEL`   | yes      | Hugging Face model repository | `Arm/vit-base-int8-xnnpack-executorch` |

## Usage

The easiest way to deploy is using `topo`. Download and install `topo` from [arm/topo](https://github.com/arm/topo).

### Clone the project:

```bash
topo clone git@github.com:Arm-Examples/topo-image-classifier.git
```

You will be prompted for the model repository - there is no default, so you must provide `MODEL`.

### Build and Deploy the project:

```bash
cd topo-image-classifier
export HF_TOKEN=<your-hf-read-token>
topo deploy --target <ip-address-of-target>
```

topo builds the image **on your machine** (where the token lives) and ships the finished image to the device over SSH. The target needs neither the token nor network access for the model.

### What you will see

Once deployment completes, open a browser to `http://<ip-address-of-target>:7860`. Upload an image to get the top-k predicted classes, each with a confidence score, plus the measured inference latency (median / p90 over repeated runs after a warmup) for that target.
