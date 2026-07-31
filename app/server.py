"""Serve the image classifier through Gradio."""

import gradio as gr

import branding
from classifier import Classifier

MODEL_DIR = "model"


def build_demo(classifier: Classifier) -> gr.Interface:
    return gr.Interface(
        fn=classifier.classify,
        inputs=gr.Image(type="pil", label="Image", sources=["upload", "clipboard"]),
        outputs=[
            gr.Label(num_top_classes=classifier.top_k, label="Predictions"),
            gr.Markdown(),
        ],
        examples=[classifier.sample_image] if classifier.sample_image else None,
        title="Edge Image Classifier",
        description="Upload an image to classify it and see the top predictions.",
        submit_btn="Classify",
        clear_btn=None,
        flagging_mode="never",
    )


def main() -> None:
    demo = build_demo(Classifier(MODEL_DIR))
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=branding.theme(),
    )


if __name__ == "__main__":
    main()
