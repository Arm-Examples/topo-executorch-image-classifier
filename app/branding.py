"""Arm-branded Gradio colour and typography."""

import gradio as gr

ARM = {
    "primary": "#0066CC",
    "primary_hover": "#0052A3",
    "text": "#333333",
    "background": "#FFFFFF",
}

FONT_STACK = [
    "Aeonik",
    "system-ui",
    "-apple-system",
    "Segoe UI",
    "Roboto",
    "Helvetica Neue",
    "Arial",
    "sans-serif",
]


def theme() -> gr.Theme:
    """Return a Gradio theme with Arm colours and typography."""
    return gr.themes.Base(primary_hue="blue", font=FONT_STACK).set(
        button_primary_background_fill=ARM["primary"],
        button_primary_background_fill_hover=ARM["primary_hover"],
        button_primary_text_color=ARM["background"],
        body_text_color=ARM["text"],
        block_title_text_color=ARM["primary"],
        block_label_text_color=ARM["text"],
    )
