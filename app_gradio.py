#!/usr/bin/env python3
"""Minimal Gradio UI: left upload, right result with boxes.

Usage: python3 app_gradio.py
"""
import os
import glob
from typing import Optional
from PIL import Image
import numpy as np
import gradio as gr

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


MODEL_CACHE = {}


def find_latest_model(project_root: Optional[str] = None) -> Optional[str]:
    if project_root is None:
        project_root = os.path.dirname(os.path.abspath(__file__))
    pattern = os.path.join(project_root, 'runs', '*', 'weights', 'best.pt')
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def load_model(model_path: str):
    if model_path in MODEL_CACHE:
        return MODEL_CACHE[model_path]
    if YOLO is None:
        raise RuntimeError('ultralytics not installed')
    model = YOLO(model_path)
    MODEL_CACHE[model_path] = model
    return model


def predict_image(image: Image.Image, model_path: str = '', device: str = 'cpu', imgsz: int = 640, conf: float = 0.25):
    if image is None:
        return None
    if not model_path or not os.path.exists(model_path):
        model_path = find_latest_model()
        if model_path is None:
            return 'No model found. Place runs/*/weights/best.pt or provide model path.'
    model = load_model(model_path)
    arr = np.array(image.convert('RGB'))
    results = model.predict(source=arr, device=device, imgsz=imgsz, conf=conf, save=False)
    if not results:
        return image
    # results[0].plot() returns numpy array with boxes drawn
    rendered = results[0].plot()
    return Image.fromarray(rendered)


def build_ui():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown('### Upload')
                inp = gr.Image(type='pil', label='Upload image')
                model_input = gr.Textbox(label='Model path (optional)', value=find_latest_model() or '')
                device = gr.Dropdown(['cpu','0','1','2','3'], value='cpu', label='Device')
                imgsz = gr.Slider(320, 1280, value=640, step=32, label='Image size')
                conf = gr.Slider(0.01, 1.0, value=0.25, step=0.01, label='Confidence')
                btn = gr.Button('Run')
            with gr.Column(scale=1):
                gr.Markdown('### Result')
                out = gr.Image(label='Detection')

        btn.click(fn=predict_image, inputs=[inp, model_input, device, imgsz, conf], outputs=[out])
    return demo


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=7861, help='server port')
    args = p.parse_args()

    demo = build_ui()
    demo.launch(server_name='0.0.0.0', server_port=args.port)
