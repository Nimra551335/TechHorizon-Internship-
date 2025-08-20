#!/usr/bin/env python3
"""
Task 3 (Optional): Digit Drawing Recognition Web App using Gradio
- Loads the trained MNIST model from Task 1 (mnist_dense_model.keras)
- Provides a canvas for drawing a digit
- Returns the predicted digit with probabilities
Run:
    python task3_digit_app.py
Then open the printed Gradio URL in your browser.
"""
import numpy as np
import gradio as gr
from tensorflow import keras
import cv2

MODEL_PATH = "mnist_dense_model.keras"

def preprocess(img):
    """
    Convert canvas (H,W,3) uint8 image to 28x28 grayscale [0,1].
    """
    if img is None:
        return np.zeros((28,28), dtype=np.float32)
    # Convert to grayscale and invert (canvas is white background, black drawing)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Centering and resizing to 28x28
    # Resize with padding to keep aspect
    h, w = gray.shape
    scale = 20.0 / max(h, w)
    resized = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    canvas = np.full((28, 28), 255, dtype=np.uint8)
    y_off = (28 - resized.shape[0]) // 2
    x_off = (28 - resized.shape[1]) // 2
    canvas[y_off:y_off+resized.shape[0], x_off:x_off+resized.shape[1]] = resized
    # Normalize: MNIST digits are dark=0, here drawing is dark, so invert
    norm = 1.0 - (canvas.astype("float32") / 255.0)
    return norm

def predict_digit(img):
    model = keras.models.load_model(MODEL_PATH)
    x = preprocess(img).reshape(1, -1)  # flatten to (1, 784)
    preds = model.predict(x, verbose=0)[0]
    top = int(np.argmax(preds))
    # Build a {class: prob} dict for Gradio
    probs = {str(i): float(p) for i, p in enumerate(preds)}
    return top, probs

def clear_canvas():
    return None

with gr.Blocks() as demo:
    gr.Markdown("# MNIST Digit Recognizer (Gradio)")
    gr.Markdown("Draw a digit (0–9) in the canvas, then click **Predict**.")
    with gr.Row():
        canvas = gr.ImageEditor(sources=["canvas"], type="numpy", image_mode="RGB", width=280, height=280)
        with gr.Column():
            btn = gr.Button("Predict")
            clear = gr.Button("Clear")
            label = gr.Label(num_top_classes=10)
            number = gr.Number(label="Predicted Digit", precision=0)
    btn.click(predict_digit, inputs=[canvas], outputs=[number, label])
    clear.click(fn=clear_canvas, inputs=None, outputs=[canvas])

if __name__ == "__main__":
    demo.launch()
