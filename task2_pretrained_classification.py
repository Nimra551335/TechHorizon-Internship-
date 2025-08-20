#!/usr/bin/env python3
"""
Task 2: Image Classification using a Pre-trained Model (MobileNetV2)
- Reads 10–15 images from ./sample_images
- Resizes & preprocesses
- Predicts with ImageNet labels
- Saves a grid with top-1 predictions

Run:
    python task2_pretrained_classification.py
Notes:
    Put JPG/PNG images into ./sample_images before running.
"""
import os, glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

def load_images_from_folder(folder, max_images=15, target_size=(224,224)):
    paths = []
    for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"):
        paths.extend(glob.glob(os.path.join(folder, ext)))
    paths = paths[:max_images]
    imgs, raw_imgs = [], []
    for p in paths:
        img = image.load_img(p, target_size=target_size)
        arr = image.img_to_array(img)
        raw_imgs.append(np.array(img)/255.0)
        imgs.append(arr)
    return np.array(imgs), np.array(raw_imgs), paths

def main():
    folder = "sample_images"
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder '{folder}' not found. Create it and add some images.")
    x, raw, paths = load_images_from_folder(folder)
    if len(paths) == 0:
        raise RuntimeError("No images found in ./sample_images. Add a few JPG/PNG files and retry.")

    model = MobileNetV2(weights="imagenet")
    x_pp = preprocess_input(x.copy())
    preds = model.predict(x_pp, verbose=0)
    decoded = decode_predictions(preds, top=1)

    # Build a grid
    n = len(paths)
    cols = min(5, n)
    rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(3*cols, 3*rows))
    for i in range(n):
        ax = plt.subplot(rows, cols, i+1)
        ax.imshow(raw[i])
        label = decoded[i][0][1].replace('_',' ')
        prob  = decoded[i][0][2]
        base  = os.path.basename(paths[i])
        ax.set_title(f"{label} ({prob:.2f})\n{base}", fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    out_path = "task2_predictions_grid.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved predictions grid to {out_path}")

if __name__ == "__main__":
    main()
