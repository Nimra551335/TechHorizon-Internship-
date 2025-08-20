# TechHorizon – Deep Learning Internship (Code + Report)

This folder contains completed code for all tasks and a short report.

## Contents
- `task1_mnist_nn.py` — Train & evaluate a simple MNIST classifier and save predictions + model.
- `task2_pretrained_classification.py` — Classify your own images with MobileNetV2 and save a predictions grid.
- `task3_digit_app.py` — (Optional) Gradio web app to draw digits and see predictions in real-time.
- `sample_images/` — Put 10–15 JPG/PNG files here for Task 2.
- `requirements.txt` — Python dependencies.
- `REPORT.md` — 1–2 page internship report.

## Quickstart
```bash
# 1) Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Task 1: Train MNIST model
python task1_mnist_nn.py

# 4) Task 2: Copy 10–15 images into ./sample_images and run
python task2_pretrained_classification.py

# 5) Task 3 (optional): Start the Gradio app (requires model from Task 1 in the same folder)
python task3_digit_app.py
```

### Outputs
- Task 1: `task1_predictions.png`, `mnist_dense_model.keras`
- Task 2: `task2_predictions_grid.png`
- Task 3: Gradio will print a local URL to open the app.

