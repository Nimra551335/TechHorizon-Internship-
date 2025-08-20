# TechHorizon Internship – Deep Learning Projects (Report)

**Intern:** Nimra Razzaq  
**Track:** Deep Learning (Keras/TensorFlow, Gradio)  
**Date:** August 16, 2025

---

## 1. Approach Overview

**Task 1 (MNIST):** I trained a small fully-connected neural network to classify 28×28 grayscale digit images from the Keras MNIST dataset. Images were normalized to `[0,1]` and flattened to 784 features. The network used two hidden layers (256 and 128 units, ReLU) with dropout (0.2) to reduce overfitting, and a softmax output over 10 classes. The model was optimized with Adam and cross-entropy loss, with a 10% validation split.

**Task 2 (Pre-trained Model):** I used the MobileNetV2 model pre-trained on ImageNet to classify ~10–15 images. Images are resized to 224×224, preprocessed with the model’s canonical transform, and top-1 labels are decoded with `decode_predictions`. The script outputs a grid figure showing each image and its predicted label with probability.

**Task 3 (Optional App):** I built a Gradio app that loads the MNIST model and provides a drawing canvas. The app preprocesses the sketch (grayscale, centered into 28×28, inversion), flattens it, and returns both the predicted digit and class probabilities.

---

## 2. Key Challenges & Solutions

- **Overfitting on MNIST:** Even small models can overfit quickly. I added dropout layers (0.2) and used validation monitoring. For further improvement, I could apply early stopping or L2 regularization.
- **Preprocessing Consistency:** Ensuring the canvas drawing matched the MNIST distribution required inversion and centering. I implemented padding-based centering and normalization to align with MNIST.
- **Handling Varied Images in Task 2:** Real-world images vary in size and aspect ratio. I used Keras preprocessing utilities to resize and batch images consistently.

---

## 3. Results (Representative)

> Exact numbers depend on the random seed and environment; the following are typical.

- **Task 1:** Test accuracy commonly ~**0.97–0.98** after ~8 epochs for this MLP setup. The script saves a grid of predictions (`task1_predictions.png`) and the trained model (`mnist_dense_model.keras`).
- **Task 2:** MobileNetV2 achieves strong top-1 predictions on many everyday objects. The script saves a collage of predictions (`task2_predictions_grid.png`) with labels and probabilities.
- **Task 3:** The Gradio app produces intuitive outputs; digits drawn clearly are correctly classified most of the time. Messy or off-center drawings reduce accuracy; centering steps help mitigate this.

---

## 4. Learnings

- **Model Capacity vs. Data Simplicity:** For MNIST, a small dense network suffices; CNNs would offer even better performance but the exercise solidified fundamentals of dense layers and softmax classifiers.
- **Transfer Learning Practicality:** Using pre-trained models drastically reduces compute needs and setup time while delivering strong baseline accuracy on generic images.
- **Deployment Basics:** Gradio provides a fast path from a trained model to a usable interface, reinforcing the importance of preprocessing consistency between training and inference.

---

## 5. Next Steps (Improvements)

- Add **EarlyStopping** / **ModelCheckpoint** callbacks in Task 1.
- Try a **small CNN** for MNIST to compare with the MLP baseline.
- Expand Task 2 to support **top-5** predictions and **batch inference** folders recursively.
- Containerize the app with **Docker** and add a simple CI workflow for reproducibility.

---

**Files Provided:** `task1_mnist_nn.py`, `task2_pretrained_classification.py`, `task3_digit_app.py`, `requirements.txt`, `REPORT.md`, `README.md`.
