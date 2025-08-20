#!/usr/bin/env python3
"""
Task 1: Simple Neural Network to classify handwritten digits (MNIST)
- Loads Keras MNIST dataset
- Normalizes and flattens images
- Builds a 3-layer Dense network
- Trains & evaluates
- Displays a few predictions
Run:
    python task1_mnist_nn.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

def build_model(input_dim=784, num_classes=10):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def main():
    # 1) Load data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # 2) Normalize and flatten
    x_train = x_train.astype("float32")/255.0
    x_test  = x_test.astype("float32")/255.0
    x_train = x_train.reshape((x_train.shape[0], -1))  # (60000, 784)
    x_test  = x_test.reshape((x_test.shape[0], -1))    # (10000, 784)

    # 3) Build & train
    model = build_model(input_dim=x_train.shape[1], num_classes=10)
    history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=8,
        batch_size=128,
        verbose=2
    )

    # 4) Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

    # 5) Show a few predictions
    idx = np.random.choice(len(x_test), size=12, replace=False)
    imgs = x_test[idx]
    labels = y_test[idx]
    preds = np.argmax(model.predict(imgs, verbose=0), axis=1)

    # reshape for display
    imgs_disp = imgs.reshape((-1, 28, 28))

    fig = plt.figure(figsize=(10, 5))
    for i in range(12):
        ax = plt.subplot(3, 4, i+1)
        ax.imshow(imgs_disp[i], cmap="gray")
        ax.set_title(f"Pred: {preds[i]} | True: {labels[i]}")
        ax.axis("off")
    fig.suptitle(f"MNIST Predictions | Test Acc: {test_acc:.4f}")
    plt.tight_layout()
    # Save plot
    out_path = "task1_predictions.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved prediction grid to {out_path}")

    # Save model
    model.save("mnist_dense_model.keras")
    print("Saved trained model to mnist_dense_model.keras")

if __name__ == "__main__":
    main()
