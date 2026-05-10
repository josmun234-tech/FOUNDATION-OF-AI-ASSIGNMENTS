"""
MNIST Digit Classifier (0–9)
CNN with TensorFlow/Keras — targets ~99%+ test accuracy
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ── 1. Load & preprocess ───────────────────────────────────────────────────────
print("Loading MNIST dataset...")
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalise to [0, 1] and add channel dimension → (N, 28, 28, 1)
X_train = X_train.astype("float32") / 255.0
X_test  = X_test .astype("float32") / 255.0
X_train = X_train[..., np.newaxis]
X_test  = X_test [..., np.newaxis]

y_train_ohe = to_categorical(y_train, 10)
y_test_ohe  = to_categorical(y_test,  10)

print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

# ── 2. Build CNN model ─────────────────────────────────────────────────────────
def build_model() -> models.Model:
    m = models.Sequential([
        # ── Conv block 1 ──────────────────────────────────────────────────────
        layers.Conv2D(32, 3, activation="relu", padding="same",
                      input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),

        # ── Conv block 2 ──────────────────────────────────────────────────────
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),

        # ── Dense head ────────────────────────────────────────────────────────
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),   # one node per digit class
    ])

    m.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return m

model = build_model()
model.summary()

# ── 3. Train ───────────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
]

history = model.fit(
    X_train, y_train_ohe,
    epochs=20,
    batch_size=128,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1,
)

# ── 4. Evaluate ────────────────────────────────────────────────────────────────
test_loss, test_acc = model.evaluate(X_test, y_test_ohe, verbose=0)
print(f"\nTest accuracy : {test_acc * 100:.2f}%")
print(f"Test loss     : {test_loss:.4f}")

y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

# ── 5. Visualisations ──────────────────────────────────────────────────────────
OUT = "."   # change to your preferred output directory

# (a) Training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Training History", fontsize=14, fontweight="bold")
for ax, metric in zip(axes, ["accuracy", "loss"]):
    ax.plot(history.history[metric],          label="Train")
    ax.plot(history.history[f"val_{metric}"], label="Validation")
    ax.set_title(metric.capitalize())
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/training_history.png", dpi=150)
plt.close()

# (b) Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=range(10), yticklabels=range(10), ax=ax)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title(f"Confusion Matrix  (Test Acc = {test_acc * 100:.2f}%)")
plt.tight_layout()
plt.savefig(f"{OUT}/confusion_matrix.png", dpi=150)
plt.close()

# (c) Sample predictions grid (10 correct  +  10 wrong)
wrong_idx   = np.where(y_pred != y_test)[0]
correct_idx = np.where(y_pred == y_test)[0]

fig = plt.figure(figsize=(14, 6))
fig.suptitle("Sample Predictions", fontsize=13, fontweight="bold")
gs  = gridspec.GridSpec(2, 10, figure=fig, hspace=0.6)

for col, idx in enumerate(correct_idx[:10]):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(X_test[idx].squeeze(), cmap="gray")
    ax.set_title(f"✓{y_pred[idx]}", color="green", fontsize=9)
    ax.axis("off")

for col, idx in enumerate(wrong_idx[:10]):
    ax = fig.add_subplot(gs[1, col])
    ax.imshow(X_test[idx].squeeze(), cmap="gray")
    ax.set_title(f"T{y_test[idx]}/P{y_pred[idx]}", color="red", fontsize=8)
    ax.axis("off")

fig.text(0.01, 0.73, "Correct", va="center", rotation=90,
         fontsize=10, color="green", fontweight="bold")
fig.text(0.01, 0.28, "Wrong",   va="center", rotation=90,
         fontsize=10, color="red",   fontweight="bold")
plt.savefig(f"{OUT}/sample_predictions.png", dpi=150, bbox_inches="tight")
plt.close()

# (d) Per-class accuracy bar chart
pca = cm.diagonal() / cm.sum(axis=1)
fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.bar(range(10), pca * 100,
              color=plt.cm.RdYlGn(pca), edgecolor="black", linewidth=0.5)
ax.set_xticks(range(10))
ax.set_xticklabels([f"Digit {i}" for i in range(10)], rotation=30, ha="right")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Per-Class Accuracy")
ax.set_ylim(94, 100.5)
ax.grid(axis="y", alpha=0.3)
for bar, a in zip(bars, pca):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f"{a * 100:.1f}%", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/per_class_accuracy.png", dpi=150)
plt.close()

# ── 6. Save model ──────────────────────────────────────────────────────────────
model.save(f"{OUT}/mnist_cnn_model.keras")
print("\nAll outputs saved.  Done ✓")