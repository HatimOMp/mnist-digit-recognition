import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model import build_model

# ── Load and preprocess data ─────────────────────────────────────────
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize to [0, 1] and add channel dimension
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0
x_train = x_train[..., np.newaxis]
x_test  = x_test[..., np.newaxis]

print(f"Training samples : {len(x_train)}")
print(f"Test samples     : {len(x_test)}")

# ── Build and train ──────────────────────────────────────────────────
model = build_model()
model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=5, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", patience=3, factor=0.5
    )
]

history = model.fit(
    x_train, y_train,
    epochs=30,
    batch_size=128,
    validation_split=0.1,
    callbacks=callbacks
)

# ── Evaluate ─────────────────────────────────────────────────────────
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest accuracy : {test_acc * 100:.2f}%")
print(f"Test loss     : {test_loss:.4f}")

# ── Save model ───────────────────────────────────────────────────────
model.save("mnist_model.keras")
print("Model saved to mnist_model.keras")

# ── Plot training history ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history["accuracy"], label="Train")
axes[0].plot(history.history["val_accuracy"], label="Validation")
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(history.history["loss"], label="Train")
axes[1].plot(history.history["val_loss"], label="Validation")
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].legend()

plt.tight_layout()
plt.savefig("training_history.png", dpi=150)
plt.show()
print("Training history saved to training_history.png")

# ── Save sample predictions ───────────────────────────────────────────
predictions = model.predict(x_test[:16])
predicted_labels = np.argmax(predictions, axis=1)

fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(x_test[i].squeeze(), cmap="gray")
    color = "green" if predicted_labels[i] == y_test[i] else "red"
    ax.set_title(f"Pred: {predicted_labels[i]} | True: {y_test[i]}", color=color)
    ax.axis("off")

plt.suptitle("Sample Predictions (green=correct, red=wrong)", fontsize=12)
plt.tight_layout()
plt.savefig("sample_predictions.png", dpi=150)
plt.show()
print("Sample predictions saved to sample_predictions.png")