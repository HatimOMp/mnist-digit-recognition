import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import io
import os
from pipeline import DigitExtractionPipeline

st.set_page_config(
    page_title="Document Digit Extractor",
    page_icon="🔢",
    layout="wide"
)

st.title("🔢 Document Digit Extractor")
st.markdown(
    "An end-to-end pipeline combining **OpenCV** (computer vision) and a "
    "**CNN trained from scratch** (99.22% accuracy on MNIST) to detect and "
    "extract digits from document images."
)

# ── Load pipeline ────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    return DigitExtractionPipeline("mnist_model.keras")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_model.keras")

try:
    pipeline = load_pipeline()
    model = load_model()
    st.success("✅ Model loaded — Test accuracy: 99.22% on MNIST")
except Exception as e:
    st.error(f"❌ Could not load model: {e}")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🧪 Model Demo (Synthetic)",
    "📄 Document Upload (Real World)",
    "📊 Training Results"
])

# ════════════════════════════════════════════════════════════════════
# TAB 1 — Synthetic demo: model performs at 99.22%
# ════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("🧪 CNN Performance on Clean Digit Images")
    st.markdown(
        "This tab demonstrates the CNN classifying **clean isolated digits** — "
        "the condition it was trained for. Accuracy: **99.22%** on 10,000 test samples."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Test on MNIST samples")
        if st.button("🎲 Run on random MNIST test samples", type="primary"):
            # Load MNIST test set
            (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            x_test = x_test.astype("float32") / 255.0

            # Pick 16 random samples
            indices = np.random.choice(len(x_test), 16, replace=False)
            samples = x_test[indices]
            labels = y_test[indices]

            # Predict
            inputs = samples[..., np.newaxis]
            predictions = model.predict(inputs, verbose=0)
            predicted = np.argmax(predictions, axis=1)
            confidences = np.max(predictions, axis=1)

            # Display
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(4, 4, figsize=(8, 8))
            correct = 0
            for i, ax in enumerate(axes.flat):
                ax.imshow(samples[i], cmap="gray")
                is_correct = predicted[i] == labels[i]
                if is_correct:
                    correct += 1
                color = "green" if is_correct else "red"
                ax.set_title(
                    f"Pred: {predicted[i]} ({confidences[i]:.0%})\nTrue: {labels[i]}",
                    color=color, fontsize=8
                )
                ax.axis("off")
            plt.suptitle(f"Results: {correct}/16 correct", fontsize=13)
            plt.tight_layout()
            st.pyplot(fig)
            st.metric("Accuracy on this batch", f"{correct}/16 = {correct/16:.0%}")

    with col2:
        st.markdown("### Draw your own digit")
        st.markdown("Enter a sequence of digits to classify:")
        user_input = st.text_input("Type digits (e.g. 1234567890)", "1234567890")

        if st.button("🔍 Classify", type="primary"):
            if user_input.strip():
                digits = [c for c in user_input if c.isdigit()]
                if digits:
                    # Create synthetic digit images using MNIST style
                    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
                    x_test = x_test.astype("float32") / 255.0

                    fig, axes = plt.subplots(1, len(digits), figsize=(2 * len(digits), 2))
                    if len(digits) == 1:
                        axes = [axes]

                    for i, digit_char in enumerate(digits):
                        target = int(digit_char)
                        matches = np.where(y_test == target)[0]
                        sample = x_test[np.random.choice(matches)]

                        pred = model.predict(
                            sample[np.newaxis, ..., np.newaxis], verbose=0
                        )[0]
                        predicted_label = np.argmax(pred)
                        conf = pred[predicted_label]

                        axes[i].imshow(sample, cmap="gray")
                        color = "green" if predicted_label == target else "red"
                        axes[i].set_title(f"{predicted_label}\n{conf:.0%}", color=color, fontsize=9)
                        axes[i].axis("off")

                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.warning("Please enter at least one digit.")

# ════════════════════════════════════════════════════════════════════
# TAB 2 — Real document upload
# ════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📄 Real Document Upload")

    st.warning(
        "⚠️ **Domain Shift Notice:** This model was trained on clean, isolated, "
        "handwritten MNIST digits. Real document photos contain printed fonts, "
        "mixed text, perspective distortion, and background noise — conditions "
        "very different from training data. Results on real documents will be imperfect. "
        "See the README for a full discussion of this limitation."
    )

    st.sidebar.title("⚙️ Settings")
    confidence_threshold = st.sidebar.slider(
        "Confidence threshold", 0.5, 1.0, 0.7, 0.05
    )
    pipeline.confidence_threshold = confidence_threshold
    show_steps = st.sidebar.checkbox("Show preprocessing steps", value=True)

    uploaded_file = st.file_uploader(
        "Upload document image",
        type=["png", "jpg", "jpeg"],
        help="Best results on high-contrast printed digit-only images"
    )

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        st.subheader("📄 Original Document")
        st.image(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            caption="Uploaded image",
            use_column_width=True
        )

        with st.spinner("Running extraction pipeline..."):
            results = pipeline.run(image)

        if show_steps:
            st.subheader("🔧 Preprocessing Steps")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(results["gray"], caption="Grayscale", use_column_width=True, clamp=True)
            with col2:
                st.image(results["thresh"], caption="Adaptive Thresholding", use_column_width=True, clamp=True)
            with col3:
                annotated_rgb = cv2.cvtColor(results["annotated"], cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, caption="Detected Regions", use_column_width=True)

        st.subheader("📊 Results")
        if results["digit_count"] == 0:
            st.warning("No digits detected. Try a clearer image.")
        else:
            st.metric("Regions detected", results["digit_count"])
            annotated_rgb = cv2.cvtColor(results["annotated"], cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, caption="Annotated output", use_column_width=True)

            if results["structured"]:
                df = pd.DataFrame([
                    {
                        "Number": r["number"],
                        "Digit count": len(r["digits"]),
                        "Avg confidence": f"{r['avg_confidence']:.1%}"
                    }
                    for r in results["structured"]
                ])
                st.dataframe(df, use_container_width=True)
                csv = df.to_csv(index=False)
                st.download_button(
                    "⬇️ Download CSV",
                    data=csv,
                    file_name="extracted_digits.csv",
                    mime="text/csv"
                )

# ════════════════════════════════════════════════════════════════════
# TAB 3 — Training results
# ════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🏋️ Model Training Results")
    st.markdown("""
    | Metric | Value |
    |--------|-------|
    | Dataset | MNIST (60,000 train / 10,000 test) |
    | Architecture | CNN with BatchNorm + Dropout |
    | Data Augmentation | Rotation, Zoom, Translation |
    | Epochs | 22 (early stopping) |
    | Test Accuracy | **99.22%** |
    | Test Loss | **0.0207** |
    """)

    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists("training_history.png"):
            st.image("training_history.png", caption="Training history")
    with col2:
        if os.path.exists("sample_predictions.png"):
            st.image("sample_predictions.png", caption="Sample predictions")

st.markdown("---")
st.markdown(
    "Built with TensorFlow & Streamlit · "
    "[GitHub](https://github.com/HatimOMp/mnist-digit-recognition)"
)