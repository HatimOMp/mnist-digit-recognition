# 🔢 Document Digit Extractor

An end-to-end pipeline that combines **classical computer vision** (OpenCV) with a **CNN trained from scratch** to detect, extract and classify digits from document images (receipts, invoices, forms).

![Sample Predictions](sample_predictions.png)

## 🎯 Project Overview

This project was built to explore the full lifecycle of a computer vision pipeline — from raw image preprocessing to structured data output. It highlights a real-world challenge in ML: the gap between laboratory accuracy and real-world performance, known as **domain shift**.

The CNN achieves **99.22% accuracy** on the MNIST test set. When applied to real document photos, performance degrades due to differences in font, lighting, perspective and background — a finding that drives the architecture decisions documented below.

## 🏗️ Pipeline Architecture
📄 Input: Document photo (receipt, invoice, form)
↓
🔧 Preprocessing (OpenCV)
├── Grayscale conversion
├── Gaussian blur (denoising)
├── Adaptive thresholding (handles uneven lighting)
└── Morphological operations (noise removal)
↓
🔍 Digit Detection (OpenCV)
├── Contour detection
├── Bounding box extraction
├── Multi-criteria filtering (size, aspect ratio, solidity)
└── Spatial sorting (left-to-right, top-to-bottom)
↓
🧠 CNN Classification (TensorFlow/Keras)
├── Per-region 28×28 resizing
├── Normalization
└── Softmax prediction + confidence score
↓
📊 Structured Output
├── Annotated image with bounding boxes
├── Digits grouped by line
├── Confidence scores per digit
└── CSV export

## 🧠 Model Architecture
Input (28×28×1)
↓
Data Augmentation (RandomRotation, RandomZoom, RandomTranslation)
↓
Conv2D(32) → BatchNorm → Conv2D(32) → MaxPool → Dropout(0.25)
↓
Conv2D(64) → BatchNorm → Conv2D(64) → MaxPool → Dropout(0.25)
↓
Flatten → Dense(256) → BatchNorm → Dropout(0.5)
↓
Dense(10, softmax)

## 📊 Training Results

| Metric | Value |
|--------|-------|
| Dataset | MNIST (60,000 train / 10,000 test) |
| Optimizer | Adam (lr=1e-3) |
| Data Augmentation | Rotation, Zoom, Translation |
| Epochs | 22 (early stopping) |
| Test Accuracy | **99.22%** |
| Test Loss | **0.0207** |

## ⚠️ Domain Shift — An Honest Discussion

A key finding of this project is the performance gap between MNIST and real documents.

**Why it happens:**
- MNIST contains isolated, centered, handwritten digits on clean backgrounds
- Real receipts contain printed fonts, mixed alphanumeric text, perspective distortion, shadows and background noise
- The model was never exposed to these conditions during training

**What this means in practice:**
- On clean MNIST samples: **99.22% accuracy**, 100% confidence
- On real document photos: significant false positives from letter detection and background noise

**What a production system would need:**
- A dedicated text detection model (e.g. CRAFT, EAST) to isolate digit regions before classification
- Fine-tuning on printed fonts and real document images
- OCR-specific preprocessing (deskewing, perspective correction)

This gap between lab metrics and real-world performance is a fundamental challenge in applied ML and motivated further research into purpose-built OCR pipelines.

## 🚀 Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/HatimOMp/mnist-digit-recognition
cd mnist-digit-recognition
```

**2. Create a virtual environment**
```bash
conda create -n mnist-env python=3.10
conda activate mnist-env
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Train the model**
```bash
python train.py
```

**5. Launch the app**
```bash
streamlit run app.py
```

## 🗂️ Project Structure
mnist-digit-recognition/
│
├── model.py                  # CNN architecture
├── train.py                  # Training script with data augmentation
├── pipeline.py               # Full extraction pipeline (OpenCV + CNN)
├── app.py                    # Streamlit interface (3 tabs)
├── requirements.txt          # Dependencies
├── training_history.png      # Accuracy and loss curves
└── sample_predictions.png    # Sample test set predictions

## 🛠️ Tech Stack

- **TensorFlow / Keras** — CNN architecture and training
- **OpenCV** — image preprocessing and contour detection
- **Streamlit** — interactive web interface
- **NumPy / Pandas** — data processing and CSV export
- **Matplotlib** — training visualization

## 💡 Key Learnings

1. **Data augmentation matters** — adding rotation, zoom and translation improved generalization significantly
2. **BatchNormalization + Dropout** together prevent overfitting effectively on image data
3. **Domain shift is real** — 99% accuracy on a benchmark does not guarantee real-world performance
4. **Solidity filtering** (contour area / bounding box area) is more robust than size filtering alone for separating digits from noise

## 👤 Author

**Hatim Omari** — [LinkedIn](https://www.linkedin.com/in/hatim-omari/) · [GitHub](https://github.com/HatimOMp)