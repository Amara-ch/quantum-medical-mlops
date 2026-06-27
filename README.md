# quantum-medical-mlops
🧠⚛️ Hybrid Quantum-Classical Deep Learning for Medical Image Diagnosis — a production-grade MLOps pipeline with FastAPI, Docker, MLflow &amp; CI/CD.
# 🧠⚛️ NeuroScan — Hybrid Quantum-Classical Brain Tumor Detection

> A production-style MLOps pipeline that brings **Quantum Machine Learning** out of the notebook and into a deployed, working application — detecting brain tumors from MRI scans using a hybrid **ResNet18 CNN + 4-qubit Variational Quantum Circuit**.

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="PennyLane" src="https://img.shields.io/badge/PennyLane-Quantum-6133BD">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white">
  <img alt="MLflow" src="https://img.shields.io/badge/MLflow-0194E2?logo=mlflow&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
</p>

<p align="center">
  <a href="https://quantum-medical-mlops-apykp2ctcfwwyd867fqttu.streamlit.app"><b>🔗 Live Demo</b></a>
</p>

---

## 📌 Overview

Classical deep learning is powerful — but can a **quantum layer** integrate into a real, end-to-end machine learning system? **NeuroScan** answers that with a working, deployed application.

The model combines a classical convolutional backbone with a parameterised quantum circuit. The two are trained **jointly and end-to-end**, with gradients flowing seamlessly from PyTorch through PennyLane's quantum simulator.

| | |
|---|---|
| 🎯 **Task** | Binary classification — `tumor` vs `no_tumor` from brain MRI |
| 🧠 **Classical core** | ResNet18 (ImageNet-pretrained) feature extractor |
| ⚛️ **Quantum core** | 4-qubit Variational Quantum Circuit (angle encoding + entanglement) |
| 📊 **Accuracy** | **88%** on held-out validation data |
| ⚡ **Inference** | < 2 seconds on CPU |
| 🌐 **Deployment** | Live Streamlit web app |

---

## 🏗️ Architecture

```
   ┌─────────────┐    ┌──────────────────┐    ┌────────────────────┐    ┌──────────────┐
   │  Brain MRI  │ →  │   ResNet18 CNN   │ →  │  4-Qubit Quantum   │ →  │  Classifier  │ →  tumor / no_tumor
   │  224×224×3  │    │ (feature extract)│    │   Circuit (VQC)    │    │  Dense head  │
   └─────────────┘    └──────────────────┘    └────────────────────┘    └──────────────┘
                       512-d → 4 features      angle encode + entangle      Softmax
                                               + ⟨Z⟩ measurement
```

**The pipeline, step by step:**

1. **Preprocessing** — MRI resized to `224×224`, converted to RGB, normalised with ImageNet statistics.
2. **CNN feature extraction** — A frozen, pretrained **ResNet18** encodes spatial anatomy; its final layer is replaced to output **4 features** (one per qubit).
3. **Quantum encoding & processing** — Features are scaled to `[0, π]` via `tanh`, encoded into qubit rotations (`AngleEmbedding`), then processed by trainable **entangling layers** (`BasicEntanglerLayers`).
4. **Measurement** — The **Pauli-Z expectation value** `⟨Z⟩` is read from each of the 4 qubits, producing a 4-D real-valued vector.
5. **Classification** — A small dense head maps the quantum output to class logits; softmax yields calibrated probabilities.

> The entire network is **end-to-end differentiable** — PennyLane's `TorchLayer` bridges the quantum circuit into PyTorch's autograd, so classical and quantum parameters are optimised together with `Adam`.

---

## ⚛️ The Quantum Layer

The heart of the project is a compact 4-qubit circuit acting as a trainable neural network layer:

```python
@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS))        # encode classical features
    qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS)) # trainable entangling layers
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]  # measure each qubit
```

- **Device:** `default.qubit` simulator (runs locally, no quantum hardware required)
- **Qubits:** 4 | **Entangling layers:** 2 (trainable)
- **Gradients:** computed via PennyLane's parameter-shift rule

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| **Modelling** | PyTorch, TorchVision (ResNet18) |
| **Quantum** | PennyLane (`default.qubit`, `TorchLayer`) |
| **Experiment Tracking** | MLflow (params, metrics, artifacts, model versioning) |
| **Serving** | FastAPI (REST `/predict` endpoint) |
| **Web App** | Streamlit (interactive live demo) |
| **Data Versioning** | DVC |
| **Testing** | pytest |

---

## 📁 Project Structure

```
quantum-medical-mlops/
├── app.py                      # Streamlit web app (live demo front-end)
├── brain_tumor_model.pth       # Trained model weights
├── requirements.txt
├── src/
│   ├── data/
│   │   └── data_loader.py       # MRI loading, transforms, train/val split
│   ├── models/
│   │   ├── quantum_layer.py     # 4-qubit variational quantum circuit
│   │   └── hybrid_model.py      # ResNet18 + quantum layer + classifier
│   ├── training/
│   │   └── train.py             # Training loop with MLflow tracking
│   └── api/
│       └── main.py              # FastAPI model-serving endpoint
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone & install

```bash
git clone https://github.com/Amara-ch/quantum-medical-mlops.git
cd quantum-medical-mlops

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the web app locally

```bash
streamlit run app.py
```

Then open `http://localhost:8501`, upload a brain MRI, and get a prediction.

### 3. Serve the model via the API

```bash
uvicorn src.api.main:app --reload
```

Send an MRI to the `/predict` endpoint:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -F "file=@path/to/mri.jpg"
```

**Example response:**

```json
{
  "filename": "Y16.jpg",
  "prediction": "tumor",
  "confidence": 0.6513,
  "probabilities": { "no_tumor": 0.3487, "tumor": 0.6513 }
}
```

---

## 🧪 Training

The dataset is organised as `data/raw/brain_mri/{yes,no}/` — `ImageFolder` labels images automatically (`no → 0`, `yes → 1`).

```bash
python -m src.training.train
```

**Training details:**
- **Optimizer:** Adam (`lr = 0.001`)
- **Loss:** Cross-entropy with **class weighting** to handle imbalance (155 tumor vs 98 healthy)
- **Reproducibility:** fixed-seed train/val split
- **Tracking:** every run logs params, per-epoch metrics, and the best model to **MLflow**

Inspect experiments:

```bash
mlflow ui
```

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Validation accuracy | **88%** |
| Inference time (CPU) | < 2 s |
| Output | Class label + confidence + per-class probabilities |

> Class weighting was key — without it, the model could exploit the imbalance and over-predict the majority class. Weighting forced it to genuinely learn discriminative features.

---

## 🗺️ Roadmap

- [ ] Containerise with Docker for reproducible deployment
- [ ] GitHub Actions CI/CD (lint, test, deploy)
- [ ] Expand to multi-class tumor typing (glioma / meningioma / pituitary)
- [ ] Add Grad-CAM explainability overlays
- [ ] Benchmark the quantum layer against a purely classical baseline

---

## ⚠️ Disclaimer

NeuroScan is a **research and educational project**. It is **not** a certified medical device and must **not** be used for clinical diagnosis. All medical decisions should be made by qualified healthcare professionals.

---

## 📄 License

Released under the **MIT License** — see [LICENSE](LICENSE).

---

<p align="center">
  Built with PyTorch · PennyLane · FastAPI · MLflow · Streamlit<br>
  <i>Exploring the intersection of quantum computing, deep learning, and ML systems engineering.</i>
</p>
