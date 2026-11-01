# 🧠 NeuroDoctor AI: AI-Powered Brain Tumor Detection & XAI Visualizer

[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://share.streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)

**NeuroDoctor AI** is an end-to-end medical deep learning web application that classifies brain MRI scans into four categorical diagnostic classes: **Glioma, Meningioma, No Tumor, and Pituitary**. It also generates Explainable AI (XAI) visualizations using **Grad-CAM (Gradient-weighted Class Activation Mapping)** to highlight the image regions that influence the model's predictions.

The application combines deep learning, image preprocessing, computer vision, and explainable AI to provide an interactive brain tumor classification and visualization experience.

---

## 📌 Features

- **Fine-Tuned DenseNet121 Architecture:** Employs feature-rich dense convolutional connectivity for brain tumor classification across MRI images.
- **Advanced Preprocessing Pipeline:** Integrates single-channel CLAHE (Contrast Limited Adaptive Histogram Equalization) contrast enhancement, spatial resizing (224 × 224), and dynamic RGB normalization.
- **Explainable AI (Grad-CAM) Localization:** Computes backpropagated feature gradients relative to the final convolutional layer to visualize model decision-making through heatmap overlays.
- **Clinical Web Interface:** Built using Streamlit, featuring an intuitive dark-themed UI, progress-bar probability distributions, and instant MRI image analysis.
- **Prediction Confidence Scores:** Displays probability distributions for the four tumor categories to help users understand the model's predictions.
- **Git LFS Integration:** Model weights (`best_densenet121_mri.h5`) are version-controlled using Git Large File Storage for deployment and model management.

---

## 🛠️ Project Architecture & Pipeline

The application follows an image preprocessing, deep learning classification, and explainability pipeline.

```text
📁 Input MRI Scan
       │
       ▼
⚙️ Image Preprocessing Pipeline
  ├── Grayscale Conversion
  ├── CLAHE Contrast Enhancement
  ├── Resize (224 × 224)
  └── RGB Normalization [0.0, 1.0]
       │
       ▼
🤖 DenseNet121 Deep Learning Backbone
       │
  ┌────┴──────────────────────────────┐
  ▼                                   ▼
📊 Diagnostic Prediction           🔥 Grad-CAM XAI Map Generation
  ├── Class Determination             ├── Feature Map Gradient Extraction
  └── Confidence Score Breakdown      └── Jet Color Map Overlay
```

---

## 📂 Repository Structure

```text
NeuroDoctor-AI-App/
│
├── .gitattributes           # Git LFS configuration for tracking large .h5 files
├── app.py                   # Main Streamlit web application and Grad-CAM pipeline
├── best_densenet121_mri.h5  # Fine-tuned DenseNet121 model weights (Git LFS)
├── requirements.txt         # Python package dependencies
└── README.md                # Project documentation
```

---

## 🔬 Explainable AI (XAI) & Grad-CAM

Deep neural networks are often treated as black-box models because their internal decision-making processes are difficult to interpret.

NeuroDoctor AI utilizes **Gradient-weighted Class Activation Mapping (Grad-CAM)** to generate visual explanations of the model's predictions.

Grad-CAM extracts activation maps from the final convolutional layer and uses gradients to identify the image regions that contribute most to a particular class prediction.

1. **Gradient Computation:** Calculates the gradient of the winning class score with respect to feature activation maps of the final convolutional layer.
2. **Global Average Pooling:** Computes neuron importance weights by averaging the gradients across all spatial dimensions (width and height) of the feature maps.
3. **Heatmap Generation:** Computes a weighted combination of forward activation maps followed by a ReLU operation to eliminate negative influences and retain only positive contributing features.
4. **Visual Overlay:** Blends the jet-colored heatmap onto the preprocessed grayscale MRI scan with an adjustable opacity factor ($\alpha = 0.45$).

## 🚀 Local Installation & Setup

Follow these steps to run NeuroDoctor on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/Kavana-shree-v/NeuroDoctor-AI-App.git

cd NeuroDoctor-AI-App
```

### 2. Set Up a Virtual Environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download / Place Model Weights

Ensure that the following model file is present in the project root directory:

```text
best_densenet121_mri.h5
```

If Git LFS is used to track the model weights, retrieve the actual model file using:

```bash
git lfs pull
```

### 5. Launch the Streamlit Application

```bash
streamlit run app.py
```

Open your browser and navigate to:

```text
http://localhost:8501
```

The NeuroDoctor application will be available locally.

---

## 💻 Tech Stack & Dependencies

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning Framework | TensorFlow / Keras |
| Deep Learning Architecture | DenseNet121 |
| Computer Vision | OpenCV |
| Image Processing | Pillow (PIL) |
| Web Application Framework | Streamlit |
| Data Operations | NumPy |
| Explainable AI | Grad-CAM |
| Visualization | Matplotlib / OpenCV |
| Version Control | Git & GitHub |
| Large File Management | Git LFS |

---

## 🎯 Project Objectives

- Develop an AI-powered brain tumor classification application.
- Classify MRI images into four categories: Glioma, Meningioma, No Tumor, and Pituitary.
- Apply image preprocessing techniques to improve image quality and consistency.
- Utilize transfer learning with the DenseNet121 architecture.
- Implement Grad-CAM to provide visual explanations of model predictions.
- Develop an interactive web application using Streamlit.

---

## ⚠️ Disclaimer

NeuroDoctor AI is an academic and research-oriented AI project developed for educational and experimental purposes. Its predictions and Grad-CAM visualizations are not a substitute for professional medical diagnosis, radiological interpretation, or clinical evaluation.

Always consult qualified healthcare professionals for medical decisions.

---

## 📜 License

This project is open-source under the **MIT License**.

---

## 👤 Authors

**Charvi S S** and **Kavana Shree V**

- GitHub: [@Kavana-shree-v](https://github.com/Kavana-shree-v)
- Project Repository: [NeuroDoctor-AI-App](https://github.com/Kavana-shree-v/NeuroDoctor-AI-App)

---

