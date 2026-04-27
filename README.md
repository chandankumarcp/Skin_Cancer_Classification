# 🔬 Skin Cancer Classification using ResNet-50

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![ResNet-50](https://img.shields.io/badge/Model-ResNet--50-green?style=for-the-badge)
![Accuracy](https://img.shields.io/badge/Accuracy-92.4%25-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

<br/>

> **An AI-powered Skin Cancer Detection system that assists dermatologists in early and accurate identification of malignant skin lesions with 92.4% accuracy.**

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Model Architecture](#-model-architecture)
- [Dataset](#-dataset)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [How It Works](#-how-it-works)
- [Technologies Used](#-technologies-used)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🧠 Overview

Skin cancer is one of the most prevalent forms of cancer worldwide, and **early detection is the single most important factor in improving patient survival rates**. This project implements a deep learning–based skin cancer detection system capable of classifying skin lesion images as **malignant** or **benign** with high precision.

By leveraging the power of **transfer learning** with the pre-trained **ResNet-50** architecture (trained on ImageNet), the model achieves a classification accuracy of **92.4%** — making it a reliable tool to support dermatologists in clinical decision-making.

---

## ✨ Key Features

- 🎯 **High Accuracy** — 92.4% classification accuracy on skin lesion images
- 🏗️ **Transfer Learning** — Built on ResNet-50, pre-trained on ImageNet for robust feature extraction
- ⚡ **Fast Inference** — Rapid predictions suitable for real-time clinical assistance
- 🧪 **Malignancy Detection** — Accurately distinguishes malignant cancer cells from benign lesions
- 📊 **Comprehensive Evaluation** — Detailed metrics including precision, recall, F1-score, and confusion matrix
- 🔬 **Clinical Utility** — Designed to assist dermatologists in early diagnosis and improve patient outcomes

---

## 🏗️ Model Architecture

The system uses **ResNet-50** (Residual Network with 50 layers) as the backbone, with custom classification layers on top:

```
Input Image (224×224×3)
        ↓
  ResNet-50 Backbone
  (Pre-trained on ImageNet)
        ↓
  Global Average Pooling
        ↓
  Dense Layer (256 units, ReLU)
        ↓
  Dropout (0.5)
        ↓
  Dense Output Layer (Softmax)
        ↓
  Classification: Malignant / Benign
```

### Why ResNet-50?
- **Residual connections** combat the vanishing gradient problem, enabling training of very deep networks
- **Pre-trained weights** from ImageNet provide rich, generalized feature representations
- Proven performance on medical image classification tasks
- Excellent balance between model depth and computational efficiency

---

## 📁 Dataset

The model is trained on a labeled dataset of dermoscopic skin lesion images, consisting of:

| Class | Description |
|-------|-------------|
| **Malignant** | Cancerous skin lesions requiring medical attention |
| **Benign** | Non-cancerous skin conditions |

> **Dataset preprocessing steps:**
> - Image resizing to **224×224** pixels (ResNet-50 input requirement)
> - Pixel normalization (scaling to `[0, 1]`)
> - Data augmentation: horizontal/vertical flips, rotation, zoom, brightness adjustment
> - Train/Validation/Test split for robust evaluation

---

## 📊 Results

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **92.4%** |
| Precision | ~92% |
| Recall | ~92% |
| F1-Score | ~92% |

The model demonstrates strong generalization across unseen skin lesion images, achieving consistent performance across both malignant and benign classes.

---

## 📂 Project Structure

```
Skin_Cancer_Classification/
│
├── Skin-Cancer-detection-using-ResNet-50/   # Core detection module
│   ├── dataset/                             # Dataset directory
│   │   ├── train/                           # Training images
│   │   ├── validation/                      # Validation images
│   │   └── test/                            # Test images
│   ├── model/                               # Saved model weights
│   ├── notebooks/                           # Jupyter notebooks for EDA & training
│   ├── src/                                 # Source code
│   │   ├── data_preprocessing.py            # Data loading and augmentation
│   │   ├── model.py                         # ResNet-50 model definition
│   │   ├── train.py                         # Training script
│   │   └── predict.py                       # Inference script
│   └── requirements.txt                     # Python dependencies
│
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager
- GPU recommended (NVIDIA CUDA-compatible) for faster training

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/chandankumarcp/Skin_Cancer_Classification.git
   cd Skin_Cancer_Classification
   ```

2. **Create and activate a virtual environment**
   ```bash
   # Using venv
   python -m venv venv
   source venv/bin/activate        # Linux/macOS
   venv\Scripts\activate           # Windows

   # OR using conda
   conda create -n skin-cancer python=3.8
   conda activate skin-cancer
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify GPU setup (optional but recommended)**
   ```python
   import tensorflow as tf
   print(tf.config.list_physical_devices('GPU'))
   ```

---

## 🚀 Usage

### Training the Model

```bash
python src/train.py --epochs 50 --batch_size 32 --learning_rate 0.0001
```

### Running Inference on a Single Image

```python
from src.predict import predict_skin_lesion

result = predict_skin_lesion("path/to/skin_lesion.jpg")
print(f"Prediction: {result['label']}")
print(f"Confidence: {result['confidence']:.2f}%")
```

### Running via Jupyter Notebook

```bash
jupyter notebook notebooks/
```
Open the training or inference notebook and follow the step-by-step instructions.

---

## 🔬 How It Works

```
1. INPUT           →  Skin lesion image provided by dermatologist or patient
2. PREPROCESSING   →  Resize, normalize, and augment the image
3. FEATURE EXTRACT →  ResNet-50 backbone extracts hierarchical visual features
4. CLASSIFICATION  →  Custom dense layers output malignant/benign probability
5. OUTPUT          →  Prediction label with confidence score
```

**Transfer Learning Strategy:**
- ResNet-50 base layers are initially **frozen** to preserve ImageNet features
- Only the custom classification head is trained in the first phase
- In the second phase, the top layers of ResNet-50 are **fine-tuned** for domain-specific features

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python 3.8+** | Core programming language |
| **TensorFlow 2.x** | Deep learning framework |
| **Keras** | High-level neural network API |
| **ResNet-50** | Pre-trained CNN backbone |
| **NumPy** | Numerical computations |
| **Pandas** | Data manipulation |
| **Matplotlib / Seaborn** | Visualization & plotting |
| **scikit-learn** | Model evaluation metrics |
| **OpenCV / PIL** | Image processing |
| **Jupyter Notebook** | Experimentation & prototyping |

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help improve this project:

1. **Fork** the repository
2. **Create** a new branch: `git checkout -b feature/your-feature-name`
3. **Commit** your changes: `git commit -m 'Add some feature'`
4. **Push** to the branch: `git push origin feature/your-feature-name`
5. **Open a Pull Request**

Please make sure your contributions align with the project's goals and follow clean coding practices.

---

## ⚠️ Disclaimer

> This tool is intended to **assist** medical professionals and is **not a substitute** for professional medical diagnosis. All predictions should be reviewed and validated by a qualified dermatologist or healthcare provider.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

**Chandan Kumar**

[![GitHub](https://img.shields.io/badge/GitHub-chandankumarcp-181717?style=flat-square&logo=github)](https://github.com/chandankumarcp)
[![Email](https://img.shields.io/badge/Email-chandanbecp%40gmail.com-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:chandanbecp@gmail.com)

---

<div align="center">

**⭐ If this project helped you or inspired your work, please consider giving it a star!**

*Made with ❤️ to advance early skin cancer detection and save lives.*

</div>
