# 🧠 Brain Tumor Detection AI - Web Application

[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-98.96%25-success)](.)

> **AI-powered brain tumor detection and classification system with interactive web interface**

---

## 🎯 What This Does

Upload a brain MRI scan and instantly get:
- ✅ **Tumor Detection**: Is there a tumor present?
- 🔍 **Classification**: If yes, what type? (Glioma, Meningioma, Pituitary)
- 📊 **Confidence Score**: How certain is the AI?
- 💡 **Medical Info**: Description, severity, and recommendations

**4 Categories:**
- 🔴 **Glioma Tumor** - Originates from glial cells
- 🟠 **Meningioma Tumor** - Forms on brain/spinal cord membranes
- 🟣 **Pituitary Tumor** - Develops in pituitary gland
- 🟢 **No Tumor** - Healthy brain scan

---

## 🚀 Quick Start - Use the Web App!

### 1️⃣ Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2️⃣ Start the Application
```powershell
# Run locally (default)
streamlit run app.py

# Run as a network-accessible server (bind to 0.0.0.0)
streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
```

### 3️⃣ Open in Browser
The app will automatically open at: **http://localhost:8501**

### 4️⃣ Upload & Predict
1. Click "Browse files" button
2. Select a brain MRI image (PNG, JPG, or NPY format)
3. View instant results with confidence scores!

**Try it with test images from:** `data/processed/Testing/` folder

---

## 📊 Performance Metrics

**Trained on 7,023 brain MRI scans with GPU acceleration**

### Overall Performance
| Metric | Score |
|--------|-------|
| **Test Accuracy** | **98.96%** 🎯 |
| Training Accuracy | 99.86% |
| Validation Accuracy | 99.24% |
| Training Time | ~42 minutes (GPU) |
| GPU Speedup | 6-8x faster than CPU |

### Per-Class Performance
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Glioma | 99.58% | 97.94% | 98.76% |
| Meningioma | 97.60% | 98.79% | 98.19% |
| No Tumor | 99.67% | 99.67% | 99.67% |
| Pituitary | 98.87% | 99.24% | 99.05% |

**Only 11 misclassifications out of 1,054 test images!** ✨

---

## 🏗️ Technical Architecture

### Framework & Hardware
- **Framework**: PyTorch 2.7.1 with CUDA 11.8
- **GPU**: NVIDIA GeForce RTX 2050 (4GB VRAM)
- **Model**: ResNet50 with Transfer Learning
- **Interface**: Streamlit Web Application

### Model Details
- **Base Model**: ResNet50 (pre-trained on ImageNet)
- **Training Strategy**: Two-stage training
  - Stage 1: 20 epochs with frozen base layers
  - Stage 2: 30 epochs with fine-tuning
- **Optimizer**: Adam (lr=0.001 → 0.0001)
- **Batch Size**: 24 (optimized for 4GB GPU)
- **Image Size**: 224×224 pixels

### Data Split
- **Training**: 4,915 images (70%)
- **Validation**: 1,054 images (15%)
- **Testing**: 1,054 images (15%)

---

## 📁 Project Structure

```
mri/
├── app.py                          # 🌐 Streamlit Web Application
├── train_pytorch.py                # 🏋️ Model training script
├── evaluate_pytorch.py             # 📊 Model evaluation script
├── visualize_training.py           # 📈 Training visualization
├── predict.py                      # 🔮 Single image inference
├── requirements.txt                # 📦 Python dependencies
│
├── data/
│   ├── raw/                        # Original dataset
│   ├── processed/                  # Preprocessed images
│   │   ├── glioma/                # Glioma tumor images
│   │   ├── meningioma/            # Meningioma tumor images
│   │   ├── notumor/               # Healthy brain images
│   │   ├── pituitary/             # Pituitary tumor images
│   │   ├── Testing/               # Test set
│   │   └── Training/              # Training set
│   └── splits/
│       └── split_info.json        # Dataset split information
│
├── src/                            # 🔧 Source code modules
│   ├── data_preprocessing.py      # Data preprocessing
│   ├── data_augmentation.py       # Data augmentation
│   ├── dataset_split.py           # Dataset splitting
│   ├── data_pipeline.py           # Complete data pipeline
│   ├── data_loaders.py            # PyTorch DataLoaders
│   ├── model_architecture.py      # ResNet50 model
│   ├── model_trainer.py           # Training orchestration
│   ├── training_callbacks.py      # Training callbacks
│   ├── evaluation_metrics.py      # Evaluation metrics
│   ├── visualization.py           # Visualization tools
│   └── gradcam.py                 # Grad-CAM visualization
│
├── models/                         # 💾 Trained models
│   └── final_model_20251106_142153.pth  # Best model (98.96% accuracy)
│
├── logs/                           # 📝 Training logs
│   └── training_history_20251106_142153.csv
│
├── results/                        # 📊 Evaluation results
│   ├── evaluation_summary.txt     # Detailed metrics
│   ├── test_results.csv           # Per-image predictions
│   └── training_history.png       # Loss/accuracy plots
│
└── docs/                           # 📚 Documentation
    ├── README.md                   # Documentation hub
    ├── INSTALLATION.md             # Installation guide
    ├── GPU_SETUP.md                # GPU setup guide
    ├── PHASE1_COMPLETE.md          # Phase 1 details
    ├── PHASE2_COMPLETE.md          # Phase 2 details
    ├── PHASE3_COMPLETE.md          # Phase 3 details
    └── PROJECT_STATUS.md           # Project status
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- NVIDIA GPU with CUDA support (optional but recommended)
- Windows 10/11 (tested) or Linux

### Step 1: Clone or Download Project
```powershell
cd c:\Users\YourName\Desktop
git clone <your-repo-url> mri
cd mri
```

### Step 2: Create Virtual Environment (Recommended)
```powershell
# Recommended (creates `.venv` folder)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Or use `venv` if you prefer
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
```powershell
pip install -r requirements.txt
```

**Key Dependencies:**
- `torch==2.7.1` (with CUDA 11.8)
- `torchvision==0.20.1`
- `streamlit` (for web interface)
- `matplotlib`, `seaborn` (for visualization)
- `scikit-learn` (for metrics)
- `pillow` (for image handling)

### Step 4: Verify GPU (Optional)
```powershell
# Quick inline GPU check (no extra script required)
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

You should see:
```
✓ CUDA is available
✓ GPU: NVIDIA GeForce RTX 2050
```

Repository cleanup:
- Removed legacy/temporary files from the repository: `file.txt`, `streamlit.err`, `streamlit.log`, `UPDATES.txt`, `test_pytorch_gpu.py`. If you need any removed file, restore it with `git checkout -- <filename>`.

---

## 📖 Usage Guide

### 🌐 Web Application (Recommended)

**Start the web app:**
```powershell
streamlit run app.py
```

**Features:**
- Drag-and-drop image upload
- Instant predictions with confidence scores
- Color-coded results (green=no tumor, red=tumor detected)
- Detailed tumor information and recommendations
- Beautiful, user-friendly interface

### 🏋️ Training a New Model

**Quick training:**
```powershell
python train_pytorch.py
```

**Custom training:**
```powershell
python train_pytorch.py --epochs 50 --batch-size 24 --learning-rate 0.001
```

### 📊 Evaluate Model

**Evaluate on test set:**
```powershell
python evaluate_pytorch.py
```

**Outputs:**
- `results/evaluation_summary.txt` - Overall metrics
- `results/test_results.csv` - Per-image predictions
- Confusion matrix and per-class metrics

### 📈 Visualize Training

**Plot training history:**
```powershell
python visualize_training.py
```

**Creates:**
- `results/training_history.png` - Loss and accuracy curves

### 🔮 Single Image Prediction

**Predict on one image:**
```powershell
python predict.py --image path/to/image.jpg
```

**Or use specific model:**
```powershell
python predict.py --image path/to/image.jpg --model models/final_model.pth
```

---

## 🔬 Development Phases

### ✅ Phase 1: Data Preparation (Complete)
- Data preprocessing and normalization
- Advanced augmentation techniques
- Stratified train/val/test split (70/15/15)
- Complete data pipeline

### ✅ Phase 2: Model Training (Complete)
- ResNet50 with transfer learning
- Two-stage training strategy
- GPU acceleration with PyTorch + CUDA
- Learning rate scheduling and early stopping
- Best model checkpointing

### ✅ Phase 3: Model Evaluation (Complete)
- Comprehensive evaluation metrics
- Confusion matrix analysis
- Per-class performance metrics
- Training visualization
- Grad-CAM implementation

### ✅ Phase 4: Web Deployment (Complete)
- Interactive Streamlit web application
- Real-time predictions
- User-friendly interface
- Comprehensive result display
- Medical recommendations

---

## 🎓 Model Performance Analysis

### Training Progression
- **Epochs 1-20**: Frozen base, training only classifier head
  - Validation accuracy reached ~97%
- **Epochs 21-50**: Fine-tuning last layers
  - Validation accuracy improved to 99.24%

### Best Model
- **Saved at**: Epoch 42
- **Validation Loss**: 0.0450
- **Validation Accuracy**: 99.24%
- **Test Accuracy**: 98.96%

### Confusion Matrix Highlights
- **Glioma**: 238/243 correct (97.94% recall)
- **Meningioma**: 244/247 correct (98.79% recall)
- **No Tumor**: 299/300 correct (99.67% recall)
- **Pituitary**: 262/264 correct (99.24% recall)

Most misclassifications are between similar tumor types, which is medically understandable.

---

## 💡 Key Features

### Web Application
- ✨ Beautiful, intuitive interface
- 📤 Drag-and-drop image upload
- ⚡ Real-time GPU-accelerated predictions
- 🎨 Color-coded results
- 📊 Confidence scores for all classes
- 💬 Medical descriptions and recommendations
- ⚠️ Appropriate disclaimers

### Model Capabilities
- 🎯 98.96% test accuracy
- ⚡ Fast inference (<1 second per image)
- 🔄 Handles various MRI formats (PNG, JPG, NPY)
- 🧠 Transfer learning from ImageNet
- 🎮 GPU acceleration for speed

### Code Quality
- 📝 Well-documented code
- 🧪 Modular architecture
- 🔧 Easy to extend and customize
- 📊 Comprehensive logging
- ✅ Tested on Windows + GPU

---

## ⚠️ Important Disclaimer

**This tool is for educational and research purposes only.**

- ❌ NOT a replacement for professional medical diagnosis
- ❌ NOT approved for clinical use
- ✅ For learning and demonstration purposes
- ✅ Always consult qualified healthcare professionals

The model predictions should be verified by trained radiologists and medical professionals.

---

## 🔧 Troubleshooting

### GPU Not Detected
```powershell
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Streamlit Not Starting
```powershell
# Reinstall Streamlit
pip install --upgrade streamlit

# Run with verbose output
streamlit run app.py --logger.level=debug
```

### Model File Not Found
Ensure `models/final_model_20251106_142153.pth` exists. If not, train a new model:
```powershell
python train_pytorch.py
```

### Memory Issues
If you get CUDA out of memory errors, reduce batch size:
```powershell
python train_pytorch.py --batch-size 16
```

---

## 📚 Documentation

**Comprehensive guides available in [`docs/`](docs/) folder:**

- **[Quick Start Guide](docs/QUICK_START.md)** - ⚡ Fast reference for launching the app
- **[Installation Guide](docs/INSTALLATION.md)** - Detailed setup instructions
- **[GPU Setup Guide](docs/GPU_SETUP.md)** - Configure GPU acceleration
- **[Phase 1 Complete](docs/PHASE1_COMPLETE.md)** - Data preparation details
- **[Phase 2 Complete](docs/PHASE2_COMPLETE.md)** - Training methodology
- **[Phase 3 Complete](docs/PHASE3_COMPLETE.md)** - Evaluation process
- **[Project Status](docs/PROJECT_STATUS.md)** - Current development status

---

## 🤝 Contributing

This is an educational project. Feel free to:
- Report bugs or issues
- Suggest improvements
- Add new features
- Improve documentation

---

## 📄 License

This project is for educational purposes. Please ensure you have appropriate rights to use the dataset.

---

## 🙏 Acknowledgments

- **Dataset**: Brain Tumor MRI Dataset
- **Framework**: PyTorch and Streamlit teams
- **Model**: ResNet50 architecture (He et al., 2015)
- **GPU**: NVIDIA CUDA toolkit

---

## 📞 Support

For questions or issues:
1. Check the [docs/](docs/) folder for detailed guides
2. Review this README thoroughly
3. Check existing issues/discussions

---

**Built with ❤️ using PyTorch, Streamlit, and GPU acceleration**

**Last Updated**: November 7, 2025
**Version**: 1.0.0 - Phase 4 Complete
