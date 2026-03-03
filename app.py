"""
Brain Tumor Detection Web Application
======================================
Upload an MRI brain scan and get instant tumor detection results!
"""

import os
import sys

# CRITICAL: Patch pathlib before any imports that might use it
import sys
import pathlib

# Create a shim for pathlib._local
if not hasattr(pathlib, '_local'):
    pathlib._local = pathlib.Path

# Also add to sys.modules to intercept any dynamic imports
sys.modules['pathlib._local'] = pathlib

# Import PyTorch first before TensorFlow to avoid pathlib conflicts
import torch
import torch.nn.functional as F
from torchvision import transforms

import streamlit as st
from PIL import Image
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model_architecture import ResNet50Classifier

# Page configuration
st.set_page_config(
    page_title="Brain Tumor Detection AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .tumor-detected {
        background-color: #FFEBEE;
        border-left: 5px solid #E53935;
    }
    .no-tumor {
        background-color: #E8F5E9;
        border-left: 5px solid #43A047;
    }
    .confidence-high {
        color: #2E7D32;
        font-weight: bold;
    }
    .confidence-medium {
        color: #F57C00;
        font-weight: bold;
    }
    .confidence-low {
        color: #C62828;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Class names
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Tumor information
TUMOR_INFO = {
    'Glioma': {
        'description': 'Glioma is a tumor that originates from glial cells in the brain or spine.',
        'severity': 'High',
        'color': '#E53935',
        'recommendation': '⚠️ Immediate medical consultation required. This is a serious condition that needs expert evaluation.'
    },
    'Meningioma': {
        'description': 'Meningioma is a tumor that forms on membranes covering the brain and spinal cord.',
        'severity': 'Moderate to High',
        'color': '#FB8C00',
        'recommendation': '⚠️ Medical consultation recommended. While often benign, professional evaluation is essential.'
    },
    'No Tumor': {
        'description': 'No tumor detected in the brain scan.',
        'severity': 'None',
        'color': '#43A047',
        'recommendation': '✅ No tumor detected. However, if you have symptoms, please consult a healthcare professional.'
    },
    'Pituitary': {
        'description': 'Pituitary tumor develops in the pituitary gland at the base of the brain.',
        'severity': 'Moderate',
        'color': '#8E24AA',
        'recommendation': '⚠️ Medical consultation recommended. Pituitary tumors can affect hormone levels and require monitoring.'
    }
}

@st.cache_resource
def load_model():
    """Load the trained PyTorch model"""
    model_path = 'models/final_model_20251106_142153.pth'
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        st.info("Please ensure the trained model is in the 'models' folder.")
        return None
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ResNet50Classifier(num_classes=4)
        
        # Load the model checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Extract model state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Load weights into model
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        return model, device
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

def preprocess_image(image):
    """Preprocess uploaded image for model inference"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

def predict(model, device, image_tensor):
    """Make prediction on the image"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        
    return predicted_idx.item(), confidence.item(), probabilities[0].cpu().numpy()

def display_results(predicted_class, confidence, all_probabilities):
    """Display prediction results in a user-friendly format"""
    tumor_type = CLASS_NAMES[predicted_class]
    info = TUMOR_INFO[tumor_type]
    
    # Main result box
    if tumor_type == 'No Tumor':
        st.markdown(f'<div class="result-box no-tumor">', unsafe_allow_html=True)
        st.markdown(f"## ✅ Result: {tumor_type}")
    else:
        st.markdown(f'<div class="result-box tumor-detected">', unsafe_allow_html=True)
        st.markdown(f"## ⚠️ Result: {tumor_type} Detected")
    
    # Confidence level
    confidence_pct = confidence * 100
    if confidence_pct >= 90:
        conf_class = "confidence-high"
    elif confidence_pct >= 70:
        conf_class = "confidence-medium"
    else:
        conf_class = "confidence-low"
    
    st.markdown(f'<p class="{conf_class}">Confidence: {confidence_pct:.2f}%</p>', unsafe_allow_html=True)
    
    # Description
    st.markdown(f"**Description:** {info['description']}")
    st.markdown(f"**Severity:** {info['severity']}")
    
    # Recommendation
    st.markdown("---")
    st.markdown(f"{info['recommendation']}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Show all probabilities
    st.markdown("### 📊 Detailed Prediction Probabilities")
    
    cols = st.columns(4)
    for idx, (class_name, prob) in enumerate(zip(CLASS_NAMES, all_probabilities)):
        with cols[idx]:
            st.metric(
                label=class_name,
                value=f"{prob * 100:.2f}%",
                delta="Predicted" if idx == predicted_class else None
            )

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 Brain Tumor Detection AI</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📋 About")
        st.info("""
        This AI model analyzes brain MRI scans and detects:
        - **Glioma** 🔴
        - **Meningioma** 🟠
        - **Pituitary Tumor** 🟣
        - **No Tumor** 🟢
        
        **Accuracy:** 98.96% on test data
        """)
        
        st.markdown("## ⚙️ Model Info")
        model_loaded = load_model()
        if model_loaded:
            model, device = model_loaded
            st.success(f"✅ Model loaded successfully!")
            st.info(f"🖥️ Device: {device.type.upper()}")
            if device.type == 'cuda':
                st.info(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        
        st.markdown("## ⚠️ Disclaimer")
        st.warning("""
        This tool is for **educational purposes only** and should NOT replace professional medical diagnosis.
        
        Always consult qualified healthcare professionals for medical advice.
        """)
    
    # Main content
    st.markdown("### 📤 Upload Brain MRI Scan")
    st.markdown("Upload a brain MRI image to detect if a tumor is present and identify its type.")
    
    uploaded_file = st.file_uploader(
        "Choose an MRI image...",
        type=['png', 'jpg', 'jpeg', 'npy'],
        help="Upload a brain MRI scan in PNG, JPG, JPEG, or NPY format"
    )
    
    if uploaded_file is not None:
        try:
            # Load image
            if uploaded_file.name.endswith('.npy'):
                # Handle numpy files
                image_array = np.load(uploaded_file)
                
                # Normalize to 0-255 range if needed
                if image_array.max() <= 1.0:
                    image_array = (image_array * 255).astype('uint8')
                else:
                    image_array = image_array.astype('uint8')
                
                # Convert grayscale to RGB if needed
                if len(image_array.shape) == 2:
                    image_array = np.stack([image_array] * 3, axis=-1)
                
                image = Image.fromarray(image_array)
            else:
                image = Image.open(uploaded_file)
            
            # Display uploaded image
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📷 Uploaded Image")
                st.image(image)
            
            with col2:
                st.markdown("#### 🔍 Analysis Results")
                
                # Check if model is loaded
                model_loaded = load_model()
                if model_loaded is None:
                    st.error("Cannot make prediction: Model not loaded")
                    return
                
                model, device = model_loaded
                
                # Show loading spinner
                with st.spinner('🔄 Analyzing brain scan...'):
                    # Preprocess and predict
                    image_tensor = preprocess_image(image)
                    predicted_idx, confidence, all_probs = predict(model, device, image_tensor)
                
                # Display results
                st.success("✅ Analysis Complete!")
            
            # Display detailed results below
            st.markdown("---")
            display_results(predicted_idx, confidence, all_probs)
            
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            st.info("Please ensure you're uploading a valid brain MRI image.")
    
    else:
        # Show instructions when no file uploaded
        st.info("👆 Please upload a brain MRI scan image to begin analysis.")
        
        # Show example
        st.markdown("### 💡 How to Use")
        st.markdown("""
        1. Click the **'Browse files'** button above
        2. Select a brain MRI scan image from your computer
        3. Wait for the AI to analyze the image
        4. View the results and recommendations
        
        **Supported formats:** PNG, JPG, JPEG, NPY
        """)

if __name__ == '__main__':
    main()
