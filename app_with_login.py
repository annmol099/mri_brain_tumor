"""
Brain Tumor Detection Web Application - WITH LOGIN SYSTEM
======================================
Upload an MRI brain scan and get instant tumor detection results!
+ User login & prediction history database
"""

import os
import sys
from contextlib import contextmanager

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
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import database functions
from database import (
    init_database, register_user, login_user, 
    save_prediction, get_user_predictions, get_all_statistics
)

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'final_model_20251106_142153.pth')

from model_architecture import ResNet50Classifier

# Initialize database
init_database()

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
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(224, 242, 254, 0.95), transparent 28%),
            radial-gradient(circle at top right, rgba(240, 253, 244, 0.9), transparent 24%),
            linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .main-header {
        font-size: 3rem;
        color: #111827;
        text-align: center;
        margin-bottom: 0.4rem;
        font-weight: 800;
        letter-spacing: -0.03em;
    }
    .sub-header {
        text-align: center;
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
    }
    .result-box {
        padding: 2rem;
        border-radius: 18px;
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
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 55%, #0ea5e9 100%);
        color: white;
        padding: 1.6rem 1.8rem;
        border-radius: 24px;
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.18);
        margin: 0.75rem 0 1.4rem 0;
    }
    .hero-title {
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
    }
    .hero-text {
        color: rgba(255, 255, 255, 0.88);
        font-size: 0.98rem;
        margin-bottom: 0.9rem;
    }
    .feature-chip {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.14);
        margin-right: 0.45rem;
        margin-bottom: 0.35rem;
        font-size: 0.82rem;
        backdrop-filter: blur(10px);
    }
    .auth-card {
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 22px;
        padding: 1.1rem 1.1rem 1.3rem 1.1rem;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
        backdrop-filter: blur(12px);
    }
    .panel-title {
        font-size: 1.25rem;
        font-weight: 800;
        color: #111827;
        margin-bottom: 0.2rem;
    }
    .panel-subtitle {
        color: #64748b;
        margin-bottom: 1rem;
        font-size: 0.92rem;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.42rem 0.75rem;
        border-radius: 999px;
        background: #eef2ff;
        color: #4338ca;
        font-weight: 600;
        font-size: 0.82rem;
        margin-bottom: 0.8rem;
    }
    .metric-card {
        background: white;
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 18px;
        padding: 0.85rem 1rem;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.05);
    }
    .sidebar-card {
        background: white;
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 18px;
        padding: 1rem;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.05);
        margin-bottom: 0.9rem;
    }
    .section-title {
        font-size: 1.25rem;
        font-weight: 800;
        color: #111827;
        margin-bottom: 0.65rem;
    }
    div[data-testid="stButton"] > button {
        border-radius: 12px;
        border: none;
        padding: 0.7rem 1.1rem;
        font-weight: 700;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 22px rgba(59, 130, 246, 0.18);
    }
    div[data-baseweb="input"] input {
        border-radius: 12px;
    }
    .login-box {
        max-width: 540px;
        margin: auto;
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
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found: {MODEL_PATH}")
        st.info("Please ensure the trained model is in the 'models' folder.")
        return None
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ResNet50Classifier(num_classes=4, pretrained=False)

        @contextmanager
        def _portable_pathlib_patch():
            """Allow loading checkpoints saved on different OS path implementations."""
            original_windows_path = pathlib.WindowsPath
            original_posix_path = pathlib.PosixPath
            try:
                pathlib.WindowsPath = pathlib.PureWindowsPath
                pathlib.PosixPath = pathlib.PurePosixPath
                yield
            finally:
                pathlib.WindowsPath = original_windows_path
                pathlib.PosixPath = original_posix_path

        def _portable_torch_load(weights_only_flag):
            """Try to load checkpoint with mmap for lower memory, then fallback for older torch versions."""
            with _portable_pathlib_patch():
                try:
                    return torch.load(
                        MODEL_PATH,
                        map_location=device,
                        weights_only=weights_only_flag,
                        mmap=True
                    )
                except TypeError:
                    try:
                        return torch.load(
                            MODEL_PATH,
                            map_location=device,
                            weights_only=weights_only_flag
                        )
                    except TypeError:
                        return torch.load(MODEL_PATH, map_location=device)

        try:
            checkpoint = _portable_torch_load(weights_only_flag=True)
        except Exception:
            checkpoint = _portable_torch_load(weights_only_flag=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
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
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image_tensor = transform(image).unsqueeze(0)
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
    """Display prediction results in a modern format"""
    tumor_type = CLASS_NAMES[predicted_class]
    info = TUMOR_INFO[tumor_type]
    confidence_pct = confidence * 100
    
    if tumor_type == 'No Tumor':
        result_bg = "#ecfdf5"
        result_border = "#10b981"
        result_icon = "✅"
    else:
        result_bg = "#fef2f2"
        result_border = "#ef4444"
        result_icon = "⚠️"
    
    st.markdown(
        f"""<div style="background:{result_bg}; border:2px solid {result_border}; border-radius:16px; padding:1.5rem; margin:1rem 0;">
        <div style="font-size:1.3rem; font-weight:800; color:#111827;">{result_icon} {tumor_type}</div>
        <div style="font-size:1.9rem; font-weight:900; color:{result_border}; margin:0.5rem 0;">{confidence_pct:.1f}%</div>
        <div style="font-size:0.9rem; color:#475569; line-height:1.6;">{info['description']}</div>
        <div style="background:rgba(0,0,0,0.05); padding:0.8rem; border-radius:8px; margin-top:0.8rem; font-weight:600;color:#1f2937;">Severity: {info['severity']}</div>
        </div>""",
        unsafe_allow_html=True
    )
    
    st.markdown(f"💡 {info['recommendation']}", unsafe_allow_html=True)
    st.markdown("### 📊 Full Breakdown")
    for idx, (class_name, prob) in enumerate(zip(CLASS_NAMES, all_probabilities)):
        prob_pct = prob * 100
        is_predicted = idx == predicted_class
        bar_color = "#3b82f6" if is_predicted else "#cbd5e1"
        # Custom HTML progress bar for consistent sizing and rounded look
        st.markdown(
            f"""
            <div style="margin-bottom:0.9rem;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.35rem;">
                    <div style="font-weight:700; color:#0f172a;">{class_name}:</div>
                    <div style="font-family:monospace; color:#065f46;">{prob_pct:.1f}% {'👈 PREDICTED' if is_predicted else ''}</div>
                </div>
                <div style="background:#f1f5f9; border-radius:12px; height:14px; overflow:hidden;">
                    <div style="width:{prob_pct:.2f}%; background:{bar_color}; height:100%; border-radius:12px;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

def show_login_modal():
    """Show login/register modal when user tries to upload without logging in"""
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 55%, #0ea5e9 100%); color:white; padding:1.8rem; border-radius:22px; box-shadow: 0 18px 40px rgba(15,23,42,0.18); margin:1rem 0;">
            <div style="font-size:1.6rem;font-weight:800;margin-bottom:0.5rem">🔐 Sign in Required</div>
            <div style="color:rgba(255,255,255,0.88);font-size:0.95rem">To upload and save your MRI predictions, please login or create a new account below.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    login_tab, register_tab = st.tabs(["🔐 Login", "📝 Register"])
    
    with login_tab:
        st.markdown('<div class="auth-panel">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:1.1rem;font-weight:800;color:#111827;margin-bottom:0.5rem">Login to your account</div>', unsafe_allow_html=True)
        
        email = st.text_input("📧 Email", key="login_email_modal", placeholder="you@example.com")
        password = st.text_input("🔑 Password", type="password", key="login_password_modal", placeholder="Enter your password")
        
        if st.button("Login", key="login_btn_modal", use_container_width=True):
            if email and password:
                success, result = login_user(email, password)
                if success:
                    st.session_state['logged_in'] = True
                    st.session_state['user'] = result
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error(result)
            else:
                st.error("❌ Please enter email and password")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with register_tab:
        st.markdown('<div class="auth-panel">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:1.1rem;font-weight:800;color:#111827;margin-bottom:0.5rem">Create a new account</div>', unsafe_allow_html=True)
        
        name = st.text_input("👤 Full Name", key="reg_name_modal", placeholder="Your full name")
        email = st.text_input("📧 Email", key="reg_email_modal", placeholder="you@example.com")
        phone = st.text_input("📱 Phone Number", key="reg_phone_modal", placeholder="03xxxxxxxxx")
        password = st.text_input("🔑 Password", type="password", key="reg_password_modal", placeholder="Create a password")
        confirm_password = st.text_input("🔑 Confirm Password", type="password", key="reg_confirm_modal", placeholder="Re-enter password")
        
        if st.button("Register", key="register_btn_modal", use_container_width=True):
            if not all([name, email, phone, password, confirm_password]):
                st.error("❌ All fields are required")
            elif password != confirm_password:
                st.error("❌ Passwords don't match")
            else:
                success, message = register_user(name, email, phone, password)
                if success:
                    st.success(message)
                    st.info("👉 Now use the Login tab to sign in!")
                else:
                    st.error(message)
        
        st.markdown('</div>', unsafe_allow_html=True)

def show_upload_landing():
    """Show upload interface (main landing page - no login required to view)"""
    st.markdown('<h1 class="main-header">🧠 Brain Tumor Detection AI</h1>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload an MRI brain scan to detect tumors instantly with 98.96% accuracy</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📤 Upload Brain MRI Scan")
    
    uploaded_file = st.file_uploader(
        "Choose an MRI image...",
        type=['png', 'jpg', 'jpeg', 'npy'],
        help="Upload a brain MRI scan in PNG, JPG, JPEG, or NPY format"
    )
    
    if uploaded_file is not None:
        # Check if user is logged in
        if not st.session_state.get('logged_in', False):
            st.warning("⚠️ You need to login to upload and save results!")
            st.markdown("---")
            show_login_modal()
        else:
            # User is logged in, process upload
            if uploaded_file.name.endswith('.npy'):
                img_array = np.load(uploaded_file)
                img_array = (img_array * 255).astype(np.uint8)
                img = Image.fromarray(img_array)
            else:
                img = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1.2, 1.8])
            
            with col1:
                st.markdown('<div class="image-display">', unsafe_allow_html=True)
                st.image(img, caption="📷 MRI Scan", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                model_result = load_model()
                if model_result is None:
                    st.error("❌ Cannot load model")
                    return
                
                model, device = model_result
                
                st.markdown('<div style="padding:1rem; background:#f0f4ff; border-radius:14px; border-left:4px solid #3b82f6;">', unsafe_allow_html=True)
                st.markdown('<div style="font-size:0.9rem;color:#475569;font-weight:600;">⏳ ANALYZING SCAN</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                with st.spinner('🔬 Running AI analysis...'):
                    image_tensor = preprocess_image(img)
                    predicted_class, confidence, all_probs = predict(model, device, image_tensor)
                
                st.markdown('<div style="padding:1rem; background:#f0fdf4; border-radius:14px; border-left:4px solid #22c55e; margin-top:1rem;"><div style="font-size:0.9rem;color:#166534;font-weight:600;">✅ Analysis Complete</div></div>', unsafe_allow_html=True)
            
            st.markdown('<div class="result-section">', unsafe_allow_html=True)
            display_results(predicted_class, confidence, all_probs)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Save to database
            tumor_type = CLASS_NAMES[predicted_class]
            image_name = uploaded_file.name
            user_id = st.session_state['user']['user_id']
            
            success, message = save_prediction(
                user_id, 
                image_name, 
                tumor_type, 
                confidence
            )
            
            if success:
                st.success(message)
            else:
                st.warning(message)
    else:
        # No file uploaded - show info
        col_space1, col_info, col_space2 = st.columns([0.5, 2, 0.5])
        with col_info:
            st.markdown(
                """<div style="text-align:center; padding:2rem; background:white; border-radius:16px; border:1px solid #e2e8f0; box-shadow: 0 10px 25px rgba(15,23,42,0.05);">
                <div style="font-size:3rem; margin-bottom:0.5rem">📤</div>
                <div style="font-size:1.1rem; color:#334155; font-weight:600;">Ready to upload</div>
                <div style="font-size:0.9rem; color:#64748b; margin-top:0.3rem;">Click 'Browse files' to select an MRI scan</div>
                </div>""",
                unsafe_allow_html=True
            )
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                """<div style="background:#fff5f5; padding:1rem; border-radius:12px; border-left:4px solid #f87171;">
                <div style="font-size:0.8rem; color:#7f1d1d; font-weight:700;">🔴 GLIOMA</div>
                <div style="font-size:0.75rem; color:#991b1b; margin-top:0.3rem;">From glial cells</div>
                </div>""",
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                """<div style="background:#fff7ed; padding:1rem; border-radius:12px; border-left:4px solid #fb923c;">
                <div style="font-size:0.8rem; color:#7c2d12; font-weight:700;">🟠 MENINGIOMA</div>
                <div style="font-size:0.75rem; color:#9a3412; margin-top:0.3rem;">Brain membranes</div>
                </div>""",
                unsafe_allow_html=True
            )
        with col3:
            st.markdown(
                """<div style="background:#f0fdfa; padding:1rem; border-radius:12px; border-left:4px solid #2dd4bf;">
                <div style="font-size:0.8rem; color:#134e4a; font-weight:700;">🟢 NO TUMOR</div>
                <div style="font-size:0.75rem; color:#0d544c; margin-top:0.3rem;">Healthy scan</div>
                </div>""",
                unsafe_allow_html=True
            )

def main_app():
    """Main application after login"""
    # Header with user info
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown('<h1 class="main-header">🧠 Brain Tumor Detection AI</h1>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Upload MRI scans, get instant predictions, and save every result to your secure history.</div>', unsafe_allow_html=True)
    
    with col2:
        user_name = st.session_state['user']['name']
        st.markdown(f'<div class="metric-card"><div style="font-size:0.8rem;color:#64748b">Logged in as</div><div style="font-size:1.05rem;font-weight:800;color:#111827">{user_name}</div></div>', unsafe_allow_html=True)
        if st.button("🚪 Logout", key="logout_btn"):
            st.session_state['logged_in'] = False
            st.session_state['user'] = None
            st.rerun()
    
    # Sidebar
    with st.sidebar:
        tab = st.radio("📋 Menu", ["🏥 Predict", "📊 My History", "📈 Statistics"])
        
        st.markdown("---")
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.markdown("## ⚙️ About")
        st.info("""
        This AI model analyzes brain MRI scans and detects:
        - **Glioma** 🔴
        - **Meningioma** 🟠
        - **Pituitary Tumor** 🟣
        - **No Tumor** 🟢
        
        **Accuracy:** 98.96% on test data
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.markdown("## ⚠️ Disclaimer")
        st.warning("""
        This tool is for **educational purposes only** and should NOT replace professional medical diagnosis.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.markdown("## ✅ Quick Flow")
        st.write("1. Upload MRI")
        st.write("2. View prediction")
        st.write("3. Result auto-saves")
        st.write("4. Check history")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content
    if tab == "🏥 Predict":
        st.markdown("### 📤 Upload Brain MRI Scan")
        st.markdown("Upload a brain MRI image to detect if a tumor is present and identify its type.")
        
        uploaded_file = st.file_uploader(
            "Choose an MRI image...",
            type=['png', 'jpg', 'jpeg', 'npy'],
            help="Upload a brain MRI scan in PNG, JPG, JPEG, or NPY format"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            if uploaded_file.name.endswith('.npy'):
                img_array = np.load(uploaded_file)
                img_array = (img_array * 255).astype(np.uint8)
                img = Image.fromarray(img_array)
            else:
                img = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1.2, 1.8])
            
            with col1:
                st.markdown('<div style="border-radius:12px; overflow:hidden; box-shadow:0 10px 30px rgba(15,23,42,0.12);">', unsafe_allow_html=True)
                st.image(img, caption="🧠 MRI Scan", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div style="padding:1rem; background:#f0f4ff; border-radius:12px; border-left:4px solid #3b82f6; margin-bottom:1rem;">', unsafe_allow_html=True)
                st.markdown('<div style="font-size:0.85rem; color:#3730a3; font-weight:700;">⚡ ANALYZING SCAN</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                model_result = load_model()
                if model_result is None:
                    st.error("❌ Cannot load model")
                    return
                
                model, device = model_result
                
                with st.spinner('🔬 Running AI analysis...'):
                    image_tensor = preprocess_image(img)
                    predicted_class, confidence, all_probs = predict(model, device, image_tensor)
                
                st.markdown('<div style="padding:1rem; background:#f0fdf4; border-radius:12px; border-left:4px solid #22c55e;">', unsafe_allow_html=True)
                st.markdown('<div style="font-size:0.85rem; color:#15803d; font-weight:700;">✅ Analysis Complete</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            display_results(predicted_class, confidence, all_probs)
            
            # Save to database
            tumor_type = CLASS_NAMES[predicted_class]
            image_name = uploaded_file.name
            user_id = st.session_state['user']['user_id']
            
            success, message = save_prediction(
                user_id, 
                image_name, 
                tumor_type, 
                confidence
            )
            
            if success:
                st.success(message)
            else:
                st.warning(message)
    
    elif tab == "📊 My History":
        st.markdown("### 📋 Your Prediction History")
        
        user_id = st.session_state['user']['user_id']
        predictions = get_user_predictions(user_id)
        
        if predictions:
            st.markdown(f"**Total Predictions:** {len(predictions)}")
            
            # Display as table
            for pred in predictions:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.write(f"📸 {pred[1]}")
                with col2:
                    st.write(f"🔍 {pred[2]}")
                with col3:
                    st.write(f"📊 {pred[3]*100:.2f}%")
                with col4:
                    st.write(f"⏰ {pred[4]}")
                st.divider()
        else:
            st.info("No predictions yet. Go to 'Predict' to upload an MRI scan!")
    
    elif tab == "📈 Statistics":
        st.markdown("### 📊 Database Statistics")
        
        stats = get_all_statistics()
        if stats:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("👥 Total Users", stats['total_users'])
                st.metric("🔍 Total Predictions", stats['total_predictions'])
            
            with col2:
                st.markdown("**Tumor Distribution:**")
                for tumor_type, count in stats['tumor_distribution']:
                    st.write(f"- {tumor_type}: {count}")

# Main execution
def main():
    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
        st.session_state['user'] = None
    
    # Check login status
    if st.session_state['logged_in']:
        # Show dashboard with navigation
        main_app()
    else:
        # Show upload landing page (login prompt appears only when trying to upload)
        show_upload_landing()

if __name__ == "__main__":
    main()
