"""
Intel Image Classification Dashboard
=====================================
Streamlit dashboard for image classification with CNN models.
"""

import streamlit as st
import os
import sys
import requests
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODELS_DIR, FIGURES_DIR

API_URL = os.getenv("API_URL", "http://localhost:8000")

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Intel Image Classifier",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Intel Image Classification Dashboard - Built with Streamlit & TensorFlow"
    }
)

# -----------------------------------------------------------------------------
# CUSTOM CSS - Professional Dark Theme
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Main container */
    .block-container { padding: 2rem 3rem; }
    
    /* Headers */
    .main-title {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Cards */
    .info-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border: 1px solid #333;
    }
    .stat-card {
        background: #1e1e2f;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #333;
    }
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    .stat-label {
        color: #888;
        font-size: 0.9rem;
    }
    
    /* Feature cards */
    .feature-card {
        background: #1e1e2f;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem;
        border-left: 4px solid #667eea;
        transition: transform 0.2s;
    }
    .feature-card:hover {
        transform: translateX(5px);
    }
    .feature-icon { font-size: 1.5rem; margin-bottom: 0.5rem; }
    .feature-title { font-weight: 600; color: #fff; margin-bottom: 0.3rem; }
    .feature-desc { color: #888; font-size: 0.9rem; }
    
    /* Status badges */
    .status-online { color: #4ade80; }
    .status-offline { color: #f87171; }
    
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0f0f1a 100%);
    }
    [data-testid="stSidebar"] .stMarkdown { color: #ccc; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
@st.cache_data(ttl=30)
def check_api_status():
    """Check if API is running."""
    try:
        resp = requests.get(f"{API_URL}/", timeout=2)
        if resp.status_code == 200:
            data = resp.json()
            return True, len(data.get("models_available", []))
        return False, 0
    except:
        return False, 0

def count_models():
    """Count available model files."""
    if MODELS_DIR.exists():
        return len(list(MODELS_DIR.glob("*.keras")))
    return 0

def count_training_runs():
    """Count training runs with metadata."""
    count = 0
    if FIGURES_DIR.exists():
        for d in FIGURES_DIR.iterdir():
            if d.is_dir() and (d / "run_metadata.json").exists():
                count += 1
    return count

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🖼️ Intel Classifier")
    st.markdown("---")
    
    # System Status
    api_online, api_models = check_api_status()
    local_models = count_models()
    training_runs = count_training_runs()
    
    st.markdown("### System Status")
    
    col1, col2 = st.columns(2)
    with col1:
        if api_online:
            st.markdown("🟢 **API Online**")
        else:
            st.markdown("🔴 **API Offline**")
    with col2:
        st.markdown(f"📦 **{local_models} Models**")
    
    st.markdown(f"📊 **{training_runs} Training Runs**")
    
    st.markdown("---")
    st.markdown("### Quick Links")
    st.markdown("""
    - 🏠 **Home** - Classify images
    - 📊 **Comparison** - Compare models
    - 📈 **Analysis** - View data & figures
    - 💬 **Feedback** - User feedback
    """)
    
    st.markdown("---")
    st.caption("v2.0 • TensorFlow + Streamlit")

# -----------------------------------------------------------------------------
# MAIN CONTENT
# -----------------------------------------------------------------------------
st.markdown('<h1 class="main-title">Intel Image Classification</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Classify natural scenes using deep learning models</p>', unsafe_allow_html=True)

# Stats row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-value">{local_models}</div>
        <div class="stat-label">Trained Models</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-value">{training_runs}</div>
        <div class="stat-label">Training Runs</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-value">6</div>
        <div class="stat-label">Scene Classes</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    status = "Online" if api_online else "Offline"
    color = "#4ade80" if api_online else "#f87171"
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-value" style="color: {color};">●</div>
        <div class="stat-label">API {status}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Feature cards
st.markdown("### 🚀 Features")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🏠</div>
        <div class="feature-title">Image Classification</div>
        <div class="feature-desc">Upload any image and get instant predictions with confidence scores for all 6 scene categories.</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📊</div>
        <div class="feature-title">Model Comparison</div>
        <div class="feature-desc">Compare accuracy and performance across Baseline, Regularized, and Transfer Learning models.</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📈</div>
        <div class="feature-title">Training Analytics</div>
        <div class="feature-desc">View training curves, confusion matrices, and per-class metrics from all training runs.</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🔄</div>
        <div class="feature-title">Version Control</div>
        <div class="feature-desc">Each training run is timestamped. Compare different model versions side by side.</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Model info
st.markdown("### 🧠 Available Models")
model_data = {
    "Model": ["Baseline CNN", "Regularized CNN", "Transfer Learning"],
    "Architecture": ["3-Block Conv + Dense", "Conv + Dropout + L2", "MobileNetV2 + Custom Head"],
    "Input Size": ["150×150", "150×150", "224×224"],
    "Parameters": ["~500K", "~500K", "~3.5M"],
    "Best For": ["Quick training", "Prevent overfitting", "Best accuracy"]
}
st.table(model_data)

# Classes
st.markdown("### 🏷️ Scene Categories")
classes = ["🏢 Buildings", "🌲 Forest", "🏔️ Glacier", "⛰️ Mountain", "🌊 Sea", "🛣️ Street"]
cols = st.columns(6)
for i, cls in enumerate(classes):
    with cols[i]:
        st.markdown(f"<div style='text-align:center; padding:1rem; background:#1e1e2f; border-radius:8px;'>{cls}</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#666; padding:1rem;">
    <p>Built with ❤️ using Streamlit & TensorFlow</p>
    <p style="font-size:0.8rem;">Intel Image Classification Dataset • 14,000+ Training Images</p>
</div>
""", unsafe_allow_html=True)
