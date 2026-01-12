"""
Home Page - Image Classification
=================================
Upload images and get predictions from trained models.
"""

import streamlit as st
from PIL import Image
import io
import requests
import os
import sys
from pathlib import Path
import plotly.express as px
import pandas as pd

# Paths
STREAMLIT_DIR = Path(__file__).parent.parent.resolve()
PROJECT_ROOT = STREAMLIT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODELS_DIR

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Classify - Intel Image Classifier",
    page_icon="🏠",
    layout="wide"
)

API_URL = os.getenv("API_URL", "http://localhost:8000")

# -----------------------------------------------------------------------------
# CUSTOM CSS
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    .page-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .upload-box {
        border: 2px dashed #667eea;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        background: rgba(102, 126, 234, 0.05);
    }
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 2rem;
        color: white;
        text-align: center;
    }
    .result-class {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .result-confidence {
        font-size: 1.5rem;
        opacity: 0.9;
    }
    .metric-box {
        background: #1e1e2f;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #333;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
    }
    .metric-label {
        color: #888;
        font-size: 0.85rem;
    }
    .model-badge {
        display: inline-block;
        background: #667eea;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
@st.cache_data(ttl=60)
def get_available_models():
    """Fetch available models from API or local directory."""
    try:
        response = requests.get(f"{API_URL}/models", timeout=3)
        if response.status_code == 200:
            return response.json().get("models", [])
    except:
        pass
    # Fallback to local
    if MODELS_DIR.exists():
        return [{"filename": f.name, "type": "cnn"} for f in MODELS_DIR.glob("*.keras")]
    return []


def get_model_type(filename):
    """Get model type from filename."""
    if "tl_" in filename or "transfer" in filename.lower():
        return "Transfer Learning"
    elif "regularized" in filename:
        return "Regularized CNN"
    else:
        return "Baseline CNN"


def create_probability_chart(predictions):
    """Create a horizontal bar chart for predictions."""
    df = pd.DataFrame([
        {"Class": k, "Probability": float(v.rstrip('%')) if isinstance(v, str) else v * 100}
        for k, v in predictions.items()
    ]).sort_values("Probability", ascending=True)
    
    fig = px.bar(
        df, x="Probability", y="Class", orientation='h',
        color="Probability",
        color_continuous_scale=["#1e1e2f", "#667eea", "#764ba2"],
        range_color=[0, 100]
    )
    fig.update_layout(
        showlegend=False,
        coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=10, b=0),
        height=250,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ccc'),
        xaxis=dict(range=[0, 100], gridcolor='#333'),
        yaxis=dict(gridcolor='#333')
    )
    fig.update_traces(marker_line_width=0)
    return fig


# -----------------------------------------------------------------------------
# SESSION STATE
# -----------------------------------------------------------------------------
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🏠 Classify")
    st.markdown("---")
    
    # Model selection
    st.markdown("### Select Model")
    models = get_available_models()
    model_names = [m['filename'] for m in models] if models else []
    
    if model_names:
        # Group models by type
        baseline_models = [m for m in model_names if "baseline" in m]
        regularized_models = [m for m in model_names if "regularized" in m]
        tl_models = [m for m in model_names if "tl_" in m]
        
        model_type = st.radio(
            "Model Type",
            ["Baseline", "Regularized", "Transfer Learning"],
            horizontal=True
        )
        
        if model_type == "Baseline" and baseline_models:
            selected_model = st.selectbox("Version", baseline_models, label_visibility="collapsed")
        elif model_type == "Regularized" and regularized_models:
            selected_model = st.selectbox("Version", regularized_models, label_visibility="collapsed")
        elif model_type == "Transfer Learning" and tl_models:
            selected_model = st.selectbox("Version", tl_models, label_visibility="collapsed")
        else:
            selected_model = st.selectbox("All Models", model_names)
    else:
        st.warning("No models found")
        selected_model = None
    
    st.markdown("---")
    st.markdown("### Scene Classes")
    classes_info = ["🏢 Buildings", "🌲 Forest", "🏔️ Glacier", "⛰️ Mountain", "🌊 Sea", "🛣️ Street"]
    for c in classes_info:
        st.markdown(f"- {c}")

# -----------------------------------------------------------------------------
# MAIN CONTENT
# -----------------------------------------------------------------------------
st.markdown('<h1 class="page-title">🏠 Image Classification</h1>', unsafe_allow_html=True)
st.markdown("Upload an image to classify it into one of 6 scene categories.")

# Two columns: Upload | Results
col_upload, col_results = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown("### 📤 Upload Image")
    
    uploaded_file = st.file_uploader(
        "Drag and drop or click to upload",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
        
        # Classify button
        if selected_model:
            if st.button("🔍 Classify Image", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    try:
                        img_byte_arr = io.BytesIO()
                        image.save(img_byte_arr, format=image.format or 'PNG')
                        img_bytes = img_byte_arr.getvalue()
                        
                        files = {"file": ("image.jpg", img_bytes, "image/jpeg")}
                        response = requests.post(
                            f"{API_URL}/predict/{selected_model}",
                            files=files,
                            timeout=30
                        )
                        response.raise_for_status()
                        st.session_state.prediction_result = response.json()
                        st.session_state.uploaded_image = image
                        st.rerun()
                    except requests.exceptions.ConnectionError:
                        st.error("❌ Cannot connect to API. Make sure it's running on port 8000.")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("Select a model from the sidebar first.")
    else:
        st.markdown("""
        <div class="upload-box">
            <p style="font-size:3rem; margin:0;">📷</p>
            <p style="color:#888;">Drop an image here or click to browse</p>
            <p style="color:#666; font-size:0.8rem;">Supports JPG, JPEG, PNG</p>
        </div>
        """, unsafe_allow_html=True)

with col_results:
    st.markdown("### 📊 Results")
    
    if st.session_state.prediction_result:
        result = st.session_state.prediction_result
        pred = result.get('prediction', {})
        pred_class = pred.get('class', 'Unknown')
        confidence_str = pred.get('confidence', '0%')
        inference_time = pred.get('inference_time_ms', 0)
        
        # Parse confidence
        if isinstance(confidence_str, str) and '%' in confidence_str:
            confidence = float(confidence_str.rstrip('%'))
        else:
            confidence = float(confidence_str) * 100
        
        # Result card
        st.markdown(f"""
        <div class="result-card">
            <div class="result-class">{pred_class.upper()}</div>
            <div class="result-confidence">{confidence:.1f}% confidence</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Metrics row
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{inference_time}</div>
                <div class="metric-label">ms inference</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{confidence:.0f}%</div>
                <div class="metric-label">confidence</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            model_type = get_model_type(result.get('model_used', ''))
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label" style="margin-top:0.5rem;">{model_type}</div>
                <div class="metric-label">model type</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Probability chart
        if 'all_predictions' in result:
            st.markdown("**All Probabilities**")
            fig = create_probability_chart(result['all_predictions'])
            st.plotly_chart(fig, use_container_width=True)
        
        # Feedback section
        st.markdown("---")
        st.markdown("### 💬 Provide Feedback")
        
        if "feedback_submitted" not in st.session_state:
            st.session_state.feedback_submitted = False
        
        if st.session_state.feedback_submitted:
            st.success("✅ Thank you! Your feedback has been recorded.")
            if st.button("Submit another feedback"):
                st.session_state.feedback_submitted = False
                st.rerun()
        else:
            st.markdown("Was the prediction correct?")
            
            # Quick buttons row
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                if st.button("✅ Correct", type="primary", use_container_width=True):
                    try:
                        feedback_data = {
                            "predicted": pred_class,
                            "true_class": pred_class,  # Same as predicted = correct
                            "model": result.get('model_used', 'unknown'),
                            "confidence": confidence / 100
                        }
                        resp = requests.post(f"{API_URL}/feedback", json=feedback_data, timeout=5)
                        if resp.status_code == 200:
                            st.session_state.feedback_submitted = True
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
            
            with col2:
                if st.button("❌ Wrong", use_container_width=True):
                    st.session_state.show_class_selector = True
                    st.rerun()
            
            # Show class selector if user clicked "Wrong"
            if st.session_state.get("show_class_selector", False):
                st.markdown("Select the correct class:")
                classes = list(result.get('all_predictions', {}).keys())
                if not classes:
                    classes = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
                
                col_a, col_b = st.columns([2, 1])
                with col_a:
                    true_class = st.selectbox(
                        "Correct class",
                        classes,
                        label_visibility="collapsed"
                    )
                with col_b:
                    if st.button("📤 Submit", type="primary", use_container_width=True):
                        try:
                            feedback_data = {
                                "predicted": pred_class,
                                "true_class": true_class,
                                "model": result.get('model_used', 'unknown'),
                                "confidence": confidence / 100
                            }
                            resp = requests.post(f"{API_URL}/feedback", json=feedback_data, timeout=5)
                            if resp.status_code == 200:
                                st.session_state.feedback_submitted = True
                                st.session_state.show_class_selector = False
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Clear button
        if st.button("🗑️ Clear Results", use_container_width=True):
            st.session_state.prediction_result = None
            st.session_state.uploaded_image = None
            st.session_state.feedback_submitted = False
            st.session_state.show_class_selector = False
            st.rerun()
    else:
        st.markdown("""
        <div style="text-align:center; padding:3rem; color:#666;">
            <p style="font-size:4rem; margin:0;">🔮</p>
            <p>Upload an image to see predictions</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("Intel Image Classification • Powered by TensorFlow")
