"""
Data Analysis Page
===================
Explore dataset statistics and sample images.
"""

import streamlit as st
import sys
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Paths
STREAMLIT_DIR = Path(__file__).parent.parent.resolve()
PROJECT_ROOT = STREAMLIT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_train_dir, get_test_dir, CLASS_NAMES

# Initialize session state for custom paths
if "custom_train_path" not in st.session_state:
    st.session_state.custom_train_path = str(get_train_dir())
if "custom_test_path" not in st.session_state:
    st.session_state.custom_test_path = str(get_test_dir())
if "use_custom_paths" not in st.session_state:
    st.session_state.use_custom_paths = False

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Data - Intel Image Classifier",
    page_icon="📈",
    layout="wide"
)

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
    .stat-card {
        background: #1e1e2f;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #333;
    }
    .stat-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #667eea;
    }
    .stat-label {
        color: #888;
        font-size: 0.9rem;
    }
    .class-card {
        background: #1e1e2f;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #333;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def get_data_paths():
    """Get train and test paths based on settings."""
    if st.session_state.use_custom_paths:
        train_path = Path(st.session_state.custom_train_path)
        test_path = Path(st.session_state.custom_test_path)
    else:
        train_path = get_train_dir()
        test_path = get_test_dir()
    return train_path, test_path

@st.cache_data
def count_images(directory: Path):
    """Count images in each class directory."""
    counts = {}
    if directory and directory.exists():
        for cls in CLASS_NAMES:
            class_dir = directory / cls
            if class_dir.exists():
                counts[cls] = len(list(class_dir.glob("*.jpg"))) + len(list(class_dir.glob("*.png")))
            else:
                counts[cls] = 0
    else:
        counts = {cls: 0 for cls in CLASS_NAMES}
    return counts


@st.cache_data
def get_sample_images(directory: Path, n_per_class: int = 1):
    """Get sample image paths from each class."""
    samples = {}
    if directory and directory.exists():
        for cls in CLASS_NAMES:
            class_dir = directory / cls
            if class_dir.exists():
                images = list(class_dir.glob("*.jpg"))[:n_per_class]
                if not images:
                    images = list(class_dir.glob("*.png"))[:n_per_class]
                samples[cls] = [str(img) for img in images]
    return samples


def create_distribution_chart(counts, title):
    """Create a bar chart for class distribution."""
    df = pd.DataFrame([
        {"Class": k, "Count": v} for k, v in counts.items()
    ])
    
    fig = px.bar(
        df, x="Class", y="Count",
        color="Count",
        color_continuous_scale=["#1e1e2f", "#667eea", "#764ba2"]
    )
    fig.update_layout(
        title=title,
        showlegend=False,
        coloraxis_showscale=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ccc'),
        xaxis=dict(gridcolor='#333'),
        yaxis=dict(gridcolor='#333'),
        margin=dict(l=0, r=0, t=40, b=0),
        height=300
    )
    return fig


def create_pie_chart(counts, title):
    """Create a pie chart for class distribution."""
    fig = px.pie(
        names=list(counts.keys()),
        values=list(counts.values()),
        color_discrete_sequence=px.colors.sequential.Purples_r
    )
    fig.update_layout(
        title=title,
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ccc'),
        margin=dict(l=0, r=0, t=40, b=0),
        height=300
    )
    return fig


# -----------------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------------
train_dir, test_dir = get_data_paths()

train_counts = count_images(train_dir)
test_counts = count_images(test_dir)

total_train = sum(train_counts.values())
total_test = sum(test_counts.values())
total_images = total_train + total_test

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📈 Data Analysis")
    st.markdown("---")
    
    st.markdown(f"**{total_images:,}** Total Images")
    st.markdown(f"**{total_train:,}** Training")
    st.markdown(f"**{total_test:,}** Testing")
    st.markdown(f"**{len(CLASS_NAMES)}** Classes")
    
    st.markdown("---")
    st.markdown("### Classes")
    for cls in CLASS_NAMES:
        emoji = {"buildings": "🏢", "forest": "🌲", "glacier": "🏔️", 
                 "mountain": "⛰️", "sea": "🌊", "street": "🛣️"}.get(cls, "📷")
        st.markdown(f"- {emoji} {cls.title()}")

# -----------------------------------------------------------------------------
# MAIN CONTENT
# -----------------------------------------------------------------------------
st.markdown('<h1 class="page-title">📈 Data Analysis</h1>', unsafe_allow_html=True)
st.markdown("Explore the Intel Image Classification dataset statistics.")

# Settings link
if st.session_state.use_custom_paths:
    st.info(f"📁 Using custom paths. [Change in Settings](/_Settings)")
else:
    st.info(f"📁 Using default paths. [Configure custom paths in Settings](/_Settings)")

# Stats row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-value">{total_images:,}</div>
        <div class="stat-label">Total Images</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-value">{total_train:,}</div>
        <div class="stat-label">Training Set</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-value">{total_test:,}</div>
        <div class="stat-label">Test Set</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-value">{len(CLASS_NAMES)}</div>
        <div class="stat-label">Classes</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Distribution", "🖼️ Samples", "📋 Details"])

# -----------------------------------------------------------------------------
# TAB 1: DISTRIBUTION
# -----------------------------------------------------------------------------
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Training Set")
        if total_train > 0:
            fig = create_distribution_chart(train_counts, "Images per Class")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Training data not found")
    
    with col2:
        st.markdown("### Test Set")
        if total_test > 0:
            fig = create_distribution_chart(test_counts, "Images per Class")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Test data not found")
    
    # Pie charts
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if total_train > 0:
            fig = create_pie_chart(train_counts, "Training Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if total_test > 0:
            fig = create_pie_chart(test_counts, "Test Distribution")
            st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 2: SAMPLES
# -----------------------------------------------------------------------------
with tab2:
    st.markdown("### Sample Images by Class")
    
    if train_dir and train_dir.exists():
        samples = get_sample_images(train_dir, n_per_class=2)
        
        cols = st.columns(3)
        for idx, cls in enumerate(CLASS_NAMES):
            with cols[idx % 3]:
                emoji = {"buildings": "🏢", "forest": "🌲", "glacier": "🏔️", 
                         "mountain": "⛰️", "sea": "🌊", "street": "🛣️"}.get(cls, "📷")
                st.markdown(f"**{emoji} {cls.title()}**")
                
                if cls in samples and samples[cls]:
                    for img_path in samples[cls]:
                        st.image(img_path, use_container_width=True)
                else:
                    st.info("No samples")
    else:
        st.warning("Training data directory not found. Cannot display sample images.")

# -----------------------------------------------------------------------------
# TAB 3: DETAILS
# -----------------------------------------------------------------------------
with tab3:
    st.markdown("### Dataset Details")
    
    # Combined table
    df = pd.DataFrame({
        "Class": CLASS_NAMES,
        "Train Images": [train_counts.get(c, 0) for c in CLASS_NAMES],
        "Test Images": [test_counts.get(c, 0) for c in CLASS_NAMES],
        "Total": [train_counts.get(c, 0) + test_counts.get(c, 0) for c in CLASS_NAMES]
    })
    df["Train %"] = (df["Train Images"] / df["Train Images"].sum() * 100).round(1)
    df["Test %"] = (df["Test Images"] / df["Test Images"].sum() * 100).round(1)
    
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Paths info
    st.markdown("### Data Paths")
    st.code(f"Training: {train_dir}")
    st.code(f"Testing: {test_dir}")

# Footer
st.markdown("---")
st.caption("Intel Image Classification • Data Analysis")