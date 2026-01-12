"""
Model Comparison Page
======================
Compare training runs, view figures, and manage models.
"""

import streamlit as st
import pandas as pd
import json
import sys
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# Paths
STREAMLIT_DIR = Path(__file__).parent.parent.resolve()
PROJECT_ROOT = STREAMLIT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODELS_DIR, FIGURES_DIR

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Compare - Intel Image Classifier",
    page_icon="📊",
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
    .run-card {
        background: #1e1e2f;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #333;
        margin-bottom: 1rem;
    }
    .run-timestamp {
        font-size: 1.2rem;
        font-weight: 600;
        color: #667eea;
    }
    .accuracy-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        margin: 0.2rem;
    }
    .model-card {
        background: #1e1e2f;
        border-radius: 8px;
        padding: 1rem;
        border-left: 3px solid #667eea;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def get_training_runs():
    """Get all training runs with metadata."""
    runs = []
    if FIGURES_DIR.exists():
        for run_dir in sorted(FIGURES_DIR.iterdir(), reverse=True):
            if run_dir.is_dir():
                metadata_file = run_dir / "run_metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file) as f:
                            metadata = json.load(f)
                        runs.append({
                            "timestamp": run_dir.name,
                            "path": run_dir,
                            "metadata": metadata
                        })
                    except:
                        pass
    return runs


def get_all_models():
    """Get all model files."""
    models = []
    if MODELS_DIR.exists():
        for model_file in sorted(MODELS_DIR.glob("*.keras"), reverse=True):
            models.append({
                "filename": model_file.name,
                "path": model_file,
                "size_mb": model_file.stat().st_size / (1024 * 1024),
                "type": get_model_type(model_file.name)
            })
    return models


def get_model_type(filename):
    """Get model type from filename."""
    if "tl_" in filename:
        return "Transfer Learning"
    elif "regularized" in filename:
        return "Regularized"
    else:
        return "Baseline"


def create_accuracy_chart(runs_data):
    """Create comparison chart across runs."""
    data = []
    for run in runs_data:
        ts = run["timestamp"]
        # Format timestamp nicely: YYYYMMDD_HHMMSS -> YYYY-MM-DD HH:MM
        formatted_ts = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]} {ts[9:11]}:{ts[11:13]}" if len(ts) >= 13 else ts
        for model_name, info in run["metadata"].get("models", {}).items():
            data.append({
                "Run": formatted_ts,
                "Model": model_name,
                "Accuracy": info["accuracy"] * 100
            })
    
    if not data:
        return None
    
    df = pd.DataFrame(data)
    fig = px.bar(
        df, x="Run", y="Accuracy", color="Model",
        barmode="group",
        color_discrete_sequence=["#667eea", "#764ba2", "#4ade80"],
        text=df["Accuracy"].apply(lambda x: f"{x:.1f}%")
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ccc'),
        xaxis=dict(gridcolor='#333', type='category', title='Training Run'),
        yaxis=dict(gridcolor='#333', range=[0, 100], title='Accuracy (%)'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, title="Model Type"),
        margin=dict(l=0, r=0, t=30, b=0),
        height=400
    )
    fig.update_traces(textposition='outside')
    return fig


def create_models_accuracy_chart(models):
    """Create chart showing all models with their types."""
    data = []
    for m in models:
        # Extract accuracy from filename if possible, otherwise show size
        data.append({
            "Model": m["filename"].replace(".keras", ""),
            "Type": m["type"],
            "Size (MB)": m["size_mb"]
        })
    
    if not data:
        return None
    
    df = pd.DataFrame(data)
    fig = px.bar(
        df, x="Model", y="Size (MB)", color="Type",
        color_discrete_map={
            "Baseline": "#667eea",
            "Regularized": "#764ba2",
            "Transfer Learning": "#4ade80"
        },
        text=df["Size (MB)"].apply(lambda x: f"{x:.1f} MB")
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ccc'),
        xaxis=dict(gridcolor='#333', type='category', title='Model', tickangle=45),
        yaxis=dict(gridcolor='#333', title='Size (MB)'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, title="Model Type"),
        margin=dict(l=0, r=0, t=30, b=60),
        height=400
    )
    fig.update_traces(textposition='outside')
    return fig


# -----------------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------------
training_runs = get_training_runs()
all_models = get_all_models()

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📊 Compare")
    st.markdown("---")
    
    st.markdown(f"**{len(training_runs)}** Training Runs")
    st.markdown(f"**{len(all_models)}** Model Files")
    
    st.markdown("---")
    
    if training_runs:
        st.markdown("### Select Run")
        run_options = [r["timestamp"] for r in training_runs]
        selected_idx = st.selectbox(
            "Training Run",
            range(len(run_options)),
            format_func=lambda x: run_options[x],
            label_visibility="collapsed"
        )
        selected_run = training_runs[selected_idx]
    else:
        selected_run = None
        st.warning("No runs found")

# -----------------------------------------------------------------------------
# MAIN CONTENT
# -----------------------------------------------------------------------------
st.markdown('<h1 class="page-title">📊 Model Comparison</h1>', unsafe_allow_html=True)
st.markdown("Compare training runs, view metrics, and browse training figures.")

# Tabs
tab1, tab2, tab3 = st.tabs(["📈 Performance", "🖼️ Figures", "📦 Models"])

# -----------------------------------------------------------------------------
# TAB 1: PERFORMANCE
# -----------------------------------------------------------------------------
with tab1:
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        if training_runs:
            # Multi-run comparison chart
            st.markdown("### Accuracy Across Training Runs")
            chart = create_accuracy_chart(training_runs[:5])  # Last 5 runs
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            else:
                st.info("No accuracy data available in run metadata.")
        else:
            st.info("No training runs found. Run the training pipeline to generate results.")
    
    with col_right:
        st.markdown("### All Available Models")
        if all_models:
            for m in all_models:
                type_color = {"Baseline": "#667eea", "Regularized": "#764ba2", "Transfer Learning": "#4ade80"}
                st.markdown(f"""
                <div class="model-card" style="border-left-color: {type_color.get(m['type'], '#667eea')};">
                    <strong>{m['filename']}</strong><br>
                    <span style="color:#888; font-size:0.8rem;">{m['type']} • {m['size_mb']:.1f} MB</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No models found.")
    
    st.markdown("---")
    
    # Selected run details
    if training_runs and selected_run:
        st.markdown(f"### Selected Run: `{selected_run['timestamp']}`")
        
        metadata = selected_run["metadata"]
        models_data = metadata.get("models", {})
        
        if models_data:
            # Accuracy cards
            cols = st.columns(min(len(models_data), 4))
            for i, (model_name, info) in enumerate(models_data.items()):
                with cols[i % 4]:
                    acc = info["accuracy"] * 100
                    color = "#4ade80" if acc >= 90 else "#667eea" if acc >= 80 else "#f59e0b"
                    st.markdown(f"""
                    <div class="run-card" style="text-align:center;">
                        <div style="color:#888; font-size:0.9rem;">{model_name}</div>
                        <div style="font-size:2.5rem; font-weight:700; color:{color};">{acc:.1f}%</div>
                        <div style="color:#666; font-size:0.8rem;">{info.get('model_file', 'N/A')}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Details table
            st.markdown("### Run Details")
            df = pd.DataFrame([
                {
                    "Model": name,
                    "Accuracy": f"{info['accuracy']*100:.2f}%",
                    "File": info.get("model_file", "N/A")
                }
                for name, info in models_data.items()
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.warning("No model data in this run's metadata.")

# -----------------------------------------------------------------------------
# TAB 2: FIGURES
# -----------------------------------------------------------------------------
with tab2:
    # Get all figures from all runs and also from root figures directory
    all_figures = []
    
    # Figures from training runs
    if selected_run:
        run_path = selected_run["path"]
        all_figures.extend(list(run_path.glob("*.png")))
    
    # Also check root figures directory for any standalone figures
    if FIGURES_DIR.exists():
        all_figures.extend([f for f in FIGURES_DIR.glob("*.png") if f.is_file()])
    
    # Remove duplicates
    all_figures = list(set(all_figures))
    
    if all_figures:
        st.markdown(f"### Training Figures ({len(all_figures)} images)")
        
        # Filter by model type
        col1, col2 = st.columns([1, 3])
        with col1:
            model_filter = st.selectbox(
                "Filter by Model",
                ["All Models", "Baseline", "Regularized", "Transfer Learning"],
                key="fig_model_filter"
            )
        with col2:
            fig_type_filter = st.selectbox(
                "Filter by Type",
                ["All Types", "Training Curves", "Confusion Matrix", "Class Metrics", "Grad-CAM", "Comparison"],
                key="fig_type_filter"
            )
        
        # Apply filters
        filtered_figures = all_figures
        
        if model_filter != "All Models":
            filter_map = {"Baseline": "baseline", "Regularized": "regularized", "Transfer Learning": "tl_"}
            filter_key = filter_map.get(model_filter, "")
            filtered_figures = [f for f in filtered_figures if filter_key in f.name.lower()]
        
        if fig_type_filter != "All Types":
            type_map = {
                "Training Curves": "training_curves",
                "Confusion Matrix": "confusion_matrix",
                "Class Metrics": "class_metrics",
                "Grad-CAM": "grad_cam",
                "Comparison": "comparison"
            }
            filter_key = type_map.get(fig_type_filter, "")
            filtered_figures = [f for f in filtered_figures if filter_key in f.name.lower()]
        
        # Display in grid
        if filtered_figures:
            st.markdown(f"Showing **{len(filtered_figures)}** figures")
            cols = st.columns(2)
            for idx, fig_path in enumerate(sorted(filtered_figures, key=lambda x: x.name)):
                with cols[idx % 2]:
                    # Clean up caption
                    caption = fig_path.stem.replace("_", " ").title()
                    st.image(str(fig_path), caption=caption, use_container_width=True)
        else:
            st.info(f"No figures match the selected filters.")
    else:
        st.info("No figures found. Run the training pipeline to generate visualizations.")

# -----------------------------------------------------------------------------
# TAB 3: MODELS
# -----------------------------------------------------------------------------
with tab3:
    st.markdown("### All Model Files")
    
    if all_models:
        # Summary stats
        col1, col2, col3 = st.columns(3)
        baseline_count = len([m for m in all_models if m["type"] == "Baseline"])
        reg_count = len([m for m in all_models if m["type"] == "Regularized"])
        tl_count = len([m for m in all_models if m["type"] == "Transfer Learning"])
        
        with col1:
            st.metric("Baseline Models", baseline_count)
        with col2:
            st.metric("Regularized Models", reg_count)
        with col3:
            st.metric("Transfer Learning", tl_count)
        
        st.markdown("---")
        
        # Models table
        df = pd.DataFrame([
            {
                "Filename": m["filename"],
                "Type": m["type"],
                "Size (MB)": f"{m['size_mb']:.1f}"
            }
            for m in all_models
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Grouped view
        st.markdown("### By Type")
        for model_type in ["Baseline", "Regularized", "Transfer Learning"]:
            type_models = [m for m in all_models if m["type"] == model_type]
            if type_models:
                with st.expander(f"**{model_type}** ({len(type_models)} models)", expanded=False):
                    for m in type_models:
                        st.markdown(f"""
                        <div class="model-card">
                            <strong>{m['filename']}</strong>
                            <span style="color:#888; float:right;">{m['size_mb']:.1f} MB</span>
                        </div>
                        """, unsafe_allow_html=True)
    else:
        st.info("No models found. Run the training pipeline first.")

# Footer
st.markdown("---")
st.caption("Intel Image Classification • Model Comparison")