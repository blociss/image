"""
Feedback Dashboard
===================
View and analyze user feedback on predictions.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import os
import sys
from pathlib import Path

# Paths
STREAMLIT_DIR = Path(__file__).parent.parent.resolve()
PROJECT_ROOT = STREAMLIT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

API_URL = os.getenv("API_URL", "http://localhost:8000")

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Feedback - Intel Image Classifier",
    page_icon="💬",
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
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
@st.cache_data(ttl=30)
def load_feedback():
    """Load feedback from API."""
    try:
        resp = requests.get(f"{API_URL}/feedback", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            if "confidence" in df.columns:
                df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
            
            if "predicted" in df.columns and "true_class" in df.columns:
                df["correct"] = df["predicted"] == df["true_class"]
            
            return df
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 💬 Feedback")
    st.markdown("---")
    
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("View user feedback on model predictions.")

# -----------------------------------------------------------------------------
# MAIN CONTENT
# -----------------------------------------------------------------------------
st.markdown('<h1 class="page-title">💬 Feedback Dashboard</h1>', unsafe_allow_html=True)
st.markdown("Analyze user feedback and model performance.")

feedback_df = load_feedback()

if feedback_df.empty:
    st.info("📭 No feedback yet. Submit feedback from the Home page after making predictions.")
    
    col1, col2, col3, col4 = st.columns(4)
    for col, val, label in [(col1, "0", "Total"), (col2, "-", "Accuracy"), 
                             (col3, "0", "Models"), (col4, "-", "Latest")]:
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{val}</div>
                <div class="stat-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)
    st.stop()

# Calculate metrics
total = len(feedback_df)
correct = feedback_df["correct"].sum() if "correct" in feedback_df.columns else 0
accuracy = (correct / total * 100) if total > 0 else 0
models_used = feedback_df["model"].nunique() if "model" in feedback_df.columns else 0
latest = feedback_df["timestamp"].max() if "timestamp" in feedback_df.columns else None

# Stats row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="stat-card"><div class="stat-value">{total}</div><div class="stat-label">Total Feedback</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="stat-card"><div class="stat-value">{accuracy:.1f}%</div><div class="stat-label">Accuracy</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="stat-card"><div class="stat-value">{models_used}</div><div class="stat-label">Models Used</div></div>', unsafe_allow_html=True)
with col4:
    latest_str = latest.strftime("%m/%d %H:%M") if pd.notna(latest) else "-"
    st.markdown(f'<div class="stat-card"><div class="stat-value" style="font-size:1.5rem;">{latest_str}</div><div class="stat-label">Latest</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Analytics", "📋 Data", "🔧 Manage"])

with tab1:
    if "model" in feedback_df.columns and feedback_df["model"].notna().any():
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Feedback by Model")
            model_counts = feedback_df["model"].value_counts().reset_index()
            model_counts.columns = ["Model", "Count"]
            fig = px.bar(model_counts, x="Model", y="Count", color="Count",
                        color_continuous_scale=["#1e1e2f", "#667eea", "#764ba2"])
            fig.update_layout(showlegend=False, coloraxis_showscale=False,
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#ccc'), margin=dict(l=0, r=0, t=10, b=0), height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Accuracy by Model")
            if "correct" in feedback_df.columns:
                model_acc = feedback_df.groupby("model")["correct"].mean().reset_index()
                model_acc.columns = ["Model", "Accuracy"]
                model_acc["Accuracy"] = model_acc["Accuracy"] * 100
                
                fig = px.bar(model_acc, x="Model", y="Accuracy", 
                            text=model_acc["Accuracy"].apply(lambda x: f"{x:.0f}%"))
                fig.update_traces(
                    marker_color=model_acc["Accuracy"].apply(
                        lambda x: "#4ade80" if x >= 70 else "#fbbf24" if x >= 40 else "#f87171"
                    ),
                    textposition="outside"
                )
                fig.update_layout(showlegend=False,
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='#ccc'), 
                                yaxis=dict(range=[0, 105], title="Accuracy (%)"),
                                xaxis=dict(title="Model"),
                                margin=dict(l=0, r=0, t=10, b=0), height=300)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No accuracy data available")

with tab2:
    st.markdown("### Raw Feedback Data")
    display_cols = [c for c in ["timestamp", "model", "predicted", "true_class", "confidence", "correct"] 
                    if c in feedback_df.columns]
    
    if display_cols:
        display_df = feedback_df[display_cols].copy()
        if "timestamp" in display_df.columns:
            display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        if "confidence" in display_df.columns:
            display_df["confidence"] = display_df["confidence"].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "-")
        if "correct" in display_df.columns:
            display_df["correct"] = display_df["correct"].map({True: "✅", False: "❌"})
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        csv = feedback_df.to_csv(index=False)
        st.download_button("📥 Download CSV", csv, "feedback_export.csv", "text/csv", use_container_width=True)

with tab3:
    st.markdown("### Manage Feedback Data")
    st.warning("⚠️ This action cannot be undone.")
    
    if st.button("🗑️ Clear All Feedback", type="secondary", use_container_width=True):
        try:
            resp = requests.delete(f"{API_URL}/feedback", timeout=5)
            if resp.status_code == 200:
                st.success("Feedback cleared!")
                st.cache_data.clear()
                st.rerun()
            else:
                st.error(f"Failed: {resp.text}")
        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("---")
st.caption("Intel Image Classification • Feedback Dashboard")
