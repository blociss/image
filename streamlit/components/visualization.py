"""
Visualization Components
=========================
Reusable visualization functions for the Streamlit dashboard.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, List
import numpy as np


def create_bar_chart(data: Dict[str, float], title: str = "", height: int = 300):
    """Create a styled bar chart using Plotly."""
    df = pd.DataFrame([
        {"Label": k, "Value": v} for k, v in data.items()
    ])
    
    fig = px.bar(
        df, x="Label", y="Value",
        color="Value",
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
        margin=dict(l=0, r=0, t=40 if title else 10, b=0),
        height=height
    )
    return fig


def create_horizontal_bar_chart(data: Dict[str, float], title: str = "", height: int = 250):
    """Create a horizontal bar chart for probabilities."""
    df = pd.DataFrame([
        {"Label": k, "Value": v} for k, v in data.items()
    ]).sort_values("Value", ascending=True)
    
    fig = px.bar(
        df, x="Value", y="Label", orientation='h',
        color="Value",
        color_continuous_scale=["#1e1e2f", "#667eea", "#764ba2"],
        range_color=[0, 100]
    )
    fig.update_layout(
        title=title,
        showlegend=False,
        coloraxis_showscale=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ccc'),
        xaxis=dict(range=[0, 100], gridcolor='#333'),
        yaxis=dict(gridcolor='#333'),
        margin=dict(l=0, r=0, t=40 if title else 10, b=0),
        height=height
    )
    return fig


def create_pie_chart(data: Dict[str, float], title: str = "", height: int = 300):
    """Create a styled pie chart."""
    fig = px.pie(
        names=list(data.keys()),
        values=list(data.values()),
        color_discrete_sequence=px.colors.sequential.Purples_r
    )
    fig.update_layout(
        title=title,
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ccc'),
        margin=dict(l=0, r=0, t=40 if title else 10, b=0),
        height=height
    )
    return fig


def create_confusion_matrix(cm: np.ndarray, labels: List[str], title: str = ""):
    """Create a confusion matrix heatmap."""
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="True", color="Count"),
        x=labels,
        y=labels,
        color_continuous_scale="Purples",
        text_auto=True
    )
    fig.update_layout(
        title=title,
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ccc'),
        margin=dict(l=0, r=0, t=40 if title else 10, b=0),
        height=400
    )
    return fig


def display_metric_card(value, label: str, color: str = "#667eea"):
    """Display a styled metric card."""
    st.markdown(f"""
    <div style="background:#1e1e2f; border-radius:12px; padding:1.5rem; text-align:center; border:1px solid #333;">
        <div style="font-size:2.5rem; font-weight:700; color:{color};">{value}</div>
        <div style="color:#888; font-size:0.9rem;">{label}</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------
# DATASET SUMMARY (PyArrow-free)
# ---------------------------------------------------------

def display_dataset_summary(train_counts, test_counts):
    df = pd.DataFrame({
        "Class": list(train_counts.keys()),
        "Train Images": list(train_counts.values()),
        "Test Images": list(test_counts.values())
    })

    styled = (
        df.style
          .background_gradient(subset=["Train Images", "Test Images"], cmap="YlGn")
          .format("{:.0f}", subset=["Train Images", "Test Images"])
    )

    st.subheader("Dataset Summary")
    st.markdown(styled.to_html(), unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = create_pie_chart(train_counts, "Train Set Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = create_pie_chart(test_counts, "Test Set Distribution")
        st.plotly_chart(fig, use_container_width=True)