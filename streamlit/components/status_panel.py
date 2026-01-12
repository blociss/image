# components/status_panel.py
import streamlit as st
import os
import sys
import requests
from typing import Dict, Any
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.config import MODELS_DIR, LEGACY_TRAIN_DIR, LEGACY_TEST_DIR
except ImportError:
    # Fallback paths if src.config not available
    MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
    LEGACY_TRAIN_DIR = PROJECT_ROOT / "data" / "train" / "seg_train" / "seg_train"
    LEGACY_TEST_DIR = PROJECT_ROOT / "data" / "test" / "seg_test" / "seg_test"

def get_system_status() -> Dict[str, Any]:
    status = {
        "api": _check_api(),
        "tensorflow": _check_tensorflow(),
        "models": _check_models(),
        "datasets": _check_datasets()
    }
    return status

def _check_api() -> Dict[str, Any]:
    try:
        api_url = os.getenv("API_URL", "http://localhost:8000")
        response = requests.get(f"{api_url}/", timeout=5)
        return {
            "status": "✅" if response.status_code == 200 else "❌",
            "message": f"API is {'running' if response.status_code == 200 else 'not responding'}"
        }
    except Exception as e:
        return {"status": "❌", "message": f"API error: {str(e)}"}

def _check_tensorflow() -> Dict[str, str]:
    try:
        import tensorflow as tf
        return {"status": "✅", "message": f"TensorFlow {tf.__version__} available"}
    except ImportError:
        return {"status": "⚠️", "message": "TensorFlow not installed"}

def _check_models() -> Dict[str, Any]:
    models = ['baseline_model.keras', 'regularized_model.keras', 'tl_model.keras']
    results = {}
    for model in models:
        path = MODELS_DIR / model
        exists = path.exists()
        results[model] = {"status": "✅" if exists else "❌", "exists": exists}
    return results

def _check_datasets() -> Dict[str, Any]:
    datasets = {
        "train": LEGACY_TRAIN_DIR,
        "test": LEGACY_TEST_DIR
    }
    results = {}
    for name, path in datasets.items():
        exists = path.exists() if hasattr(path, 'exists') else os.path.isdir(path)
        results[name] = {"status": "✅" if exists else "❌", "path": str(path), "exists": exists}
    return results

def render_status_panel():
    with st.sidebar.expander("🔍 System Status", expanded=True):
        status = get_system_status()
        st.markdown("### API Status")
        st.markdown(f"{status['api']['status']} {status['api']['message']}")
        st.markdown("### TensorFlow")
        st.markdown(f"{status['tensorflow']['status']} {status['tensorflow']['message']}")
        st.markdown("### Models")
        for model, info in status['models'].items():
            st.markdown(f"{info['status']} {model} {'(found)' if info['exists'] else '(missing)'}")
        st.markdown("### Datasets")
        for name, info in status['datasets'].items():
            st.markdown(f"{info['status']} {name} set: {info['path']}")