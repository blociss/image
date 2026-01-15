"""
Settings Page
=============
Configure dataset paths and application settings.
"""

import streamlit as st
import sys
from pathlib import Path
import os

# Paths
STREAMLIT_DIR = Path(__file__).parent.parent.resolve()
PROJECT_ROOT = STREAMLIT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Settings - Intel Image Classifier",
    page_icon="⚙️",
    layout="wide"
)

# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if "custom_train_path" not in st.session_state:
    st.session_state.custom_train_path = str(PROJECT_ROOT / "data" / "train")
if "custom_test_path" not in st.session_state:
    st.session_state.custom_test_path = str(PROJECT_ROOT / "data" / "test")
if "custom_pred_path" not in st.session_state:
    st.session_state.custom_pred_path = str(PROJECT_ROOT / "data" / "predictions")
if "use_custom_paths" not in st.session_state:
    st.session_state.use_custom_paths = False

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
    .settings-card {
        background: #1e1e2f;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #333;
    }
    .path-valid {
        color: #4ade80;
        font-weight: 600;
    }
    .path-invalid {
        color: #f87171;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def validate_path(path_str):
    """Check if path exists and is a directory."""
    if not path_str:
        return False, "Path is empty"
    
    path = Path(path_str)
    if not path.exists():
        return False, "Path does not exist"
    if not path.is_dir():
        return False, "Path is not a directory"
    
    return True, "Valid"

def count_images_in_path(path_str):
    """Count total images in a directory."""
    try:
        path = Path(path_str)
        if not path.exists():
            return 0
        
        count = 0
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            count += len(list(path.rglob(ext)))
        return count
    except:
        return 0

def get_subdirectories(path_str):
    """Get list of subdirectories (class folders)."""
    try:
        path = Path(path_str)
        if not path.exists():
            return []
        return [d.name for d in path.iterdir() if d.is_dir()]
    except:
        return []

# -----------------------------------------------------------------------------
# MAIN CONTENT
# -----------------------------------------------------------------------------
st.markdown('<h1 class="page-title">⚙️ Settings</h1>', unsafe_allow_html=True)
st.markdown("Configure dataset paths to work with any image classification dataset.")

IN_DOCKER = Path("/.dockerenv").exists()
if IN_DOCKER:
    st.info(
        "Running in Docker: Streamlit can only access folders mounted into the container. "
        "Mount your dataset with the DATASET_PATH env var and use /app/data/train and /app/data/test here."
    )

st.markdown("---")

# Dataset Path Configuration
st.markdown("### 📁 Dataset Path Configuration")

st.info("""
**Expected Dataset Structure:**
```
your_dataset/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
└── test/
    ├── class1/
    ├── class2/
    └── ...
```
Each class folder should contain images in JPG or PNG format.
""")

# Toggle custom paths
use_custom = st.checkbox(
    "Use custom dataset paths",
    value=st.session_state.use_custom_paths,
    help="Enable this to specify custom paths for your dataset"
)
st.session_state.use_custom_paths = use_custom

if use_custom:
    st.markdown("#### Training Data Path")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        train_path = st.text_input(
            "Training data directory",
            value=st.session_state.custom_train_path,
            placeholder="/path/to/your/dataset/train",
            help="Full path to the training data directory",
            key="train_path_input"
        )
        if train_path:
            st.session_state.custom_train_path = train_path
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📂 Browse", key="browse_train"):
            st.info("Enter path manually or use file system browser")
    
    # Validate training path
    train_valid, train_msg = validate_path(train_path)
    if train_valid:
        train_images = count_images_in_path(train_path)
        train_classes = get_subdirectories(train_path)
        st.markdown(f'<p class="path-valid">✓ {train_msg} • {train_images} images • {len(train_classes)} classes</p>', unsafe_allow_html=True)
        if train_classes:
            st.caption(f"Classes found: {', '.join(train_classes)}")
    else:
        st.markdown(f'<p class="path-invalid">✗ {train_msg}</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("#### Test Data Path")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        test_path = st.text_input(
            "Test data directory",
            value=st.session_state.custom_test_path,
            placeholder="/path/to/your/dataset/test",
            help="Full path to the test data directory",
            key="test_path_input"
        )
        if test_path:
            st.session_state.custom_test_path = test_path
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📂 Browse", key="browse_test"):
            st.info("Enter path manually or use file system browser")
    
    # Validate test path
    test_valid, test_msg = validate_path(test_path)
    if test_valid:
        test_images = count_images_in_path(test_path)
        test_classes = get_subdirectories(test_path)
        st.markdown(f'<p class="path-valid">✓ {test_msg} • {test_images} images • {len(test_classes)} classes</p>', unsafe_allow_html=True)
        if test_classes:
            st.caption(f"Classes found: {', '.join(test_classes)}")
    else:
        st.markdown(f'<p class="path-invalid">✗ {test_msg}</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("#### Predictions Output Path")
    pred_path = st.text_input(
        "Predictions output directory",
        value=st.session_state.custom_pred_path,
        placeholder="/path/to/your/dataset/predictions",
        help="Directory where prediction outputs will be saved",
        key="pred_path_input"
    )
    if pred_path:
        st.session_state.custom_pred_path = pred_path
    
    pred_valid, pred_msg = validate_path(pred_path)
    if pred_valid:
        st.markdown(f'<p class="path-valid">✓ {pred_msg}</p>', unsafe_allow_html=True)
    else:
        st.markdown(f'<p class="path-invalid">✗ {pred_msg}</p>', unsafe_allow_html=True)
        if pred_path and st.button("Create directory", key="create_pred_dir"):
            try:
                Path(pred_path).mkdir(parents=True, exist_ok=True)
                st.success(f"✅ Created directory: {pred_path}")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to create directory: {str(e)}")
    
    st.markdown("---")
    
    # Save button
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("💾 Save Settings", type="primary", use_container_width=True):
            if train_valid and test_valid:
                st.success("✅ Settings saved! The Data Analysis page will now use these paths.")
                st.balloons()
            else:
                st.error("❌ Please fix invalid paths before saving.")
    
    with col2:
        if st.button("🔄 Reset to Default", use_container_width=True):
            st.session_state.custom_train_path = str(PROJECT_ROOT / "data" / "train")
            st.session_state.custom_test_path = str(PROJECT_ROOT / "data" / "test")
            st.session_state.custom_pred_path = str(PROJECT_ROOT / "data" / "predictions")
            st.session_state.use_custom_paths = False
            st.rerun()

else:
    st.markdown("### Default Paths")
    default_train = PROJECT_ROOT / "data" / "train"
    default_test = PROJECT_ROOT / "data" / "test"
    
    st.code(f"Training: {default_train}", language="text")
    st.code(f"Test: {default_test}", language="text")
    
    train_images = count_images_in_path(str(default_train))
    test_images = count_images_in_path(str(default_test))
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training Images", train_images)
    with col2:
        st.metric("Test Images", test_images)

# Footer
st.markdown("---")
st.markdown("### 💡 Tips")
st.markdown("""
- **For Docker:** Mount your dataset directory as a volume in `docker-compose.yml`
- **For Local:** Place your dataset in the `data/` folder or use custom paths
- **Dataset Format:** Ensure images are organized in class subdirectories
- **Supported Formats:** JPG, JPEG, PNG
""")

st.caption("Settings are stored in session state and will reset when you refresh the page.")
