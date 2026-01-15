# Image Classification Project

CNN-based image classification with FastAPI backend and Streamlit dashboard.

## Screenshots

<img src="screenshots/streamlit_home.png" alt="Streamlit Dashboard" width="500">
<img src="screenshots/api_docs.png" alt="API Documentation" width="500">

## Quick Start

**Easiest - GitHub Container Registry (any OS):**
```bash
./deploy.sh run
```

**Local Development (no Docker):**
```bash
pip install -r requirements.txt
uvicorn api.main:app --port 8000 &
API_URL=http://localhost:8000 streamlit run streamlit/app.py
```

**Docker Build (developers):**
```bash
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose up --build
```

## Dataset

Intel Image Classification from Kaggle (~25,000 images, 6 classes)

Download: [kaggle.com/datasets/puneet6060/intel-image-classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)

## Models

| Model | Accuracy | Time |
|-------|----------|------|
| Baseline | 73% | ~5 min |
| Regularized | 79% | ~8 min |
| Transfer Learning | 88% | ~15 min |

## Documentation

- **[DEPLOY.md](DEPLOY.md)** - Complete deployment guide
- **[RUN.md](RUN.md)** - Detailed run instructions  
- **[DATASET_SETUP.md](DATASET_SETUP.md)** - Dataset configuration
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Common issues

## Technologies

TensorFlow, FastAPI, Streamlit, Docker
