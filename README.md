# Image Classification Project

An end-to-end image classification system for the Intel Image Classification dataset, featuring CNN models, FastAPI backend, and interactive Streamlit dashboard.

## 🎯 Overview

This project classifies landscape images into 6 categories:
- **Buildings** - Urban structures
- **Forest** - Wooded areas
- **Glacier** - Ice formations
- **Mountain** - Mountain landscapes
- **Sea** - Ocean/water scenes
- **Street** - Urban streets

## 📦 Dataset

This project uses the **Intel Image Classification Dataset** from Kaggle.

**Dataset Details:**
- ~25,000 images (150x150 RGB)
- 6 balanced classes
- Split: ~14,000 train, ~3,000 validation, ~7,000 test

**Download:**
1. Visit [Intel Image Classification on Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
2. Download and extract to the `data/` folder
3. Organize as:
   ```
   data/
   ├── train/
   │   ├── buildings/
   │   ├── forest/
   │   ├── glacier/
   │   ├── mountain/
   │   ├── sea/
   │   └── street/
   └── test/
       ├── buildings/
       ├── forest/
       ├── glacier/
       ├── mountain/
       ├── sea/
       └── street/
   ```

**Note:** The dataset is not included in this repository due to size. You must download it separately.

## 🚀 Quick Start with Docker

### Prerequisites

- Docker and Docker Compose installed
- No other services running on ports 8000 and 8501

### Run the Application

**1. Build and start the services:**
```bash
docker compose up --build
```

**2. Access the applications:**
- **Streamlit Dashboard**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs

**3. Stop the services:**
```bash
docker compose down
```

### Docker Commands

**Run in background (detached mode):**
```bash
docker compose up -d
```

**View logs:**
```bash
docker compose logs -f
```

**Rebuild from scratch:**
```bash
docker compose down
docker compose build --no-cache
docker compose up
```

**Check service status:**
```bash
docker compose ps
```

## 🤖 Available Models

Three CNN models with different architectures:

| Model | Accuracy | Description |
|-------|----------|-------------|
| **Baseline** | 73% | Simple CNN baseline |
| **Regularized** | 79% | Enhanced with regularization |
| **Transfer Learning** | 88% | MobileNetV2 (best performance) |

## 🔌 Using the API

The FastAPI backend provides automatic interactive documentation at http://localhost:8000/docs

**Available endpoints:**
- `GET /` - Health check
- `GET /models` - List available models
- `POST /predict/{model}` - Classify an image
- `POST /feedback` - Submit feedback

**Models:** `baseline`, `regularized`, `transfer_learning`

## 📊 Streamlit Dashboard Features

Access the interactive dashboard at http://localhost:8501

**Pages:**
- **Home** - Upload images and get predictions with confidence scores
- **Model Comparison** - Compare all three models side-by-side
- **Data Analysis** - View dataset statistics and samples
- **Feedback Dashboard** - Track user feedback and accuracy

**Features:**
- Drag & drop image upload
- Real-time predictions
- Grad-CAM visualizations (see what the model focuses on)
- Confusion matrices
- Model performance metrics

## 🛠️ Technologies

- **ML/DL:** TensorFlow, Keras
- **API:** FastAPI
- **Dashboard:** Streamlit
- **Deployment:** Docker Compose

## 📝 Training Your Own Models

See [RUN.md](RUN.md) for instructions on training models locally.

## 📄 License

MIT License
