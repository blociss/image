# How to Run

## Choose Your Mode

### 1️⃣ GitHub Container Registry (Easiest)
Pre-built images, no Docker build required.

### 2️⃣ Local Development (No Docker)
Run directly on your machine.

### 3️⃣ Docker Build (Developers)
Build images locally from source.

---

## 1️⃣ GitHub Container Registry

```bash
# Set your username
export GITHUB_USERNAME=yourusername

# Quick start
./deploy.sh run

# Access
# Streamlit: http://localhost:8501
# API docs:  http://localhost:8000/docs

# Stop
docker compose -f docker-compose.ghcr.yml down
```

**Custom dataset path:**
```bash
DATASET_PATH=/path/to/dataset ./deploy.sh run
```

**Streamlit settings:**
- Train path: `/app/data/train`
- Test path: `/app/data/test`

---

## 2️⃣ Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Terminal 1: API
uvicorn api.main:app --port 8000

# Terminal 2: Streamlit
API_URL=http://localhost:8000 streamlit run streamlit/app.py
```

**Streamlit settings:**
- Train path: `/home/user/datasets/train`
- Test path: `/home/user/datasets/test`

---

## 3️⃣ Docker Build (Developers)

```bash
# Start with permissions
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose up --build

# Stop
docker compose down
```

**Why HOST_UID/HOST_GID?**
Fixes permission errors for `outputs/feedback.csv`.

**Custom dataset path:**
```bash
HOST_UID=$(id -u) HOST_GID=$(id -g) DATASET_PATH=/path/to/dataset docker compose up --build
```

---

## Model Training

```bash
# Quick test (all modes)
docker compose exec api python scripts/train_pipeline.py --speed-mode
# or
python scripts/train_pipeline.py --speed-mode

# Full training
python scripts/train_pipeline.py

# Single model
python scripts/train_pipeline.py --baseline-only
python scripts/train_pipeline.py --regularized-only
python scripts/train_pipeline.py --tl-only
```

**Training times:**
- Baseline: ~5 min
- Regularized: ~8 min
- Transfer Learning: ~15 min

---

## Common Commands

```bash
# Check containers
docker compose ps

# View logs
docker compose logs -f

# Enter container
docker compose exec api bash

# Restart
docker compose restart

# Kill ports
lsof -ti:8000 | xargs kill -9
lsof -ti:8501 | xargs kill -9
```

For detailed troubleshooting, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
