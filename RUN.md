# How to Run

## Docker (Recommended)

```bash
# Start (with permission fix)
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose up --build

# Access
# Streamlit: http://localhost:8501
# API docs:  http://localhost:8000/docs

# Stop
docker compose down
```

### Why HOST_UID/HOST_GID?

Ensures containers can write to `outputs/feedback.csv`. Without it, you get permission errors.

### Use .env File (optional)

```bash
echo "HOST_UID=$(id -u)" >> .env
echo "HOST_GID=$(id -g)" >> .env
```
 
Then just: `docker compose up --build`

### Custom Dataset Path

```bash
HOST_UID=$(id -u) HOST_GID=$(id -g) DATASET_PATH=/path/to/dataset docker compose up --build
```

In Streamlit Settings: `/app/data/train` and `/app/data/test`

---

## Local (No Docker)

```bash
# Install
pip install -r requirements.txt

# Terminal 1: API
uvicorn api.main:app --port 8000

# Terminal 2: Streamlit
API_URL=http://localhost:8000 streamlit run streamlit/app.py
```

In local mode, use any path in Settings (e.g., `/home/user/datasets/train`).

---

## Train Models

```bash
python scripts/train_pipeline.py
```

Options:
- `--speed-mode` - Quick test
- `--baseline-only` / `--regularized-only` / `--tl-only` - Train single model

Models saved to `outputs/models/`.

---

## Troubleshooting

**Permission denied on feedback.csv:**
```bash
sudo chown -R $(id -u):$(id -g) outputs
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose up --build
```

**Port in use:**
```bash
lsof -ti:8000 | xargs kill -9
lsof -ti:8501 | xargs kill -9
```

**Rebuild:**
```bash
docker compose down
docker compose build --no-cache
docker compose up
```

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for more.



