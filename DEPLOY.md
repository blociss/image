# Deployment Guide

## Prerequisites

- Docker and Docker Compose installed
- GitHub token with packages permissions (for building)

---

## For Users: Deploy from GitHub Container Registry

### Quick Start
```bash
./deploy.sh run
```

### Step-by-Step
```bash
# 1. Clone/download project
cd image

# 2. Make script executable
chmod +x deploy.sh

# 3. Deploy (auto-pulls images)
./deploy.sh run

# 4. Train models (optional)
./deploy.sh train

# 5. Access
# Streamlit: http://localhost:8501
# API docs:  http://localhost:8000/docs
```

### Custom Dataset Path
```bash
DATASET_PATH=/path/to/dataset ./deploy.sh run
```

---

## For Developers: Build & Push to GHCR

### Login to GitHub Container Registry
```bash
# Set your username
export GITHUB_USERNAME=yourusername

# Login with your token
echo "YOUR_GITHUB_TOKEN" | docker login ghcr.io -u $GITHUB_USERNAME --password-stdin
```

### Build and Push
```bash
./deploy.sh build
```

### Manual Build & Push
```bash
# Build images
docker compose build

# Tag for GHCR (replace with your username)
docker tag image-api:latest ghcr.io/$GITHUB_USERNAME/image-classification-api:latest
docker tag image-streamlit:latest ghcr.io/$GITHUB_USERNAME/image-classification-streamlit:latest

# Push to GHCR
docker push ghcr.io/$GITHUB_USERNAME/image-classification-api:latest
docker push ghcr.io/$GITHUB_USERNAME/image-classification-streamlit:latest
```

---

## Deploy Script Commands

```bash
./deploy.sh build    # Build and push to GHCR (developers)
./deploy.sh run      # Pull and deploy from GHCR (users)
./deploy.sh train    # Train models after deployment
./deploy.sh stop     # Stop all containers
./deploy.sh status   # Check deployment status
```

---

## File Structure After Deployment

```
project/
├── docker-compose.ghcr.yml    # GHCR compose file
├── deploy.sh                  # Deployment script
├── data/                      # Dataset folder
│   ├── train/                 # Training images
│   └── test/                  # Test images
└── outputs/                   # Generated files
    ├── models/                # Trained models
    └── feedback.csv           # User feedback
```

---

## Environment Variables

Create `.env` file (optional):
```env
HOST_UID=1001
HOST_GID=1001
DATASET_PATH=/path/to/dataset
```

---

## Troubleshooting

### Images Not Pulling
```bash
export GITHUB_USERNAME=yourusername
docker login ghcr.io -u $GITHUB_USERNAME --password-stdin
```

### Containers Not Starting
```bash
docker compose -f docker-compose.ghcr.yml logs
docker compose -f docker-compose.ghcr.yml down
docker compose -f docker-compose.ghcr.yml up -d
```

### No Models Available
```bash
./deploy.sh train
```

For detailed troubleshooting, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
