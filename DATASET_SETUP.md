# Dataset Setup

## Download Dataset

**Intel Image Classification** from Kaggle:
[kaggle.com/datasets/puneet6060/intel-image-classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)

## Required Structure

**Classes must be directly under train/test folders:**

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

## Fix Kaggle Dataset Structure

The Kaggle download has nested folders like `seg_train/seg_train/` and `seg_test/seg_test/`. You need to flatten them:

**Wrong (from Kaggle):**
```
archive/
├── seg_train/
│   └── seg_train/      ← extra nested folder
│       ├── buildings/
│       └── ...
└── seg_test/
    └── seg_test/       ← extra nested folder
        ├── buildings/
        └── ...
```

**Fix it:**
```bash
# Move the inner folders to data/
mv archive/seg_train/seg_train data/train
mv archive/seg_test/seg_test data/test
```

**Or if you already have nested structure in data/:**
```bash
# If you have data/train/seg_train/seg_train/
mv data/train/seg_train/seg_train/* data/train/
rm -r data/train/seg_train

# If you have data/test/seg_test/seg_test/
mv data/test/seg_test/seg_test/* data/test/
rm -r data/test/seg_test
```

**Verify structure:**
```bash
ls data/train/
# Should show: buildings  forest  glacier  mountain  sea  street

ls data/test/
# Should show: buildings  forest  glacier  mountain  sea  street
```

---

## Docker Mode

### Default: Use `./data` folder

Put dataset in project's `data/` folder, then:
```bash
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose up --build
```

### Custom: Use any folder

```bash
HOST_UID=$(id -u) HOST_GID=$(id -g) DATASET_PATH=/path/to/dataset docker compose up --build
```

### Streamlit Settings (Docker)

- Train: `/app/data/train`
- Test: `/app/data/test`

> Docker mounts your host folder to `/app/data` inside containers.

---

## Local Mode (No Docker)

```bash
uvicorn api.main:app --port 8000
API_URL=http://localhost:8000 streamlit run streamlit/app.py
```

### Streamlit Settings (Local)

Use any absolute path:
- Train: `/home/user/datasets/intel/train`
- Test: `/home/user/datasets/intel/test`

---

## Troubleshooting

**"Path does not exist"**
- Docker: Use `/app/data/...` not `/home/...`
- Local: Use full absolute path

**Images not showing**
- Check folder structure has class subdirectories
- Restart Docker after path changes

**Browse button doesn't work**
- Browser security limitation
- Type path manually
