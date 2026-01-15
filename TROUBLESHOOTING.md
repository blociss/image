# Troubleshooting

## Permission Denied on feedback.csv

**Most common issue in Docker.**

```bash
# Fix ownership
sudo chown -R $(id -u):$(id -g) outputs

# Restart with HOST_UID/HOST_GID
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose up --build
```

---

## Training Data Not Found

**Docker mode:** Use `/app/data/train` in Settings (not `/home/...`)

**Local mode:** Use full absolute path like `/home/user/datasets/train`

**Check structure (classes must be directly under train/test):**
```
data/train/buildings/*.jpg   ✓ Correct
data/train/forest/*.jpg      ✓ Correct

data/train/seg_train/buildings/*.jpg   ✗ Wrong (nested folder)
data/test/seg_test/seg_test/...        ✗ Wrong (double nested)
```

**Fix nested Kaggle structure:**
```bash
# If you have nested folders like seg_train/seg_train/
mv data/train/seg_train/seg_train/* data/train/
rm -r data/train/seg_train

mv data/test/seg_test/seg_test/* data/test/
rm -r data/test/seg_test
```

**Verify:**
```bash
ls data/train/
# Should show: buildings  forest  glacier  mountain  sea  street
```

---

## Port Already in Use

```bash
lsof -ti:8000 | xargs kill -9
lsof -ti:8501 | xargs kill -9
```

---

## Container Won't Start

```bash
# Check logs
docker compose logs

# Rebuild
docker compose down
docker compose build --no-cache
docker compose up
```

---

## No Models Found

Train models first:
```bash
python scripts/train_pipeline.py
```

Or check `outputs/models/` has `.keras` files.

---

## Feedback Not Saving

1. Check API logs: `docker compose logs api`
2. Test API directly:
```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"predicted":"mountain","true_class":"mountain","model":"test.keras","confidence":0.95}'
```

---

## Useful Commands

```bash
# Status
docker compose ps

# Logs
docker compose logs -f

# Enter container
docker compose exec api bash

# Check permissions
ls -la outputs/
docker compose exec api ls -la /app/outputs/
```
