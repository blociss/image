# Train only regularized model with speed mode (3 epochs)
python scripts/train_pipeline.py --regularized-only --speed-mode

# Train only transfer learning with speed mode (2+2 epochs)
python scripts/train_pipeline.py --tl-only --speed-mode

# Train both regularized and TL (not baseline)
python scripts/train_pipeline.py --regularized-only --speed-mode
python scripts/train_pipeline.py --tl-only --speed-mode

# Train all models with speed mode
python scripts/train_pipeline.py --speed-mode
