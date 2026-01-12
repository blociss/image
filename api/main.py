"""
Intel Image Classification API
===============================
FastAPI REST API for image classification using trained CNN models.

Endpoints:
- GET  /           : Health check and status
- GET  /models     : List available models
- GET  /models/{filename}/info : Get model details
- POST /predict/{filename}     : Classify an image

Author: Intel Image Classification Project
Version: 2.0.0
"""

import sys
from pathlib import Path
from io import BytesIO
from contextlib import asynccontextmanager
import logging
import time
from typing import Dict, List, Any

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import numpy as np

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODELS_DIR, IMG_SIZE, TL_IMG_SIZE, CLASS_NAMES

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# RESPONSE MODELS
# -----------------------------------------------------------------------------
class PredictionResult(BaseModel):
    predicted_class: str
    confidence: float
    inference_time_ms: int

class PredictionResponse(BaseModel):
    prediction: Dict[str, Any]
    all_predictions: Dict[str, str]
    model_used: str

class ModelInfo(BaseModel):
    filename: str
    type: str
    img_size: tuple
    loaded: bool

class HealthResponse(BaseModel):
    status: str
    version: str
    models_available: int
    models_loaded: int

# -----------------------------------------------------------------------------
# MODEL MANAGER
# -----------------------------------------------------------------------------
class ModelManager:
    """Manages model loading and caching."""
    
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.loaded_models: Dict[str, Dict] = {}
        self._tf = None
    
    @property
    def tf(self):
        """Lazy load TensorFlow."""
        if self._tf is None:
            import tensorflow as tf
            self._tf = tf
        return self._tf
    
    def get_available(self) -> List[Dict]:
        """Get list of available model files."""
        models = []
        if self.models_dir.exists():
            for f in sorted(self.models_dir.glob("*.keras")):
                is_tl = 'tl_' in f.name or 'transfer' in f.name.lower()
                models.append({
                    "filename": f.name,
                    "type": "transfer_learning" if is_tl else "cnn",
                    "img_size": TL_IMG_SIZE if is_tl else IMG_SIZE,
                    "loaded": f.name in self.loaded_models
                })
        return models
    
    def get_model(self, filename: str) -> Dict:
        """Load and return a model (with caching)."""
        if filename in self.loaded_models:
            return self.loaded_models[filename]
        
        model_path = self.models_dir / filename
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {filename}")
        
        is_tl = 'tl_' in filename or 'transfer' in filename.lower()
        img_size = TL_IMG_SIZE if is_tl else IMG_SIZE
        
        logger.info(f"Loading model: {filename}")
        model = self.tf.keras.models.load_model(str(model_path))
        
        self.loaded_models[filename] = {
            'model': model,
            'img_size': img_size,
            'type': 'transfer_learning' if is_tl else 'cnn'
        }
        logger.info(f"Model loaded: {filename}")
        return self.loaded_models[filename]
    
    def unload_model(self, filename: str) -> bool:
        """Unload a model from memory."""
        if filename in self.loaded_models:
            del self.loaded_models[filename]
            return True
        return False

# Global model manager
model_manager = ModelManager(MODELS_DIR)

# -----------------------------------------------------------------------------
# LIFESPAN
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    models = model_manager.get_available()
    logger.info(f"API Started - {len(models)} models available")
    yield
    # Shutdown
    logger.info("API Shutting down")

# -----------------------------------------------------------------------------
# FASTAPI APP
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Intel Image Classification API",
    description="Classify images into 6 scene categories using trained CNN models.",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# ENDPOINTS
# -----------------------------------------------------------------------------
@app.get("/", tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns API status and model availability.
    """
    models = model_manager.get_available()
    return {
        "status": "online",
        "version": "2.0.0",
        "models_available": [m["filename"] for m in models],
        "models_loaded": list(model_manager.loaded_models.keys()),
        "classes": CLASS_NAMES
    }


@app.get("/models", tags=["Models"])
async def list_models():
    """
    List all available models.
    
    Returns model filenames, types, and input sizes.
    """
    return {"models": model_manager.get_available()}


@app.get("/models/{filename}/info", tags=["Models"])
async def model_info(filename: str):
    """
    Get detailed information about a specific model.
    """
    models = model_manager.get_available()
    for m in models:
        if m["filename"] == filename:
            return m
    raise HTTPException(status_code=404, detail=f"Model not found: {filename}")


@app.post("/predict/{model_filename}", tags=["Prediction"])
async def predict(model_filename: str, file: UploadFile = File(...)):
    """
    Classify an image using the specified model.
    
    - **model_filename**: Name of the model file (e.g., baseline_20240101_120000.keras)
    - **file**: Image file (JPG, PNG)
    
    Returns predicted class, confidence, and all class probabilities.
    """
    start_time = time.time()
    
    # Validate model exists
    available = [m['filename'] for m in model_manager.get_available()]
    if model_filename not in available:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_filename}' not found. Available: {available}"
        )
    
    try:
        # Load model
        model_data = model_manager.get_model(model_filename)
        model = model_data['model']
        img_size = model_data['img_size']
        is_transfer = model_data['type'] == 'transfer_learning'
        
        # Read image
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert('RGB')
        image = image.resize(img_size)
        
        # Preprocess
        if is_transfer:
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
            img_array = preprocess_input(np.array(image).astype(np.float32))
        else:
            img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = model.predict(img_array, verbose=0)
        pred_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][pred_idx])
        
        inference_ms = int((time.time() - start_time) * 1000)
        
        # Build response
        all_preds = {
            CLASS_NAMES[i]: f"{predictions[0][i] * 100:.2f}%"
            for i in range(len(CLASS_NAMES))
        }
        
        return {
            "prediction": {
                "class": CLASS_NAMES[pred_idx],
                "confidence": f"{confidence * 100:.2f}%",
                "inference_time_ms": inference_ms
            },
            "all_predictions": all_preds,
            "model_used": model_filename
        }
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.delete("/models/{filename}/unload", tags=["Models"])
async def unload_model(filename: str):
    """
    Unload a model from memory to free resources.
    """
    if model_manager.unload_model(filename):
        return {"message": f"Model {filename} unloaded"}
    raise HTTPException(status_code=404, detail=f"Model not loaded: {filename}")


# -----------------------------------------------------------------------------
# FEEDBACK ENDPOINTS
# -----------------------------------------------------------------------------
import csv
import uuid
from datetime import datetime

FEEDBACK_FILE = PROJECT_ROOT / "outputs" / "feedback.csv"
FEEDBACK_FIELDS = ["id", "timestamp", "model", "predicted", "true_class", "confidence"]


def ensure_feedback_file():
    """Ensure feedback CSV exists with headers."""
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not FEEDBACK_FILE.exists():
        with open(FEEDBACK_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(FEEDBACK_FIELDS)


class FeedbackInput(BaseModel):
    predicted: str
    true_class: str
    model: str
    confidence: float


@app.post("/feedback", tags=["Feedback"])
async def submit_feedback(feedback: FeedbackInput):
    """
    Submit feedback on a prediction.
    
    - **predicted**: The class predicted by the model
    - **true_class**: The actual correct class (from user)
    - **model**: Model filename used
    - **confidence**: Prediction confidence
    """
    ensure_feedback_file()
    
    feedback_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().isoformat()
    
    try:
        with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                feedback_id,
                timestamp,
                feedback.model,
                feedback.predicted,
                feedback.true_class,
                feedback.confidence
            ])
        
        return {
            "status": "success",
            "message": "Feedback recorded",
            "id": feedback_id
        }
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(e)}")


@app.get("/feedback", tags=["Feedback"])
async def get_feedback():
    """
    Get all feedback entries.
    """
    ensure_feedback_file()
    
    try:
        feedback_list = []
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                feedback_list.append(row)
        return feedback_list
    except Exception as e:
        logger.error(f"Error reading feedback: {e}")
        return []


@app.delete("/feedback", tags=["Feedback"])
async def clear_feedback():
    """
    Clear all feedback (creates backup first).
    """
    if FEEDBACK_FILE.exists():
        backup = FEEDBACK_FILE.parent / f"feedback_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        FEEDBACK_FILE.rename(backup)
    
    ensure_feedback_file()
    return {"status": "success", "message": "Feedback cleared"}


# -----------------------------------------------------------------------------
# ERROR HANDLERS
# -----------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
