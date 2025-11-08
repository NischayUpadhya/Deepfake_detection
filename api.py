"""
Neural Sentinel - FastAPI Backend
Updated for modern frontend integration
"""

import os
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from werkzeug.utils import secure_filename
import numpy as np
import cv2
from typing import Dict, Any

# Import your existing code
try:
    from inference import DeepfakeDetector
    from config import Config
except ImportError:
    print("Warning: Could not import inference or config modules")
    DeepfakeDetector = None
    Config = None

# --- CONFIGURATION ---
UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS_IMG = {'png', 'jpg', 'jpeg'}
ALLOWED_EXTENSIONS_VID = {'mp4', 'avi', 'mov'}

# Create the FastAPI app
app = FastAPI(
    title="Neural Sentinel API",
    description="AI-Powered Deepfake Detection System",
    version="1.0.0"
)

# Add CORS middleware - CRITICAL for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GLOBAL VARIABLES ---
config = None
detector = None
MODEL_PATH = None

# --- STARTUP: LOAD MODEL ---
@app.on_event("startup")
def load_model():
    """Load the ML model when the server starts."""
    global detector, config, MODEL_PATH
    
    if Config is None:
        print("⚠️  Config module not available")
        return
    
    config = Config()
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    MODEL_PATH = os.path.join(config.MODEL_SAVE_PATH, 'best_efficientnet.h5')
    
    if not os.path.exists(MODEL_PATH):
        print("=" * 80)
        print("⚠️  WARNING: Model file not found!")
        print(f"Expected location: {MODEL_PATH}")
        print("\nThe API will run in DEMO MODE.")
        print("To use real detection:")
        print("  1. Train a model: python main.py train --model efficientnet")
        print("  2. Or place a trained model at the path above")
        print("=" * 80)
    else:
        try:
            print(f"📦 Loading model from {MODEL_PATH}...")
            detector = DeepfakeDetector(MODEL_PATH, img_size=config.IMG_SIZE)
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Running in DEMO MODE")

# --- HEALTH CHECK ---
@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Check if the API is running and model is loaded."""
    return {
        "status": "online",
        "model_loaded": detector is not None,
        "demo_mode": detector is None
    }

# --- MAIN DETECTION ENDPOINT ---
@app.post("/detect")
async def detect_deepfake(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Main detection endpoint. Receives a file and returns prediction.
    
    Returns:
        {
            "success": bool,
            "label": "REAL" or "FAKE",
            "confidence": float (0-1),
            "probability_real": float (0-1),
            "probability_fake": float (0-1),
            "num_frames_analyzed": int (for videos only),
            "error": str (if failed)
        }
    """
    
    # Demo mode - return simulated results
    if detector is None:
        print("🎭 Running in DEMO MODE - returning simulated results")
        return get_demo_result(file.filename)
    
    # Validate file type
    file_type = get_file_type(file.filename)
    if not file_type:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Supported: JPG, PNG, MP4, MOV, AVI"
        )
    
    # Save file temporarily
    filename = secure_filename(file.filename)
    temp_path = os.path.join(UPLOAD_FOLDER, filename)
    
    result = {}
    try:
        # Write uploaded file
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())
        
        print(f"🔍 Analyzing {file_type}: {filename}")
        
        # Run detection
        if file_type == 'image':
            result = detector.predict_image(temp_path)
        else:
            result = detector.predict_video(
                temp_path,
                num_frames=config.FRAMES_PER_VIDEO
            )
        
        print(f"✅ Analysis complete: {result.get('label')} ({result.get('confidence', 0)*100:.1f}%)")
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    # Validate result
    if not result.get('success'):
        raise HTTPException(
            status_code=400,
            detail=result.get('error', 'Prediction failed')
        )
    
    return result

# --- HELPER FUNCTIONS ---
def get_file_type(filename: str) -> str:
    """Determine if file is image or video based on extension."""
    if not filename or '.' not in filename:
        return None
    
    ext = filename.rsplit('.', 1)[1].lower()
    
    if ext in ALLOWED_EXTENSIONS_IMG:
        return 'image'
    elif ext in ALLOWED_EXTENSIONS_VID:
        return 'video'
    
    return None

def get_demo_result(filename: str) -> Dict[str, Any]:
    """
    Return simulated results for demo mode.
    This allows testing the UI without a trained model.
    """
    import random
    
    file_type = get_file_type(filename)
    
    # Simulate detection
    is_fake = random.random() > 0.5
    confidence = random.uniform(0.7, 0.95)
    
    result = {
        "success": True,
        "label": "FAKE" if is_fake else "REAL",
        "confidence": confidence,
        "probability_fake": confidence if is_fake else 1 - confidence,
        "probability_real": 1 - confidence if is_fake else confidence,
        "demo_mode": True
    }
    
    if file_type == 'video':
        result["num_frames_analyzed"] = 10
    
    return result

# --- STATIC FILES (Serve Frontend) ---
# This MUST be last to avoid route conflicts
@app.get("/styles.css")
def get_styles():
    return FileResponse("styles.css")

@app.get("/script.js")
def get_script():
    return FileResponse("script.js")

@app.get("/")
def read_root():
    return FileResponse("index.html")

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🛡️  NEURAL SENTINEL")
    print("=" * 80)
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
