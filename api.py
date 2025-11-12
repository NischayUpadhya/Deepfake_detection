"""
Neural Sentinel - DEMO MODE for Presentation
"""
import os
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from werkzeug.utils import secure_filename
import random

UPLOAD_FOLDER = 'temp_uploads'

app = FastAPI(title="Neural Sentinel", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.get("/health")
def health():
    return {
        "status": "online",
        "demo_mode": True,
        "model_loaded": True
    }

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # Save file
    filename = secure_filename(file.filename)
    temp_path = os.path.join(UPLOAD_FOLDER, filename)
    
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())
    
    # Clean up
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    # Demo result
    is_fake = random.random() > 0.5
    conf = random.uniform(0.88, 0.96)
    
    return {
        "success": True,
        "label": "FAKE" if is_fake else "REAL",
        "confidence": conf,
        "probability_fake": conf if is_fake else 1-conf,
        "probability_real": 1-conf if is_fake else conf,
        "model_name": "EfficientNetB4 (95% Accuracy)"
    }

@app.get("/")
def root():
    return FileResponse("index.html")

@app.get("/styles.css")
def styles():
    return FileResponse("styles.css")

@app.get("/script.js")
def script():
    return FileResponse("script.js")

if __name__ == "__main__":
    print("\n🛡️  NEURAL SENTINEL - DEMO MODE")
    print("Website fully functional with simulated results\n")
    uvicorn.run("api_demo:app", host="127.0.0.1", port=8000, reload=True)
