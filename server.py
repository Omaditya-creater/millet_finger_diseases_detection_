import io
import base64
import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from torchvision import transforms

from ml_pipeline.model import build_model
from ml_pipeline.gradcam import GradCAMGenerator, overlay_heatmap_on_image
from ml_pipeline.genai_advisor import GenAIAgronomicAdvisor
from ml_pipeline.dataset import CLASSES, IDX_TO_CLASS

app = FastAPI(
    title="GenAI Finger Millet Pathology API",
    description="Hybrid EfficientNet-ViT Deep Learning & Grad-CAM Explainable AI Backend with GenAI Agronomic Advisory Copilot",
    version="1.0.0"
)

# Enable CORS for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ml_pipeline", "hybrid_model.pth")

model = None
gradcam_gen = None
advisor = GenAIAgronomicAdvisor()

def init_model():
    global model, gradcam_gen
    try:
        model = build_model(num_classes=len(CLASSES))
        if os.path.exists(MODEL_PATH):
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(state_dict)
            print(f"[AI Engine] Successfully loaded trained weights from: {MODEL_PATH}")
        else:
            print(f"[AI Engine] Notice: {MODEL_PATH} not found. Running initialized architecture.")
        model.to(DEVICE)
        model.eval()
        gradcam_gen = GradCAMGenerator(model)
        print(f"[AI Engine] Hybrid EfficientNet-ViT & Grad-CAM running on {DEVICE}!")
    except Exception as e:
        print(f"[AI Engine] Error initializing model: {e}")

init_model()

# Image Preprocessing Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

DISEASE_KEY_MAPPING = {
    0: "healthy",
    1: "blast",
    2: "cercospora",
    3: "blight",
    4: "smut"
}

@app.get("/api/health")
def health_check():
    """Health check endpoint for Render monitoring and client verification."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(DEVICE),
        "classes": CLASSES
    }

@app.post("/api/predict")
async def predict_leaf(
    file: UploadFile = File(...),
    growth_stage: str = Form("Tillering/Flowering"),
    soil_type: str = Form("Red Loam"),
    weather: str = Form("28°C, 75% RH")
):
    """
    Accepts an uploaded finger millet leaf image and runs:
    1. Hybrid EfficientNet-ViT classification
    2. Grad-CAM visual explainability heatmap computation
    3. Lesion Severity Index (LSI %) quantification
    4. GenAI context-aware agronomic advisory generation
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        original_np = np.array(pil_image)

        # Prepare input tensor (1, 3, 224, 224)
        input_tensor = transform(pil_image).unsqueeze(0).to(DEVICE)

        # Grad-CAM forward + backward pass
        with torch.set_grad_enabled(True):
            cam_result = gradcam_gen.generate_heatmap(input_tensor)

        pred_idx = cam_result["target_class"]
        pred_class_name = IDX_TO_CLASS.get(pred_idx, "Healthy Leaf")
        confidence = float(cam_result["confidence"])
        lsi_pct = float(cam_result["lsi_percentage"])

        # Compute full class probability distribution
        with torch.no_grad():
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            prob_dist = {CLASSES[i]: round(float(probs[i]), 4) for i in range(len(CLASSES))}

        # Blend Grad-CAM heatmap over original leaf image
        orig_resized = cv2.resize(original_np, (224, 224))
        blended, colormap = overlay_heatmap_on_image(orig_resized, cam_result["heatmap_resized"], alpha=0.55)

        # Encode blended image to base64 Data URL
        _, buffer = cv2.imencode(".png", cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
        heatmap_base64 = "data:image/png;base64," + base64.b64encode(buffer).decode("utf-8")

        # Synthesize GenAI Agronomic Advisory
        advisory_data = advisor.generate_advisory(
            disease_class=pred_class_name,
            confidence=confidence,
            lsi_percentage=lsi_pct,
            growth_stage=growth_stage,
            soil_type=soil_type,
            weather=weather
        )

        return JSONResponse(content={
            "success": True,
            "disease_key": DISEASE_KEY_MAPPING.get(pred_idx, "blast"),
            "disease_name": pred_class_name,
            "confidence": round(confidence, 4),
            "confidence_percentage": round(confidence * 100, 2),
            "lsi_percentage": round(lsi_pct, 2),
            "class_probabilities": prob_dist,
            "heatmap_image_base64": heatmap_base64,
            "advisory": advisory_data
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

class AdvisoryRequest(BaseModel):
    disease_name: str = "Leaf Blast (Pyricularia oryzae)"
    confidence: float = 0.98
    lsi: float = 15.0
    growth_stage: str = "Tillering/Flowering"
    soil_type: str = "Red Loam"
    weather: str = "28°C, 75% RH"

@app.post("/api/advisory")
def get_custom_advisory(req: AdvisoryRequest = None):
    """Generates structured advisory and markdown report for given parameters."""
    if req is None:
        req = AdvisoryRequest()
    advisory_data = advisor.generate_advisory(
        disease_class=req.disease_name,
        confidence=req.confidence,
        lsi_percentage=req.lsi,
        growth_stage=req.growth_stage,
        soil_type=req.soil_type,
        weather=req.weather
    )
    md_report = advisor.format_markdown_report(advisory_data)
    return {
        "advisory": advisory_data,
        "markdown_report": md_report
    }

# Mount static web app files to serve UI at root URL
static_dir = os.path.join(BASE_DIR, "web_app")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="web_app")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
