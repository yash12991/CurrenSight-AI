import os
import binascii
from io import BytesIO
from pathlib import Path
import cv2
import numpy as np
import pickle
import base64
import json
import logging
import torch
from torchvision import transforms, models
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CurrencyDetectorBackend")

APP_VERSION = "0.3.0"
app = FastAPI(
    title="CurrenSight AI - Indian Banknote Perception API",
    version=APP_VERSION,
    description="Accessibility-oriented Indian banknote denomination vision assistance with Deep Learning.",
)

# Resolve project resources independently of the process working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(
    os.getenv("CURRENSIGHT_MODEL_DIR", os.getenv("DRISHTISCAN_MODEL_DIR", os.getenv("RUPEESCAN_MODEL_DIR", PROJECT_ROOT / "backend" / "models")))
).resolve()
FRONTEND_DIR = Path(
    os.getenv("CURRENSIGHT_FRONTEND_DIR", os.getenv("DRISHTISCAN_FRONTEND_DIR", os.getenv("RUPEESCAN_FRONTEND_DIR", PROJECT_ROOT / "frontend")))
).resolve()
MAX_IMAGE_BYTES = int(os.getenv("CURRENSIGHT_MAX_IMAGE_BYTES", os.getenv("DRISHTISCAN_MAX_IMAGE_BYTES", os.getenv("RUPEESCAN_MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))))
MAX_IMAGE_PIXELS = int(os.getenv("CURRENSIGHT_MAX_IMAGE_PIXELS", os.getenv("DRISHTISCAN_MAX_IMAGE_PIXELS", os.getenv("RUPEESCAN_MAX_IMAGE_PIXELS", str(16_000_000)))))
COLOR_THRESHOLD = float(os.getenv("CURRENSIGHT_COLOR_THRESHOLD", os.getenv("DRISHTISCAN_COLOR_THRESHOLD", os.getenv("RUPEESCAN_COLOR_THRESHOLD", "0.535"))))
AUTHENTICITY_EXPERIMENTAL_ENABLED = (
    os.getenv("CURRENSIGHT_ENABLE_EXPERIMENTAL_AUTHENTICITY", os.getenv("DRISHTISCAN_ENABLE_EXPERIMENTAL_AUTHENTICITY", os.getenv("RUPEESCAN_ENABLE_EXPERIMENTAL_AUTHENTICITY", "false")))
).lower() in {"1", "true", "yes", "on"}

# Model Global Variable
MODEL = None
MODEL_LOAD_ERRORS = {}
MODEL_PATH = MODEL_DIR / "currency_classifier.pkl"
ORB = cv2.ORB_create(nfeatures=500)
BF_MATCHER = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# PyTorch Model Settings
PYTORCH_MODEL = None
PYTORCH_MODEL_PATH = MODEL_DIR / "currency_classifier_pytorch.pth"
PYTORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Authenticity Model Settings
AUTHENTICITY_MODEL = None
AUTHENTICITY_MODEL_PATH = MODEL_DIR / "currency_authenticity_pytorch.pth"

PYTORCH_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define Classes
CLASSES = {
    0: "10_rupees",
    1: "20_rupees",
    2: "50_rupees",
    3: "100_rupees",
    4: "200_rupees",
    5: "500_rupees",
    6: "2000_rupees"
}

# Friendly display names for speech feedback
FRIENDLY_NAMES = {
    "10_rupees": "Ten Rupees",
    "20_rupees": "Twenty Rupees",
    "50_rupees": "Fifty Rupees",
    "100_rupees": "One Hundred Rupees",
    "200_rupees": "Two Hundred Rupees",
    "500_rupees": "Five Hundred Rupees",
    "2000_rupees": "Two Thousand Rupees"
}

def load_model():
    global MODEL
    if not os.path.exists(MODEL_PATH):
        MODEL_LOAD_ERRORS["classic"] = "file_not_found"
        logger.warning(f"Model file not found at '{MODEL_PATH}'. Running in dummy inference mode.")
        return False
    try:
        with open(MODEL_PATH, "rb") as f:
            MODEL = pickle.load(f)
        # Convert lists back to numpy arrays for fast matching
        for t in MODEL["templates"]:
            if t["descriptors"] is not None:
                t["descriptors"] = np.array(t["descriptors"], dtype=np.uint8)
        MODEL_LOAD_ERRORS.pop("classic", None)
        logger.info(f"Loaded CV model with {len(MODEL['templates'])} templates.")
        return True
    except Exception as e:
        MODEL = None
        MODEL_LOAD_ERRORS["classic"] = "load_failed"
        logger.error(f"Failed to load model: {e}")
        return False

def load_pytorch_model():
    global PYTORCH_MODEL
    if not os.path.exists(PYTORCH_MODEL_PATH):
        MODEL_LOAD_ERRORS["pytorch"] = "file_not_found"
        logger.warning(f"PyTorch model file not found at '{PYTORCH_MODEL_PATH}'. Running in fallback mode.")
        return False
    try:
        logger.info(f"Loading PyTorch ResNet18 model on {PYTORCH_DEVICE}...")
        model = models.resnet18()
        model.fc = torch.nn.Linear(model.fc.in_features, 7)
        model.load_state_dict(torch.load(PYTORCH_MODEL_PATH, map_location=PYTORCH_DEVICE, weights_only=True))
        model.to(PYTORCH_DEVICE)
        model.eval()
        PYTORCH_MODEL = model
        MODEL_LOAD_ERRORS.pop("pytorch", None)
        logger.info("Successfully loaded PyTorch model for currency classification.")
        return True
    except Exception as e:
        PYTORCH_MODEL = None
        MODEL_LOAD_ERRORS["pytorch"] = "load_failed"
        logger.error(f"Failed to load PyTorch model: {e}")
        return False

def load_authenticity_model():
    global AUTHENTICITY_MODEL
    if not AUTHENTICITY_EXPERIMENTAL_ENABLED:
        MODEL_LOAD_ERRORS.pop("authenticity", None)
        logger.info(
            "Experimental authenticity model is disabled. Set "
            "CURRENSIGHT_ENABLE_EXPERIMENTAL_AUTHENTICITY=true to opt in."
        )
        return False
    if not os.path.exists(AUTHENTICITY_MODEL_PATH):
        MODEL_LOAD_ERRORS["authenticity"] = "file_not_found"
        logger.warning(f"Authenticity PyTorch model file not found at '{AUTHENTICITY_MODEL_PATH}'. Currency authenticity verification will be disabled.")
        return False
    try:
        logger.info(f"Loading Authenticity PyTorch ResNet18 model on {PYTORCH_DEVICE}...")
        model = models.resnet18()
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(torch.load(AUTHENTICITY_MODEL_PATH, map_location=PYTORCH_DEVICE, weights_only=True))
        model.to(PYTORCH_DEVICE)
        model.eval()
        AUTHENTICITY_MODEL = model
        MODEL_LOAD_ERRORS.pop("authenticity", None)
        logger.info("Successfully loaded PyTorch model for currency authenticity verification.")
        return True
    except Exception as e:
        AUTHENTICITY_MODEL = None
        MODEL_LOAD_ERRORS["authenticity"] = "load_failed"
        logger.error(f"Failed to load Authenticity PyTorch model: {e}")
        return False

# Initialize models
model_loaded = load_model()
pytorch_model_loaded = load_pytorch_model()
authenticity_model_loaded = load_authenticity_model()

class BoundingBox(BaseModel):
    x: float
    y: float
    w: float
    h: float


class DetectionResult(BaseModel):
    class_id: int
    class_name: str
    friendly_name: str
    confidence: float
    message: str
    authentic: Optional[bool] = None
    authenticity_confidence: Optional[float] = None
    authenticity_experimental: bool = False
    bbox: Optional[BoundingBox] = None
    error_code: Optional[str] = None


class ImageValidationError(ValueError):
    def __init__(self, message, error_code, status_code=400):
        super().__init__(message)
        self.error_code = error_code
        self.status_code = status_code


def error_result(friendly_name, message, error_code):
    return {
        "class_id": -1,
        "class_name": "error",
        "friendly_name": friendly_name,
        "confidence": 0.0,
        "message": message,
        "error_code": error_code,
    }


def decode_image_bytes(contents):
    """Validate compressed image bytes before decoding them with OpenCV."""
    if not contents:
        raise ImageValidationError("Image file is empty", "empty_image")
    if len(contents) > MAX_IMAGE_BYTES:
        raise ImageValidationError(
            "Image exceeds the configured upload limit", "image_too_large", status_code=413
        )

    try:
        with Image.open(BytesIO(contents)) as candidate:
            width, height = candidate.size
            if width <= 0 or height <= 0:
                raise ImageValidationError("Image dimensions are invalid", "invalid_dimensions")
            if width * height > MAX_IMAGE_PIXELS:
                raise ImageValidationError(
                    "Image dimensions exceed the configured pixel limit",
                    "too_many_pixels",
                    status_code=413,
                )
            candidate.verify()
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ImageValidationError("Failed to decode image file", "invalid_image") from exc

    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ImageValidationError("Failed to decode image file", "invalid_image")
    return image

def extract_features(img):
    """
    Extracts HSV histogram and ORB descriptors.
    """
    # HSV Histogram
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    hist_flat = hist.flatten()
    
    # ORB descriptors
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, descriptors = ORB.detectAndCompute(gray, None)
    
    return hist_flat, descriptors

def check_color_consistency(query_hist, class_id):
    """
    Checks if the color histogram of the query image matches the reference templates of the predicted class.
    Returns True if consistent, False if color mismatch (e.g. grayscale/sketch or wrong color).
    """
    global MODEL
    if MODEL is None or "templates" not in MODEL:
        return True, 1.0 # If classic templates are not loaded, bypass check
        
    scores = []
    for template in MODEL["templates"]:
        if template["class_id"] == class_id:
            t_hist = template["histogram"]
            corr = cv2.compareHist(query_hist, t_hist, cv2.HISTCMP_CORREL)
            score = max(0.0, (corr + 1) / 2.0)
            scores.append(score)
            
    if not scores:
        return True, 1.0
        
    avg_score = np.mean(scores)
    # Threshold for color consistency (0.535 is a safe boundary based on diagnostics)
    is_consistent = avg_score >= COLOR_THRESHOLD
    logger.info(f"Color consistency score: {avg_score:.4f} (Threshold: {COLOR_THRESHOLD})")
    return is_consistent, avg_score

def check_texture_validity(img):
    """
    Checks if the texture complexity and edge density matches a printed banknote.
    Returns True if valid, False if it resembles a hand-drawn sketch or plain paper.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Laplacian Variance (measures local contrast/sharpness)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 2. Canny Edge Density (percentage of pixels that form high-frequency edges)
    canny = cv2.Canny(gray, 50, 150)
    edge_density = np.mean(canny == 255) * 100
    
    # Thresholds determined by stats diagnostics:
    # Real notes have >15% edge density and >2500 Laplacian variance.
    # Hand-drawn sketches have ~7.7% edge density and <1000 Laplacian variance.
    is_valid = (edge_density >= 11.0) and (lap_var >= 1300)
    logger.info(f"Texture check: Edge Density = {edge_density:.2f}% (Threshold: >=11.0%), Laplacian Var = {lap_var:.2f} (Threshold: >=1300)")
    return is_valid, edge_density, lap_var

def find_banknote_bbox(img):
    """
    Finds the bounding box of the banknote in the image using contour detection.
    Returns normalized coordinates {"x": x, "y": y, "w": w, "h": h} or None.
    """
    try:
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection
        edged = cv2.Canny(blurred, 30, 150)
        
        # Dilate to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edged, kernel, iterations=1)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_bbox = None
        max_area = 0
        min_area = w * h * 0.04 # Banknote must cover at least 4% of image area
        
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
                
            # Get bounding rect
            rx, ry, rw, rh = cv2.boundingRect(c)
            
            # Check aspect ratio (banknote is wide, so allow 1.2 to 3.6 aspect ratio)
            aspect_ratio = max(rw, rh) / min(rw, rh)
            if 1.2 <= aspect_ratio <= 3.6:
                if area > max_area:
                    max_area = area
                    best_bbox = {
                        "x": float(rx) / w,
                        "y": float(ry) / h,
                        "w": float(rw) / w,
                        "h": float(rh) / h
                    }
        return best_bbox
    except Exception as e:
        logger.error(f"Error in find_banknote_bbox: {e}")
        return None

def perform_inference(img):
    """
    Main inference function. Runs PyTorch model if available, otherwise falls back to classic CV matching.
    """
    global PYTORCH_MODEL
    
    if PYTORCH_MODEL is not None:
        try:
            # OpenCV image is BGR, PyTorch ResNet expects RGB PIL Image
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            
            # Apply transform and move to GPU/CPU
            tensor_img = PYTORCH_TRANSFORM(pil_img).unsqueeze(0).to(PYTORCH_DEVICE)
            
            with torch.no_grad():
                outputs = PYTORCH_MODEL(tensor_img)
                probs = torch.softmax(outputs, dim=1)[0]
                confidence, predicted_idx = torch.max(probs, dim=0)
                
                class_id = int(predicted_idx.item())
                conf_val = float(confidence.item())
                
            # Confidence threshold to avoid detecting random backgrounds as banknotes
            THRESHOLD = 0.55
            if conf_val > THRESHOLD:
                class_name = CLASSES[class_id]
                logger.info(f"Denomination prediction: {class_name} (Confidence: {conf_val:.4f})")
                
                # --- RUN STAGE 2: AUTHENTICITY CLASSIFIER ---
                is_authentic = None
                auth_conf = None
                auth_msg = ""
                
                global AUTHENTICITY_MODEL
                if AUTHENTICITY_MODEL is not None:
                    try:
                        with torch.no_grad():
                            auth_outputs = AUTHENTICITY_MODEL(tensor_img)
                            auth_probs = torch.softmax(auth_outputs, dim=1)[0]
                            auth_confidence, auth_predicted_idx = torch.max(auth_probs, dim=0)
                            
                            auth_label = int(auth_predicted_idx.item())
                            auth_conf = float(auth_confidence.item())
                            is_authentic = (auth_label == 1)
                            
                            # Extract features for verification
                            query_hist, _ = extract_features(img)
                            color_ok, color_score = check_color_consistency(query_hist, class_id)
                            texture_ok, edge_density, lap_var = check_texture_validity(img)
                            
                            # Unified Dynamic Guard Rails
                            if conf_val < 0.80:
                                # 1. If it's a hand-drawn sketch (extremely sparse edges and low variance)
                                if edge_density < 9.0 and lap_var < 1000:
                                    logger.info(f"Texture indicates a hand-drawn sketch (Edge Density: {edge_density:.2f}%, Laplacian Var: {lap_var:.2f}). Overriding to FAKE.")
                                    is_authentic = False
                                    auth_conf = 0.99
                                    auth_msg = f" (Experimental counterfeit warning from sketch verification; edge density: {edge_density:.1f}%)"
                                # 2. If the color is wrong (e.g. grayscale/wrong colors)
                                elif not color_ok:
                                    logger.info(f"Color check failed (Score: {color_score:.4f} < {COLOR_THRESHOLD}). Overriding to FAKE.")
                                    is_authentic = False
                                    auth_conf = 0.95
                                    auth_msg = f" (Experimental counterfeit warning from color verification; color score: {color_score:.2f})"
                                # 3. If it's just blurry/out-of-focus (low texture but correct colors), downgrade to unknown instead of fake
                                elif lap_var < 1300 or edge_density < 11.0:
                                    logger.info(f"Note is blurry or too far (Edge Density: {edge_density:.2f}%, Laplacian Var: {lap_var:.2f}). Downgrading to unknown.")
                                    conf_val = 0.50 # Force it to "Unknown or No Note"
                            else:
                                if is_authentic:
                                    auth_msg = f" (Experimental authenticity estimate: likely genuine, confidence {auth_conf:.2f})"
                                else:
                                    auth_msg = f" (Experimental authenticity estimate: possible counterfeit, confidence {auth_conf:.2f})"
                                
                    except Exception as ae:
                        logger.error(f"Authenticity inference failed: {ae}")
                        # Fail closed: never retain or invent an authenticity result
                        # after an experimental classifier/guard-rail failure.
                        is_authentic = None
                        auth_conf = None
                        auth_msg = ""
                
                # Check if it was downgraded to unknown
                if conf_val > THRESHOLD:
                    friendly_display = FRIENDLY_NAMES[class_name]
                    if AUTHENTICITY_MODEL is not None and is_authentic is not None:
                        friendly_prefix = "Likely genuine " if is_authentic else "Possible counterfeit "
                        friendly_display = friendly_prefix + friendly_display
                    
                    bbox = find_banknote_bbox(img)
                    
                    return {
                        "class_id": class_id,
                        "class_name": class_name,
                        "friendly_name": friendly_display,
                        "confidence": conf_val,
                        "message": f"Detected {FRIENDLY_NAMES[class_name]} with confidence {conf_val:.2f}{auth_msg} (Deep Learning)",
                        "authentic": is_authentic,
                        "authenticity_confidence": auth_conf,
                        "authenticity_experimental": AUTHENTICITY_MODEL is not None,
                        "bbox": bbox
                    }
            else:
                return {
                    "class_id": -1,
                    "class_name": "unknown",
                    "friendly_name": "Unknown or No Note",
                    "confidence": conf_val,
                    "message": f"Align banknote in frame (Deep Learning low conf: {conf_val:.2f})",
                    "authentic": None,
                    "authenticity_confidence": None
                }
        except Exception as e:
            logger.error(f"PyTorch inference failed: {e}. Falling back to classic CV.")
            
    if MODEL is None:
        # Dummy detector fallback
        return {
            "class_id": -1,
            "class_name": "unknown",
            "friendly_name": "No banknote detected",
            "confidence": 0.0,
            "message": "Model not loaded"
        }
        
    # 1. Extract query features
    query_hist, query_desc = extract_features(img)
    
    # 2. Compute color similarities across all templates
    class_color_scores = {cid: [] for cid in CLASSES.keys()}
    
    for template in MODEL["templates"]:
        cid = template["class_id"]
        t_hist = template["histogram"]
        
        # Calculate Correlation distance (ranges from -1 to 1)
        corr = cv2.compareHist(query_hist, t_hist, cv2.HISTCMP_CORREL)
        # Normalize to [0, 1]
        score = max(0.0, (corr + 1) / 2.0)
        class_color_scores[cid].append((score, template))
        
    # Calculate average color score per class
    avg_color_scores = {}
    for cid, items in class_color_scores.items():
        if items:
            avg_color_scores[cid] = np.mean([x[0] for x in items])
        else:
            avg_color_scores[cid] = 0.0
            
    # Sort classes by color similarity
    sorted_classes = sorted(avg_color_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Top 2 candidate classes
    top_candidates = [sorted_classes[0][0], sorted_classes[1][0]]
    
    # 3. Perform ORB feature matching only on the top candidate templates to speed up
    class_orb_scores = {cid: [] for cid in top_candidates}
    
    if query_desc is not None:
        for cid in top_candidates:
            # Match against top templates in this class
            templates_to_test = class_color_scores[cid]
            # Sort templates of this class by color similarity and take top 3
            templates_to_test = sorted(templates_to_test, key=lambda x: x[0], reverse=True)[:3]
            
            for col_score, template in templates_to_test:
                t_desc = template["descriptors"]
                if t_desc is not None and len(t_desc) > 0:
                    try:
                        matches = BF_MATCHER.match(query_desc, t_desc)
                        # Filter matches by distance
                        good_matches = [m for m in matches if m.distance < 45]
                        match_ratio = len(good_matches) / max(1, len(t_desc))
                        class_orb_scores[cid].append(match_ratio)
                    except Exception:
                        class_orb_scores[cid].append(0.0)
                else:
                    class_orb_scores[cid].append(0.0)
    
    # Calculate average ORB match ratio per candidate class
    avg_orb_scores = {}
    for cid in top_candidates:
        scores = class_orb_scores[cid]
        avg_orb_scores[cid] = np.mean(scores) if scores else 0.0
        
    # 4. Combine color and texture scores
    final_scores = {}
    for cid in top_candidates:
        color_w = 0.7
        orb_w = 0.3
        
        # If no query ORB features were detected (e.g. solid color backgrounds, blurred image)
        if query_desc is None or avg_orb_scores[cid] == 0:
            final_scores[cid] = avg_color_scores[cid]
        else:
            final_scores[cid] = (color_w * avg_color_scores[cid]) + (orb_w * min(1.0, avg_orb_scores[cid] * 10))  # Scale ORB score for better balance
            
    # Find winning class
    best_cid = max(final_scores, key=final_scores.get)
    best_score = final_scores[best_cid]
    
    # Apply a confidence threshold (e.g. 0.45)
    THRESHOLD = 0.45
    if best_score > THRESHOLD:
        class_name = CLASSES[best_cid]
        bbox = find_banknote_bbox(img)
        return {
            "class_id": best_cid,
            "class_name": class_name,
            "friendly_name": FRIENDLY_NAMES[class_name],
            "confidence": float(best_score),
            "message": f"Detected {FRIENDLY_NAMES[class_name]} with confidence {best_score:.2f}",
            "bbox": bbox
        }
    else:
        return {
            "class_id": -1,
            "class_name": "unknown",
            "friendly_name": "Unknown or No Note",
            "confidence": float(best_score),
            "message": "Align banknote in frame"
        }

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(self), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "connect-src 'self' ws: wss:; "
        "img-src 'self' data: blob:; "
        "media-src 'self' blob:; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "script-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
    )
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/api/live")
def live():
    return {"status": "alive", "version": APP_VERSION}


@app.get("/api/health")
def health():
    denomination_ready = PYTORCH_MODEL is not None or MODEL is not None
    if PYTORCH_MODEL is not None:
        denomination_engine = "pytorch"
    elif MODEL is not None:
        denomination_engine = "classic_cv"
    else:
        denomination_engine = "unavailable"
    return {
        "status": "ok" if denomination_ready else "degraded",
        "version": APP_VERSION,
        "denomination_engine": denomination_engine,
        "pytorch_model_loaded": PYTORCH_MODEL is not None,
        "classic_model_loaded": MODEL is not None,
        "authenticity_model_loaded": AUTHENTICITY_MODEL is not None,
        "authenticity_experimental_enabled": AUTHENTICITY_EXPERIMENTAL_ENABLED,
        "model_errors": dict(MODEL_LOAD_ERRORS),
        "limits": {"max_image_bytes": MAX_IMAGE_BYTES, "max_image_pixels": MAX_IMAGE_PIXELS},
        "supported_classes": list(CLASSES.values()),
        "device": str(PYTORCH_DEVICE) if PYTORCH_MODEL is not None else "cpu"
    }

@app.post("/api/detect", response_model=DetectionResult)
async def detect_currency(file: UploadFile = File(...)):
    """
    REST Endpoint for one-off image uploads.
    """
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail={
            "error_code": "unsupported_media_type",
            "message": "Upload must use an image media type",
        })
    try:
        contents = await file.read(MAX_IMAGE_BYTES + 1)
        img = decode_image_bytes(contents)
    except ImageValidationError as exc:
        raise HTTPException(status_code=exc.status_code, detail={
            "error_code": exc.error_code,
            "message": str(exc),
        }) from exc
    finally:
        await file.close()

    result = await run_in_threadpool(perform_inference, img)
    return result

@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    """
    WebSocket endpoint for real-time video frame streaming.
    Receives base64-encoded JPEG image frames and returns JSON predictions.
    """
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        while True:
            # Receive text data (expecting base64 image data)
            data = await websocket.receive_text()
            
            try:
                if len(data) > ((MAX_IMAGE_BYTES * 4 // 3) + 4096):
                    await websocket.send_text(json.dumps(error_result(
                        "Frame Too Large",
                        "Frame exceeds the configured upload limit",
                        "image_too_large",
                    )))
                    continue

                # Remove header if present in data URL (e.g., data:image/jpeg;base64,...)
                if "," in data:
                    header, data = data.split(",", 1)
                
                # Decode and validate the frame.
                img_bytes = base64.b64decode(data, validate=True)
                img = decode_image_bytes(img_bytes)
                    
                # Run inference
                result = await run_in_threadpool(perform_inference, img)
                
                # Send result back
                await websocket.send_text(json.dumps(result))
                
            except (binascii.Error, ImageValidationError) as exc:
                error_code = getattr(exc, "error_code", "invalid_base64")
                await websocket.send_text(json.dumps(error_result(
                    "Invalid Frame", "The camera frame could not be decoded", error_code
                )))
            except Exception:
                logger.exception("Unexpected error processing WebSocket frame")
                await websocket.send_text(json.dumps(error_result(
                    "Processing Error", "The frame could not be processed", "processing_error"
                )))
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

# Serve frontend folder statically
frontend_path = str(FRONTEND_DIR)
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
    logger.info(f"Mounted static frontend at {frontend_path}")
else:
    logger.warning(f"Frontend directory '{frontend_path}' not found yet. Run frontend initialization.")
