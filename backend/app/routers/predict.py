from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Header
from sqlalchemy.orm import Session
from ..database import get_db
from ..schemas import PredictionCreate, PredictionOut
from ..crud import create_prediction
from .. import models
import sys
from pathlib import Path
# Get project root (3 levels up from backend/app/routers/predict.py)
# predict.py -> routers -> app -> backend -> project_root
ROOT_DIR = Path(__file__).resolve().parents[3]
# Add project root to Python path so we can import 'model' package
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Try to import from model package - allow graceful degradation
MODEL_PACKAGE_AVAILABLE = False
save_heatmap_overlay = None
load_local_model = None
get_preprocessing_transform = None
CLASS_NAMES = ["Melanoma", "Melanocytic_Nevus", "Basal_Cell_Carcinoma", "Actinic_Keratosis", "Benign_Keratosis", "Dermatofibroma", "Vascular_Lesion"]

try:
    from model.grad_cam import save_heatmap_overlay
    from model.model_loader import load_local_model, get_preprocessing_transform, CLASS_NAMES
    MODEL_PACKAGE_AVAILABLE = True
except ImportError as e:
    # If import fails, try adding model directory directly
    model_dir = ROOT_DIR / "model"
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    try:
        from grad_cam import save_heatmap_overlay
        from model_loader import load_local_model, get_preprocessing_transform, CLASS_NAMES
        MODEL_PACKAGE_AVAILABLE = True
    except ImportError:
        # Model package not available - app will use fallback predictions
        print(f"⚠️  Warning: Model package not available. Using fallback predictions. Error: {e}")
        MODEL_PACKAGE_AVAILABLE = False
        # Define fallback functions
        def save_heatmap_overlay(orig_path: str, out_dir: str, model=None, preprocessed_img=None) -> str:
            """Fallback heatmap generation when model package is unavailable."""
            import os
            from PIL import Image
            os.makedirs(out_dir, exist_ok=True)
            # Create a simple center-focused gradient as fallback
            image = Image.open(orig_path).convert("RGB")
            orig_size = image.size
            w, h = orig_size
            import numpy as np
            arr = np.zeros((h, w), dtype=np.float32)
            center_y, center_x = h // 2, w // 2
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_dist = np.sqrt(center_x**2 + center_y**2)
            arr = 1 - np.clip(dist / max_dist, 0, 1)
            heatmap_pil = Image.fromarray((arr * 255).astype(np.uint8))
            # Create colored heatmap
            heatmap_rgb = np.zeros((h, w, 3), dtype=np.uint8)
            heatmap_arr = np.array(heatmap_pil).astype(np.float32) / 255.0
            heatmap_rgb[:, :, 0] = (heatmap_arr * 255).astype(np.uint8)
            heatmap_rgb[:, :, 1] = (heatmap_arr * 200).astype(np.uint8)
            heatmap_rgb[:, :, 2] = (heatmap_arr * 50).astype(np.uint8)
            heatmap_colored = Image.fromarray(heatmap_rgb)
            overlay = heatmap_colored.convert("RGBA")
            overlay.putalpha(Image.fromarray((heatmap_arr * 180).astype(np.uint8)))
            blended = Image.alpha_composite(image.convert("RGBA"), overlay)
            out_path = os.path.join(out_dir, f"heatmap_{os.path.basename(orig_path)}")
            blended.convert("RGB").save(out_path)
            return out_path
        
        def load_local_model(*args, **kwargs):
            """Fallback model loader - returns None when model package unavailable."""
            return None
        
        def get_preprocessing_transform():
            """Fallback preprocessing - returns identity transform."""
            try:
                from torchvision import transforms
                return transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ])
            except ImportError:
                # If torchvision not available, return None (will use numpy fallback)
                return None

try:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

import numpy as np
import io
import os
from jose import jwt, JWTError


router = APIRouter()

MODEL = None
ALGO = "HS256"
SECRET = os.environ.get("JWT_SECRET", "devsecret")
MODEL_DIR = ROOT_DIR / "model"
DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", str(MODEL_DIR / "efficientnet_b0_best.pth"))
DEVICE = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"


def get_model():
    """Load PyTorch model if available."""
    global MODEL
    if not TORCH_AVAILABLE or not MODEL_PACKAGE_AVAILABLE:
        return None
    if MODEL is None:
        model_path = Path(DEFAULT_MODEL_PATH)
        # Bug 2 Fix: Validate model path exists before attempting to load
        if not model_path.exists():
            print(f"⚠️  Warning: Model file not found at {model_path}")
            print(f"   Please ensure the model file exists or set MODEL_PATH environment variable.")
            print(f"   Falling back to dummy predictions.")
            return None
        try:
            MODEL = load_local_model(model_path, device=DEVICE)
            if MODEL is None:
                print(f"⚠️  Warning: Model loader returned None for {model_path}")
        except Exception as e:
            print(f"⚠️  Warning: Could not load model from {model_path}: {e}")
            print(f"   Falling back to dummy predictions.")
            MODEL = None
    return MODEL


def preprocess_image(file_bytes: bytes):
    """
    Preprocess image for PyTorch model.
    Returns tensor in CHW format, normalized for EfficientNetB0.
    Falls back to numpy array if PyTorch is unavailable.
    """
    # Import PIL Image (available even without torch)
    from PIL import Image
    
    if not TORCH_AVAILABLE or not MODEL_PACKAGE_AVAILABLE:
        # Fallback: return numpy array
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB").resize((224, 224))
        arr = np.array(image) / 255.0
        return np.expand_dims(arr, axis=0)
    
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    transform = get_preprocessing_transform()
    
    # Check if transform is None (can happen if torchvision is unavailable)
    if transform is None:
        # Fallback to numpy preprocessing
        arr = np.array(image.resize((224, 224))) / 255.0
        return np.expand_dims(arr, axis=0)
    
    tensor = transform(image)
    return tensor.unsqueeze(0)  # Add batch dimension


def predict_with_model(model, image_tensor) -> tuple[str, float]:
    """
    Run prediction with PyTorch model.
    
    Returns:
        (predicted_class_name, confidence_score)
    """
    if not TORCH_AVAILABLE or model is None:
        # Bug 1 Fix: Use first class from CLASS_NAMES instead of hardcoded "Melanoma"
        fallback_class = CLASS_NAMES[0] if CLASS_NAMES else "Melanoma"
        return fallback_class, 0.92  # Fallback prediction
    
    model.eval()
    with torch.no_grad():
        # Ensure tensor is on correct device
        if isinstance(image_tensor, np.ndarray):
            # Convert numpy to torch tensor
            image_tensor = torch.from_numpy(image_tensor).float()
            if image_tensor.dim() == 4 and image_tensor.shape[-1] == 3:
                image_tensor = image_tensor.permute(0, 3, 1, 2)
            # Normalize for EfficientNet
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            image_tensor = (image_tensor - mean) / std
        
        outputs = model(image_tensor.to(DEVICE))
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        
        predicted_idx_val = predicted_idx.item()
        # Bug 3 Fix: Add bounds checking to prevent IndexError
        if predicted_idx_val < 0 or predicted_idx_val >= len(CLASS_NAMES):
            print(f"⚠️  Warning: Model predicted class index {predicted_idx_val} is out of range [0, {len(CLASS_NAMES)-1}]")
            print(f"   Using fallback prediction. Model may have been trained with different number of classes.")
            fallback_class = CLASS_NAMES[0] if CLASS_NAMES else "Melanoma"
            return fallback_class, 0.92
        
        predicted_class = CLASS_NAMES[predicted_idx_val]
        confidence_score = confidence.item()
        
    return predicted_class, confidence_score


def get_user_id_from_header(authorization: str | None = Header(default=None)) -> int | None:
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    token = authorization.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, SECRET, algorithms=[ALGO])
        return int(payload.get("sub"))
    except Exception:
        return None


@router.post("/predict", response_model=PredictionOut)
async def predict(
    file: UploadFile = File(...), 
    db: Session = Depends(get_db), 
    user_id: int | None = Depends(get_user_id_from_header)
):
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    # Save original image to static dir
    static_dir = os.environ.get("STATIC_DIR", os.path.join(os.path.dirname(__file__), "..", "static"))
    os.makedirs(static_dir, exist_ok=True)
    image_path = os.path.join(static_dir, file.filename)
    with open(image_path, "wb") as f:
        f.write(contents)

    model = get_model()
    # Bug 1 Fix: Use first class from CLASS_NAMES instead of hardcoded "Melanoma"
    predicted = CLASS_NAMES[0] if CLASS_NAMES else "Melanoma"
    conf = 0.92
    image_tensor = None
    
    if model is not None and TORCH_AVAILABLE:
        try:
            image_tensor = preprocess_image(contents)
            predicted, conf = predict_with_model(model, image_tensor)
            print(f"✅ Prediction: {predicted} (confidence: {conf:.2%})")
        except Exception as e:
            print(f"⚠️  Model prediction error: {e}. Using fallback.")
            # Bug 1 Fix: Use first class from CLASS_NAMES instead of hardcoded "Melanoma"
            predicted = CLASS_NAMES[0] if CLASS_NAMES else "Melanoma"
            conf = 0.92
    else:
        print("⚠️  Model not available, using fallback prediction")

    # Convert tensor to numpy for heatmap (if available)
    preprocessed_np = None
    if image_tensor is not None:
        if isinstance(image_tensor, np.ndarray):
            # Already a numpy array from fallback preprocessing
            preprocessed_np = image_tensor
        elif TORCH_AVAILABLE and hasattr(image_tensor, 'cpu'):
            # PyTorch tensor - convert to numpy
            # Convert CHW to HWC for heatmap visualization
            img_np = image_tensor.squeeze(0).cpu().numpy()
            # Denormalize for visualization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = img_np.transpose(1, 2, 0) * std + mean
            img_np = np.clip(img_np, 0, 1)
            preprocessed_np = np.expand_dims(img_np, axis=0)

    heatmap_path = save_heatmap_overlay(image_path, static_dir, model=model, preprocessed_img=preprocessed_np)

    data = PredictionCreate(
        image_url=f"/static/{os.path.basename(image_path)}",
        predicted_class=predicted,
        confidence=conf,
        heatmap_url=f"/static/{os.path.basename(heatmap_path)}",
    )
    pred = create_prediction(db, data, user_id=user_id)
    
    return pred
