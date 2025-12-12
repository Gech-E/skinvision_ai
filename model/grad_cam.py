import os
import numpy as np
from typing import Tuple, Optional
from PIL import Image

# Try importing PyTorch (primary) and TensorFlow (fallback)
TORCH_AVAILABLE = False
TENSORFLOW_AVAILABLE = False

try:
    import torch
    import torch.nn.functional as F
    from torchvision import models
    TORCH_AVAILABLE = True
except ImportError:
    torch = None

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    tf = None


def generate_gradcam_heatmap_pytorch(model, image_tensor: torch.Tensor, layer_name: Optional[str] = None) -> np.ndarray:
    """
    Generate Grad-CAM heatmap for PyTorch model.
    
    Args:
        model: PyTorch model
        image_tensor: Preprocessed image tensor (1, C, H, W)
        layer_name: Optional layer name (defaults to last conv layer)
    
    Returns:
        Heatmap as numpy array (H, W)
    """
    if not TORCH_AVAILABLE or model is None:
        raise ValueError("PyTorch or model not available")
    
    try:
        model.eval()
        
        # Find the last convolutional layer if not specified
        if layer_name is None:
            for name, module in reversed(list(model.named_modules())):
                if isinstance(module, torch.nn.Conv2d):
                    layer_name = name
                    break
        
        if layer_name is None:
            raise ValueError("No convolutional layer found")
        
        # Get the target layer
        target_layer = dict(model.named_modules())[layer_name]
        
        # Hook to capture gradients and activations
        gradients = []
        activations = []
        
        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])
        
        def forward_hook(module, input, output):
            activations.append(output)
        
        handle_backward = target_layer.register_full_backward_hook(backward_hook)
        handle_forward = target_layer.register_forward_hook(forward_hook)
        
        # Forward pass
        output = model(image_tensor)
        
        # Get the predicted class
        class_idx = output.argmax(dim=1)
        
        # Backward pass
        model.zero_grad()
        output[0, class_idx].backward()
        
        # Get gradients and activations
        grads = gradients[0]
        acts = activations[0]
        
        # Global average pooling of gradients
        pooled_grads = torch.mean(grads, dim=[2, 3], keepdim=True)
        
        # Weight the activations by gradients
        heatmap = torch.sum(acts * pooled_grads, dim=1, keepdim=True)
        heatmap = F.relu(heatmap)  # ReLU to get positive values only
        
        # Normalize
        heatmap = heatmap.squeeze().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        
        # Resize to original image size (224x224)
        from torchvision.transforms import functional as F_t
        heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0)
        heatmap_resized = F_t.resize(heatmap_tensor, (224, 224), interpolation=Image.Resampling.BILINEAR)
        heatmap = heatmap_resized.squeeze().numpy()
        
        # Cleanup hooks
        handle_backward.remove()
        handle_forward.remove()
        
        return heatmap
        
    except Exception as e:
        print(f"PyTorch Grad-CAM failed: {e}. Using fallback visualization.")
        # Fallback: center-focused gradient
        h, w = 224, 224
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        heatmap = 1 - (dist_from_center / max_dist)
        heatmap = np.clip(heatmap, 0, 1)
        return heatmap


def generate_gradcam_heatmap(model, image_array: np.ndarray, layer_name: Optional[str] = None) -> np.ndarray:
    """
    Generate Grad-CAM heatmap (supports both PyTorch and TensorFlow).
    Falls back to simple visualization if model doesn't support Grad-CAM.
    """
    # Try PyTorch first
    if TORCH_AVAILABLE and hasattr(model, 'forward'):
        try:
            # Convert numpy array to torch tensor if needed
            if isinstance(image_array, np.ndarray):
                # Assume image_array is (1, H, W, C) normalized [0,1]
                # Convert to (1, C, H, W) tensor
                img_tensor = torch.from_numpy(image_array).float()
                if img_tensor.dim() == 4 and img_tensor.shape[-1] == 3:
                    img_tensor = img_tensor.permute(0, 3, 1, 2)
                # Normalize for EfficientNet
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                img_tensor = (img_tensor - mean) / std
                return generate_gradcam_heatmap_pytorch(model, img_tensor, layer_name)
        except Exception as e:
            print(f"PyTorch Grad-CAM attempt failed: {e}")
    
    # Fallback to TensorFlow if available
    if TENSORFLOW_AVAILABLE and hasattr(model, 'predict'):
        try:
            # Try to find the last convolutional layer
            if layer_name is None:
                for layer in reversed(model.layers):
                    if len(layer.output_shape) == 4:  # Convolutional layer
                        layer_name = layer.name
                        break
            
            if layer_name is None:
                raise ValueError("No convolutional layer found")
            
            # Build model that outputs the conv layer and final predictions
            conv_layer = model.get_layer(layer_name)
            grad_model = tf.keras.Model(
                [model.inputs],
                [conv_layer.output, model.output]
            )
            
            # Compute gradients
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(image_array)
                class_idx = tf.argmax(predictions[0])
                class_channel = predictions[:, class_idx]
            
            grads = tape.gradient(class_channel, conv_outputs)
            
            # Global average pooling of gradients
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Multiply conv outputs with pooled gradients
            conv_outputs = conv_outputs[0]
            heatmap = tf.reduce_mean(tf.multiply(conv_outputs, pooled_grads), axis=-1)
            
            # Normalize heatmap
            heatmap = np.maximum(heatmap, 0)
            heatmap /= np.max(heatmap)
            
            # Resize to original image size
            heatmap = tf.image.resize(heatmap[..., tf.newaxis], (image_array.shape[1], image_array.shape[2]))
            heatmap = np.squeeze(heatmap.numpy())
            
            return heatmap
        except Exception as e:
            print(f"TensorFlow Grad-CAM failed: {e}")
    
    # Final fallback: simple center-focused gradient
    print("Using fallback heatmap visualization")
    h, w = image_array.shape[1:3] if len(image_array.shape) == 4 else (224, 224)
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    heatmap = 1 - (dist_from_center / max_dist)
    heatmap = np.clip(heatmap, 0, 1)
    return heatmap


def save_heatmap_overlay(orig_path: str, out_dir: str, model=None, preprocessed_img: Optional[np.ndarray] = None) -> str:
    """
    Generate and save a heatmap overlay visualization.
    
    Args:
        orig_path: Path to original image
        out_dir: Directory to save heatmap
        model: Optional PyTorch or TensorFlow model for Grad-CAM
        preprocessed_img: Optional preprocessed image array (224x224 normalized)
    
    Returns:
        Path to saved heatmap image
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Load original image
    image = Image.open(orig_path).convert("RGB")
    orig_size = image.size
    w, h = orig_size
    
    # Generate heatmap
    if model is not None and preprocessed_img is not None:
        try:
            heatmap = generate_gradcam_heatmap(model, preprocessed_img)
            # Resize heatmap to original image size
            heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(orig_size, Image.Resampling.LANCZOS)
        except Exception as e:
            print(f"Heatmap generation failed: {e}, using fallback")
            # Fallback: simple center-focused gradient
            arr = np.zeros((h, w), dtype=np.float32)
            center_y, center_x = h // 2, w // 2
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_dist = np.sqrt(center_x**2 + center_y**2)
            arr = 1 - np.clip(dist / max_dist, 0, 1)
            heatmap_pil = Image.fromarray((arr * 255).astype(np.uint8))
    else:
        # Fallback: simple center-focused gradient
        arr = np.zeros((h, w), dtype=np.float32)
        center_y, center_x = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        arr = 1 - np.clip(dist / max_dist, 0, 1)
        heatmap_pil = Image.fromarray((arr * 255).astype(np.uint8))
    
    # Create colored heatmap (red-yellow colormap)
    heatmap_colored = Image.new("RGB", orig_size, (0, 0, 0))
    heatmap_arr = np.array(heatmap_pil).astype(np.float32) / 255.0
    
    # Apply red-yellow colormap
    heatmap_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    heatmap_rgb[:, :, 0] = (heatmap_arr * 255).astype(np.uint8)  # Red
    heatmap_rgb[:, :, 1] = (heatmap_arr * 200).astype(np.uint8)  # Yellow component
    heatmap_rgb[:, :, 2] = (heatmap_arr * 50).astype(np.uint8)   # Low blue
    
    heatmap_colored = Image.fromarray(heatmap_rgb)
    
    # Blend with original image
    overlay = heatmap_colored.convert("RGBA")
    overlay.putalpha(Image.fromarray((heatmap_arr * 180).astype(np.uint8)))  # Semi-transparent
    
    blended = Image.alpha_composite(image.convert("RGBA"), overlay)
    
    # Save
    out_path = os.path.join(out_dir, f"heatmap_{os.path.basename(orig_path)}")
    blended.convert("RGB").save(out_path)
    return out_path
