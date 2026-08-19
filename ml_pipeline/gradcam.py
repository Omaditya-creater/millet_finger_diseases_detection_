import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAMGenerator:
    """
    Gradient-Weighted Class Activation Mapping (Grad-CAM) Generator
    for Visual Explainability and Lesion Severity Estimation in Finger Millet leaves.
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_heatmap(self, input_tensor, target_class=None):
        """
        Generates Grad-CAM heatmap and calculates Lesion Severity Index (LSI %).
        
        Args:
            input_tensor (torch.Tensor): Preprocessed image tensor (1, 3, H, W)
            target_class (int, optional): Target class index. If None, uses top predicted class.
            
        Returns:
            dict: {
                'pred_class': int,
                'confidence': float,
                'heatmap_raw': np.ndarray (28x28 normalized),
                'heatmap_resized': np.ndarray (HxW normalized 0-1),
                'lsi_percentage': float
            }
        """
        input_tensor.requires_grad = True
        outputs = self.model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        
        if target_class is None:
            target_class = torch.argmax(outputs, dim=1).item()
            
        confidence = probs[0, target_class].item()
        
        # Zero out previous gradients
        self.model.zero_grad()
        
        # Backpropagate target class score
        score = outputs[0, target_class]
        score.backward()
        
        # Retrieve target feature maps and backpropagated gradients
        gradients = self.model.last_conv_gradients.data.cpu().numpy()[0] # (C, H_f, W_f)
        features = self.model.last_conv_features.data.cpu().numpy()[0]   # (C, H_f, W_f)
        
        # Calculate channel importance weights alpha
        weights = np.mean(gradients, axis=(1, 2))  # (C,)
        
        # Compute weighted sum of feature maps
        cam = np.zeros(features.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * features[i, :, :]
            
        # Apply ReLU activation to restrict to positive influences
        cam = np.maximum(cam, 0)
        
        # Normalize between 0 and 1
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
        else:
            cam = np.zeros_like(cam)
            
        # Resize heatmap to match input image dimension (H, W)
        H, W = input_tensor.shape[2:]
        heatmap_resized = cv2.resize(cam, (W, H))
        
        # Compute Lesion Severity Index (LSI %) via Otsu / adaptive thresholding on saliency map
        lsi_percentage = float(np.mean(heatmap_resized > 0.35) * 100)
        
        return {
            'target_class': target_class,
            'confidence': confidence,
            'heatmap_raw': cam,
            'heatmap_resized': heatmap_resized,
            'lsi_percentage': round(lsi_percentage, 2)
        }

def overlay_heatmap_on_image(original_img_np, heatmap_resized, alpha=0.5):
    """
    Applies color map to heatmap and blends with original RGB image numpy array (H, W, 3).
    """
    # Convert heatmap 0-1 to 0-255 uint8
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)
    
    # Blend images
    blended = cv2.addWeighted(original_img_np, 1.0 - alpha, color_heatmap, alpha, 0)
    return blended, color_heatmap

if __name__ == "__main__":
    from model import build_model
    m = build_model()
    gradcam = GradCAMGenerator(m)
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    res = gradcam.generate_heatmap(x)
    print(f"Grad-CAM Success! Target Class: {res['target_class']}, Confidence: {res['confidence']:.4f}, LSI: {res['lsi_percentage']}%")
