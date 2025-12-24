from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import base64
import cv2
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# --- Konfiguracja ---
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./models/lung_cancer_model.pth"

# Transformacje (identyczne jak przy treningu)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# --- Silnik Grad-CAM ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.feature_maps = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def save_fmaps(module, input, output):
            self.feature_maps = output.detach()

        def save_grads(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hooks.append(self.target_layer.register_forward_hook(save_fmaps))
        self.hooks.append(self.target_layer.register_full_backward_hook(save_grads))

    def generate(self, img_tensor):
        self.model.zero_grad()
        output = self.model(img_tensor)

        # Backward dla klasyfikacji binarnej
        output.backward()

        # Global Average Pooling gradientów
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # Generowanie mapy (weighted sum)
        cam = torch.sum(weights * self.feature_maps, dim=1).squeeze()

        # ReLU - interesują nas tylko pozytywne wpływy na wynik "Cancer"
        cam = np.maximum(cam.cpu().numpy(), 0)

        # Normalizacja
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam

    def remove(self):
        for hook in self.hooks:
            hook.remove()


# --- Funkcje pomocnicze ---
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


model = load_model()


def apply_heatmap(img_array, heatmap, alpha=0.4):
    # Progowanie: usuwamy słabe aktywacje (poniżej 30% maxa), aby nie "śmieciły" na obrazie
    heatmap[heatmap < 0.3] = 0

    h, w = img_array.shape[:2]
    heatmap_res = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_res)

    # Nakładanie koloru
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Tworzenie nałożonego obrazu
    superimposed = cv2.addWeighted(img_array, 1.0, heatmap_color, alpha, 0)
    return superimposed


# --- API Endpoints ---
@app.route('/api/analyze', methods=['POST'])
def analyze():
    if model is None:
        return jsonify({'error': 'Model file not found'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    try:
        # Odczyt obrazu
        img_raw = Image.open(file.stream).convert('RGB')
        img_array = np.array(img_raw)

        # Przygotowanie tensora
        img_tensor = transform(img_raw).unsqueeze(0).to(DEVICE)
        img_tensor.requires_grad_()  # Wymagane do obliczenia gradientów

        # 1. Predykcja
        output = model(img_tensor)
        prob = torch.sigmoid(output).item()

        # 2. Grad-CAM (używamy layer3 dla lepszej precyzji w medycynie)
        target_layer = model.layer3[-1]
        gc = GradCAM(model, target_layer)

        try:
            heatmap = gc.generate(img_tensor)
        finally:
            gc.remove()

        # 3. Wizualizacja
        vis_array = apply_heatmap(img_array, heatmap)

        # Konwersja do Base64
        vis_pil = Image.fromarray(vis_array)
        buf = io.BytesIO()
        vis_pil.save(buf, format="JPEG", quality=90)
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        return jsonify({
            'success': True,
            'probability': round(prob * 100, 2),
            'risk_level': 'high' if prob > 0.7 else ('medium' if prob > 0.4 else 'low'),
            'heatmap': f'data:image/jpeg;base64,{img_b64}'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print(f"Server starting on {DEVICE}...")
    app.run(debug=True, host='0.0.0.0', port=7860)