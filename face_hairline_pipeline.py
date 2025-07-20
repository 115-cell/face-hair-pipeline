#!/usr/bin/env python3
"""
Face + Hairline Webcam Pipeline

Setup:
1. Create & activate a virtual environment in your project root:
   ```bash
   python -m venv .venv
   # Windows PowerShell: .\.venv\Scripts\Activate.ps1
   # Windows cmd:     .\.venv\Scripts\activate
   # macOS/Linux:     source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install opencv-python mtcnn torch torchvision face-parsing scikit-image matplotlib mediapipe
   ```

3. Download BiSeNet weights (`79999_iter.pth`) via your browser from the face-parsing.PyTorch repo and place it in a `models/` folder at your project root.

Usage:
- Ensure this script resides in your project root (alongside `.venv` and `models/`), then run to start real-time webcam processing.
- Press **q** in the window to quit.
"""
import os
import cv2
import numpy as np
import mediapipe as mp
from mtcnn import MTCNN
import torch
import torchvision.transforms as T
from face_parsing.model import BiSeNet
from skimage import measure

# --- Determine project root and weights path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(script_dir) == '.venv':
    project_root = os.path.dirname(script_dir)
else:
    project_root = script_dir
models_dir = os.path.join(project_root, 'models')
weights_path = os.path.join(models_dir, '79999_iter.pth')
if not os.path.isfile(weights_path):
    raise FileNotFoundError(f"Cannot find BiSeNet weights at {weights_path}")

# --- Face detection with padding ---
from mtcnn import MTCNN

def detect_and_crop(img, pad_ratio=0.1):
    detector = MTCNN()
    faces = detector.detect_faces(img)
    if not faces:
        return None
    x, y, w, h = faces[0]['box']
    pad = int(max(w, h) * pad_ratio)
    x0 = max(x - pad, 0)
    y0 = max(y - pad, 0)
    x1 = min(x + w + pad, img.shape[1])
    y1 = min(y + h + pad, img.shape[0])
    return img[y0:y1, x0:x1]

# --- Landmark extraction (MediaPipe Face Mesh) ---
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

def get_landmarks(face_img):
    results = mp_face_mesh.process(face_img)
    if not results.multi_face_landmarks:
        return None
    lm = results.multi_face_landmarks[0].landmark
    h, w, _ = face_img.shape
    return np.array([[int(p.x * w), int(p.y * h)] for p in lm], dtype=np.int32)

# --- Semantic parsing (hair mask) ---
class FaceParser:
    def __init__(self, weights=weights_path, device="cpu"):
        self.device = device
        self.net = BiSeNet(n_classes=19).to(device)
        self.net.load_state_dict(torch.load(weights, map_location=device))
        self.net.eval()
        self.tf = T.Compose([
            T.ToPILImage(),
            T.Resize((512,512)),
            T.ToTensor(),
            T.Normalize((0.5,)*3, (0.5,)*3),
        ])

    def parse(self, face_img):
        tensor = self.tf(face_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.net(tensor)[0]
        return out.argmax(dim=0).cpu().numpy()

# --- Hairline contour extraction ---

def extract_hairline(seg, hair_id=17, orig_shape=None):
    mask = (seg == hair_id).astype(np.uint8)
    if mask.sum() == 0:
        return None
    contours = measure.find_contours(mask, 0.5)
    hairline = max(contours, key=lambda c: c.shape[0])
    if orig_shape:
        h0, w0 = orig_shape[:2]
        h1, w1 = seg.shape
        hairline[:,1] *= (w0 / w1)
        hairline[:,0] *= (h0 / h1)
    return hairline.astype(np.int32)

# --- Real-time webcam processing ---
def run_webcam(device_index=0, pad_ratio=0.1):
    parser = FaceParser()
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        face = detect_and_crop(frame_rgb, pad_ratio)
        disp = frame_bgr
        if face is not None:
            landmarks = get_landmarks(face)
            seg = parser.parse(face)
            hairline = extract_hairline(seg, orig_shape=face.shape)
            overlay = face.copy()
            if landmarks is not None:
                for (x, y) in landmarks:
                    cv2.circle(overlay, (x, y), 1, (0, 0, 255), -1)
            if hairline is not None:
                pts = np.fliplr(hairline)
                cv2.polylines(overlay, [pts], False, (255, 255, 0), 2)
            disp = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imshow('Webcam Face+Hairline', disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# --- Entry point ---
if __name__ == "__main__":
    run_webcam()
