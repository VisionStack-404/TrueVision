import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import load_model

# ✅ Load model once
model = load_model("models/cnn_model.h5")

# ✅ Initialize face detector
detector = MTCNN()


# 🔥 STEP 1 — Extract frames (limited for performance)
def extract_frames(video_path, max_frames=20):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames


# 🔥 STEP 2 — Detect face
def get_face(frame):
    results = detector.detect_faces(frame)

    for res in results:
        x, y, w, h = res['box']

        # Fix negative values
        x, y = max(0, x), max(0, y)

        face = frame[y:y+h, x:x+w]

        if face.size == 0:
            continue

        face = cv2.resize(face, (128, 128))
        return face

    return None


# 🔥 STEP 3 — Predict single face
def predict_face(face):
    face = face / 255.0
    face = np.expand_dims(face, axis=0)

    prediction = model.predict(face, verbose=0)[0][0]
    return float(prediction)


# 🔥 STEP 4 — MAIN DETECTION FUNCTION
def detect_deepfake(video_path):
    frames = extract_frames(video_path)

    if len(frames) == 0:
        return {
            "prediction": "Error",
            "confidence": 0,
            "frames_analyzed": 0
        }

    predictions = []

    for frame in frames:
        face = get_face(frame)

        if face is not None:
            pred = predict_face(face)
            predictions.append(pred)

    if len(predictions) == 0:
        return {
            "prediction": "No Face Detected",
            "confidence": 0,
            "frames_analyzed": len(frames)
        }

    avg_pred = sum(predictions) / len(predictions)

    result = "Fake" if avg_pred > 0.5 else "Real"

    return {
        "prediction": result,
        "confidence": round(avg_pred * 100, 2),
        "frames_analyzed": len(predictions)
    }