from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os, shutil, cv2, time

from services.preprocessing import extract_frames
from services.face_detection import extract_faces
from services.backend.inference import (
    run_all_models, get_final_prediction,
    fine_tune_etcnn, get_feedback_history,
    _load_session, _load_batch,
    _auto_save, _confirmed_save, _retrain_status
)

PUBLIC_IP     = "3.238.89.41"
UPLOAD_FOLDER = "uploads"
FRAMES_FOLDER = "frames"
FACES_FOLDER  = "faces"

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAMES_FOLDER, exist_ok=True)
os.makedirs(FACES_FOLDER,  exist_ok=True)

app.mount("/frames", StaticFiles(directory=FRAMES_FOLDER), name="frames")
app.mount("/faces",  StaticFiles(directory=FACES_FOLDER),  name="faces")


@app.get("/")
def home():
    return {"message": "TrueVision API Running 🚀"}

@app.get("/detection/capabilities/")
def detection_capabilities():
    return {
        "supported_inputs": {
            "images": [".jpg", ".jpeg", ".png", ".webp"],
            "videos": [".mp4", ".avi", ".mov"]
        },
        "what_each_model_detects": {
            "CNN": {
                "looks_at":   "Raw face pixels",
                "detects":    "Face-swap artifacts, GAN blending boundaries",
                "trained_on": ["FaceForensics++", "DFDC", "CelebDF"],
                "best_for":   ["DeepFaceLab swaps", "FaceSwap", "DFDC videos"]
            },
            "CViT": {
                "looks_at":   "Edge-enhanced face (Laplacian filter)",
                "detects":    "Structural inconsistencies, boundary artifacts",
                "trained_on": ["FaceForensics++", "DFDC", "CelebDF"],
                "best_for":   ["Neural face replacement", "Attribute editing"]
            },
            "ETCNN": {
                "looks_at":   "High-frequency texture map",
                "detects":    "Unnatural skin texture, AI-generation patterns",
                "trained_on": ["FaceForensics++", "DFDC", "CelebDF combined"],
                "best_for":   ["All deepfake types", "AI-generated images"],
                "online_learning": True
            }
        },
        "what_it_cannot_detect": [
            "AI-generated video (Veo, Sora, Runway) — no face-swap artifacts",
            "Heavily compressed videos — artifacts destroyed",
            "Faces smaller than 40×40 pixels — too blurry",
            "Non-face manipulations (body, background only)"
        ],
        "final_logic": "Weighted ensemble: CNN×0.33 + CViT×0.33 + ETCNN×0.34"
    }


# ================================================
# MAIN ENDPOINT
# ================================================
@app.post("/process/")
async def process_file(request: Request, file: UploadFile = File(...)):
    try:
        print(f"=== NEW REQUEST: {file.filename} ===")

        # ── Save uploaded file ───────────────────────
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)

        file_type = "video" if file.filename.lower().endswith(
                        (".mp4", ".avi", ".mov")) else "image"

        # ── Clean old data ───────────────────────────
        shutil.rmtree(FRAMES_FOLDER, ignore_errors=True)
        shutil.rmtree(FACES_FOLDER,  ignore_errors=True)
        os.makedirs(FRAMES_FOLDER, exist_ok=True)
        os.makedirs(FACES_FOLDER,  exist_ok=True)

        # ── Extract frames (smart skip + max 30) ─────
        t0 = time.time()
        if file_type == "video":
            extract_frames(file_path, FRAMES_FOLDER)
        else:
            img = cv2.imread(file_path)
            cv2.imwrite(os.path.join(FRAMES_FOLDER, "frame_0000.jpg"), img)
        print(f"⏱ Frames: {time.time()-t0:.2f}s")

        frame_files = sorted(os.listdir(FRAMES_FOLDER))

        # ── Detect faces (1 per frame, max 10) ───────
        t1    = time.time()
        faces = extract_faces(FRAMES_FOLDER, FACES_FOLDER)
        print(f"⏱ Faces: {time.time()-t1:.2f}s")

        if not faces:
            return {
                "status":  "no_face",
                "file":    file.filename,
                "type":    file_type,
                "message": (
                    "No human faces were detected in this "
                    + ("video" if file_type == "video" else "image") + ". "
                    "TrueVision is a deepfake detection tool designed specifically "
                    "for videos and images that contain human faces. "
                    "Please upload content with a clearly visible human face."
                ),
                "user_tip": (
                    "Make sure the face is: (1) clearly visible and not blurred, "
                    "(2) at least 40×40 pixels in size, "
                    "(3) facing the camera (not a side/back view), "
                    "(4) not covered by a mask, sunglasses, or heavy shadow."
                ),
                "what_this_model_does": (
                    "This model detects deepfakes — digitally manipulated faces "
                    "where one person's face has been replaced with another using AI. "
                    "It cannot analyze videos without faces, AI-generated scenes "
                    "(like Sora or Runway), or non-face manipulations."
                )
            }

        # ── Run CNN → CViT → ETCNN ───────────────────
        t2            = time.time()
        model_results = run_all_models(faces)
        final         = get_final_prediction(model_results)
        print(f"⏱ Inference: {time.time()-t2:.2f}s")

        # ── Background: auto-save dataset (silent) ───
        _auto_save(
            faces, final["prediction"],
            file.filename, final["confidence_pct"]
        )

        # ── Build URLs ───────────────────────────────
        frame_urls = [f"http://{PUBLIC_IP}:8000/frames/{f}"
                      for f in frame_files[:5]]
        face_urls  = [f"http://{PUBLIC_IP}:8000/faces/{os.path.basename(f)}"
                      for f in faces[:5]]

        # ── Response — clean, no memory info ─────────
        return {
            "status": "success",
            "file":   file.filename,
            "type":   file_type,

            "frames": {"count": len(frame_files), "images": frame_urls},
            "faces":  {"count": len(faces),       "images": face_urls},

            # Individual model results
            "model_results": [
                {
                    "model":          r["model"],
                    "vote":           r["vote"],
                    "confidence_pct": r["confidence_pct"],
                    "p_fake":         r["p_fake"],
                    "p_real":         r["p_real"],
                    "dataset_scores": r.get("dataset_scores", {})
                }
                for r in model_results
            ],

            # Final answer — highest confidence wins
            "final_result": {
                "prediction":     final["prediction"],
                "confidence_pct": final["confidence_pct"],
                "decided_by":     final["decided_by"],
                "votes":          final["votes"]
            }
        }

    except Exception as e:
        print("❌ ERROR:", str(e))
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


# ================================================
# FEEDBACK — user confirms or corrects
# All learning silent in background
# ================================================
class FeedbackRequest(BaseModel):
    label: str    # "REAL" or "FAKE" — only thing user sends

@app.post("/feedback/")
async def feedback(req: FeedbackRequest):
    try:
        label    = req.label.upper()
        faces    = _load_session()
        batch_id = _load_batch()

        # Background — user never sees these
        fine_tune_etcnn(label, batch_id, epochs=3)
        _confirmed_save(faces, label, batch_id)

        return {
            "status":  "success",
            "message": "Thank you! Your feedback helps improve the model."
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


# ================================================
# ADMIN ENDPOINTS — not shown to regular users
# ================================================
@app.get("/admin/retrain-status/")
def retrain_status():
    return _retrain_status()

@app.get("/admin/feedback-log/")
def feedback_log():
    history = get_feedback_history()
    return {"total_events": len(history), "log": history}

@app.get("/model/status/")
def model_status():
    return {
        "CNN":        "EfficientNet-B0 × 3 datasets → avg → 1 vote",
        "CViT":       "ResNet50 + ViT × 3 datasets  → avg → 1 vote",
        "ETCNN":      "Dual-branch combined          → 1 vote (online learner)",
        "final_logic":"Equal ensemble: CNN×33% + CViT×33% + ETCNN×34%",
        "face_limit": "Max 10 faces, 1 per frame (largest face only)",
        "frame_limit": "Max 30 frames, smart skip by video duration"
    }
