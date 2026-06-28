# app.py
# TrueVision - Main FastAPI application
# Patched: confidence-flip bug fixed uncertainty-band veto; disputed-case handling.

from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os, shutil, cv2, time

from services.preprocessing import extract_frames
from services.face_detection import extract_faces
from services.artifact_analyzer import analyze_face_artifacts
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

# Inside this band the ensemble is considered genuinely uncertain; the
# artifact scanner is allowed to break the tie. Outside the band, the
# ensemble's prediction stands - but if it conflicts with HIGH-severity
# artifacts, we flag the result as 'uncertain' so the UI can warn.
UNCERTAIN_LOWER = 0.35
UNCERTAIN_UPPER = 0.50

ARTIFACT_DISPLAY_NAMES = {
    "face_boundary_artifact":  "Face-boundary blending",
    "lip_sync_artifact":       "Lip-sync mismatch",
    "unnatural_skin_texture":  "Unnatural skin texture",
    "eye_region_artifact":     "Eye region anomaly",
    "compression_artifact":    "Double-compression",
    "color_mismatch_artifact": "Face color mismatch",
}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAMES_FOLDER, exist_ok=True)
os.makedirs(FACES_FOLDER,  exist_ok=True)

app.mount("/frames", StaticFiles(directory=FRAMES_FOLDER), name="frames")
app.mount("/faces",  StaticFiles(directory=FACES_FOLDER),  name="faces")


# ============================================================
# VERDICT LOGIC
# ============================================================

def _collect_artifacts(face_details: List[Dict[str, Any]]):
    """Return (has_critical, unique_display_names).

    'Critical' now requires the artifact to appear on MULTIPLE faces,
    or to be a non-boundary type. Face-boundary alone on one face is
    too noisy on real videos (MTCNN cropping, motion blur all create it).
    """
    names: List[str] = []
    # Count how many faces each artifact type appears on
    type_face_count = {}
    non_boundary_high = False

    for face in face_details:
        types_this_face = set()
        for art in face.get("artifact_analysis", {}).get("artifacts", []):
            art_type = art.get("type", "")
            display_name = ARTIFACT_DISPLAY_NAMES.get(
                art_type,
                art.get("what_was_detected", art_type),
            )
            if display_name and display_name not in names:
                names.append(display_name)
            types_this_face.add(art_type)
            # Non-boundary HIGH artifacts are stronger evidence
            if (art.get("severity") == "HIGH" or art.get("score", 0) >= 0.95):
                if art_type != "face_boundary_artifact":
                    non_boundary_high = True
        for t in types_this_face:
            type_face_count[t] = type_face_count.get(t, 0) + 1

    # Critical only if: a non-boundary HIGH artifact exists, OR
    # face_boundary appears on 2+ faces AND at least one other artifact type exists
    boundary_faces = type_face_count.get("face_boundary_artifact", 0)
    other_types = sum(1 for t in type_face_count if t != "face_boundary_artifact")
    has_critical = non_boundary_high or (boundary_faces >= 2 and other_types >= 2)

    return has_critical, names

def generate_final_verdict(
    ensemble_prediction: str,
    ensemble_p_fake: float,
    face_details: List[Dict[str, Any]],
    file_type: str,
    model_votes: List[str],
) -> Dict[str, Any]:
    """
    Produce the final label + confidence.

    Decision rules:
      1. DEFAULT: trust the ensemble. confidence = P(reported_class).
      2. VETO: if ensemble is in [0.40, 0.60] AND a critical artifact exists,
         flip to FAKE with confidence = max(ensemble, 0.85).
      3. DISPUTED: if ensemble is confidently REAL (outside band) but
         critical artifacts exist, keep REAL but set uncertain=True and
         surface the artifacts so the UI can warn.

    This fixes the prior bug where flipping the label left confidence
    attached to the wrong class.
    """
    has_critical, why_fake_names = _collect_artifacts(face_details)
    file_str = "video" if file_type == "video" else "image"

    # Defensive clamp - upstream bugs could send values outside [0,1]
    p_fake = max(0.0, min(1.0, float(ensemble_p_fake)))
    ensemble_says_fake = p_fake >= 0.40
    in_uncertain_band  = UNCERTAIN_LOWER <= p_fake <= UNCERTAIN_UPPER

    # Strong disagreement: ensemble confidently says REAL but scanner screams FAKE
    ai_scanner_disputed = (
        has_critical and not ensemble_says_fake and not in_uncertain_band
    )

    real_votes = model_votes.count("REAL")
    fake_votes = model_votes.count("FAKE")
    total_votes = real_votes + fake_votes

    if in_uncertain_band and has_critical:
        p_fake_final = max(p_fake, 0.85)
        decided_by   = "artifact_veto"
        prediction   = "FAKE"
    else:
        p_fake_final = p_fake
        decided_by   = "ai_ensemble"
        prediction   = "FAKE" if ensemble_says_fake else "REAL"

    # Confidence is ALWAYS the probability of the reported class
    confidence     = p_fake_final if prediction == "FAKE" else (1.0 - p_fake_final)
    confidence_pct = round(confidence * 100.0, 1)

    # Build human-readable text
    if decided_by == "artifact_veto":
        example = why_fake_names[0] if why_fake_names else "unnatural blending"
        verdict_text = (
            f"This {file_str} is FAKE with {confidence_pct:.1f}% confidence. "
            f"The neural networks were uncertain, but the artifact scanner "
            f"detected a critical deepfake signature ({example})."
        )
    elif prediction == "FAKE":
        verdict_text = (
            f"This {file_str} is FAKE with {confidence_pct:.1f}% confidence. "
            f"The AI ensemble detected neural-network artifacts consistent with deepfake generation."
        )
    elif ai_scanner_disputed:
        signals = ", ".join(why_fake_names[:3])
        verdict_text = (
            f"This {file_str} appears REAL with {confidence_pct:.1f}% confidence, "
            f"BUT the artifact scanner detected strong deepfake signals ({signals}). "
            f"The AI ensemble overrode the scanner, but we recommend reviewing "
            f"this content carefully."
        )
    elif why_fake_names:
        majority_str = f"{real_votes}/{total_votes}" if total_votes else "majority"
        verdict_text = (
            f"This {file_str} appears REAL with {confidence_pct:.1f}% confidence. "
            f"The scanner flagged minor anomalies (likely shadows or compression), "
            f"but the ensemble ({majority_str} models) confirmed REAL."
        )
    else:
        verdict_text = (
            f"This {file_str} appears REAL with {confidence_pct:.1f}% confidence. "
            f"No deepfake artifacts or AI-generated patterns detected."
        )

    return {
        "prediction":     prediction,
        "confidence_pct": confidence_pct,
        "p_fake":         round(p_fake_final, 4),
        "decided_by":     decided_by,
        "uncertain":      (in_uncertain_band and not has_critical) or ai_scanner_disputed,
        "verdict_text":   verdict_text,
        "why_fake":       why_fake_names if (prediction == "FAKE" or ai_scanner_disputed) else [],
    }


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def home():
    return {"message": "TrueVision API Running"}


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
            "AI-generated video (Veo, Sora, Runway) - no face-swap artifacts",
            "Heavily compressed videos - artifacts destroyed",
            "Faces smaller than 40x40 pixels - too blurry",
            "Non-face manipulations (body, background only)"
        ],
        "final_logic": (
            "Weighted ensemble (CNN/CViT/ETCNN = 0.33/0.33/0.34) with "
            "uncertainty-band artifact veto and disputed-case flagging"
        )
    }


@app.post("/process/")
async def process_file(request: Request, file: UploadFile = File(...)):
    try:
        print(f"\n{'='*60}")
        print(f"=== NEW REQUEST: {file.filename} ===")
        print(f"{'='*60}")

        # Save uploaded file
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)

        file_type = "video" if file.filename.lower().endswith(
            (".mp4", ".avi", ".mov")) else "image"

        # Clean old data
        shutil.rmtree(FRAMES_FOLDER, ignore_errors=True)
        shutil.rmtree(FACES_FOLDER,  ignore_errors=True)
        os.makedirs(FRAMES_FOLDER, exist_ok=True)
        os.makedirs(FACES_FOLDER,  exist_ok=True)

        # Extract frames
        t0 = time.time()
        if file_type == "video":
            extract_frames(file_path, FRAMES_FOLDER)
        else:
            img = cv2.imread(file_path)
            if img is None:
                return {
                    "status": "error",
                    "message": f"Could not read image file: {file.filename}. "
                               "Make sure it is a valid .jpg, .png, or .webp file."
                }
            cv2.imwrite(os.path.join(FRAMES_FOLDER, "frame_0000.jpg"), img)

        print(f"[time] Frames: {time.time()-t0:.2f}s")
        frame_files = sorted(os.listdir(FRAMES_FOLDER))

        if not frame_files:
            return {
                "status":  "error",
                "message": "Could not extract any frames from the file."
            }

        # Detect faces
        t1 = time.time()
        faces = extract_faces(FRAMES_FOLDER, FACES_FOLDER)
        print(f"[time] Faces: {time.time()-t1:.2f}s")

        if not faces:
            return {
                "status":  "no_face",
                "file":    file.filename,
                "type":    file_type,
                "message": (
                    f"No human faces were detected in this "
                    f"{'video' if file_type == 'video' else 'image'}. "
                    "TrueVision is a deepfake detection tool designed specifically "
                    "for videos and images that contain human faces. "
                    "Please upload content with a clearly visible human face."
                ),
                "user_tip": (
                    "Make sure the face is: "
                    "(1) clearly visible and not blurred, "
                    "(2) at least 40x40 pixels in size, "
                    "(3) facing the camera - not a side or back view, "
                    "(4) not covered by a mask, sunglasses, or heavy shadow."
                ),
                "what_this_model_does": (
                    "This model detects deepfakes - digitally manipulated faces "
                    "where one person's face has been replaced with another using AI. "
                    "It cannot analyze videos without faces, AI-generated scenes "
                    "(like Sora or Runway), or non-face manipulations."
                )
            }

        # Run the three models
        t2 = time.time()
        model_results = run_all_models(faces)
        raw_final     = get_final_prediction(model_results)
        print(f"[time] Inference: {time.time()-t2:.2f}s")

        # Per-face artifact analysis
        face_details = []
        for i, face_path in enumerate(faces):
            face_filename   = os.path.basename(face_path)
            frame_name      = frame_files[i] if i < len(frame_files) else "unknown"
            artifact_result = analyze_face_artifacts(face_path)

            face_details.append({
                "face_index":        i + 1,
                "frame_number":      str(i),
                "frame_file":        frame_name,
                "face_file":         face_filename,
                "face_url":          f"http://{PUBLIC_IP}:8000/faces/{face_filename}",
                "frame_url":         f"http://{PUBLIC_IP}:8000/frames/{frame_name}",
                "artifact_analysis": artifact_result
            })

        # Build final verdict
        if raw_final.get("prediction") == "UNKNOWN":
            final_verdict_data = {
                "prediction":     "UNKNOWN",
                "confidence_pct": 0.0,
                "p_fake":         None,
                "decided_by":     "system",
                "uncertain":      True,
                "verdict_text":   "Unable to produce a prediction (no valid model outputs).",
                "why_fake":       [],
            }
        else:
            raw_p_fake = raw_final.get(
                "ensemble_sharp",
                raw_final.get("ensemble_raw", 0.5)
            )
            final_verdict_data = generate_final_verdict(
                ensemble_prediction=raw_final["prediction"],
                ensemble_p_fake=raw_p_fake,
                face_details=face_details,
                file_type=file_type,
                model_votes=raw_final.get("votes", []),
            )
        final_verdict_data["votes"] = raw_final.get("votes", [])

        # Auto-save for dataset building
        if final_verdict_data["prediction"] != "UNKNOWN":
            _auto_save(
                faces,
                final_verdict_data["prediction"],
                file.filename,
                final_verdict_data["confidence_pct"]
            )

        # Build URLs
        frame_urls = [
            f"http://{PUBLIC_IP}:8000/frames/{f}"
            for f in frame_files[:5]
        ]
        face_urls = [
            f"http://{PUBLIC_IP}:8000/faces/{os.path.basename(f)}"
            for f in faces[:5]
        ]

        return {
            "status": "success",
            "file":   file.filename,
            "type":   file_type,
            "frames": {"count": len(frame_files), "images": frame_urls},
            "faces":  {"count": len(faces),       "images": face_urls},
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
            "final_result": final_verdict_data,
            "face_details": face_details
        }

    except Exception as e:
        print(f"[ERROR]: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


# ============================================================
# FEEDBACK
# ============================================================
class FeedbackRequest(BaseModel):
    label: str  # "REAL" or "FAKE"


@app.post("/feedback/")
async def feedback(req: FeedbackRequest):
    try:
        label    = req.label.upper()
        faces    = _load_session()
        batch_id = _load_batch()

        fine_tune_etcnn(label, batch_id, epochs=3)
        _confirmed_save(faces, label, batch_id)

        return {
            "status":  "success",
            "message": "Thank you! Your feedback helps improve the model."
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ============================================================
# ADMIN
# ============================================================
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
        "CNN":         "EfficientNet-B0 x 3 datasets -> avg -> 1 vote",
        "CViT":        "ResNet50 + ViT x 3 datasets  -> avg -> 1 vote",
        "ETCNN":       "Dual-branch combined          -> 1 vote (online learner)",
        "final_logic": "Weighted ensemble (0.33/0.33/0.34) with uncertainty-band artifact veto + disputed-case flagging",
        "face_limit":  "Max 10 faces, 1 per frame (largest face only)",
        "frame_limit": "Max 30 frames, smart skip by video duration"
    }
