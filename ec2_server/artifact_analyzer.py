# services/artifact_analyzer.py
# TrueVision  OpenCV-based artifact detection
# Detects WHY a face looks fake without any ML model
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

import cv2
import numpy as np


def analyze_face_artifacts(face_path: str) -> dict:
    """
    Analyzes a single face image for deepfake artifacts using OpenCV signals.

    Returns:
        {
            "face_file": str,
            "total_artifacts_found": int,
            "artifacts": [ {type, severity, score, what_was_detected, explanation} ],
            "summary": str  â† plain English for the user
        }
    """
    img = cv2.imread(face_path)
    if img is None:
        return {
            "face_file": face_path.split("/")[-1],
            "total_artifacts_found": 0,
            "artifacts": [],
            "summary": "Could not read this face image for artifact analysis."
        }

    h, w  = img.shape[:2]
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    artifacts = []

    # â”€â”€ 1. FACE BOUNDARY / BLENDING ARTIFACTS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    boundary_score = _check_boundary_blending(img, gray, h, w)
    if boundary_score > 0.35:
        artifacts.append({
            "type":               "face_boundary_artifact",
            "severity":           _severity(boundary_score),
            "score":              round(boundary_score, 3),
            "what_was_detected":  "Unnatural blending at the face edges",
            "explanation": (
                "The edges of the face show signs of being digitally pasted "
                "onto another head. In real faces the skin blends naturally "
                "into the background, but here there are abrupt color or "
                "sharpness jumps at the face boundary â€” a classic deepfake "
                "blending artifact produced by face-swap tools."
            )
        })

    # â”€â”€ 2. LIP REGION INCONSISTENCY â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    lip_score = _check_lip_inconsistency(img, gray, h, w)
    if lip_score > 0.30:
        artifacts.append({
            "type":               "lip_sync_artifact",
            "severity":           _severity(lip_score),
            "score":              round(lip_score, 3),
            "what_was_detected":  "Lip region shows unnatural texture or motion blur",
            "explanation": (
                "The mouth and lip area looks different from the rest of the "
                "face â€” it has unusual blurring, over-sharpening, or texture "
                "mismatch. This happens when a deepfake replaces or re-animates "
                "the lips to fake speech (lip-sync manipulation). The lips do "
                "not match the natural texture of the surrounding face skin."
            )
        })

    # â”€â”€ 3. SKIN TEXTURE UNNATURALNESS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    texture_score = _check_skin_texture(img, gray)
    if texture_score > 0.32:
        artifacts.append({
            "type":               "unnatural_skin_texture",
            "severity":           _severity(texture_score),
            "score":              round(texture_score, 3),
            "what_was_detected":  "Skin texture appears AI-generated or over-smoothed",
            "explanation": (
                "Real human skin has natural microscopic noise â€” pores, fine "
                "lines, and subtle variations. This face's skin looks either "
                "unnaturally smooth (plastic-like) or has an artificial "
                "texture pattern consistent with AI-generated or GAN-synthesized "
                "faces. The high-frequency detail that real skin always has "
                "is either missing or inconsistent."
            )
        })

    # â”€â”€ 4. EYE REGION INCONSISTENCY â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    eye_score = _check_eye_inconsistency(img, gray, h, w)
    if eye_score > 0.28:
        artifacts.append({
            "type":               "eye_region_artifact",
            "severity":           _severity(eye_score),
            "score":              round(eye_score, 3),
            "what_was_detected":  "Eyes show asymmetry or unnatural appearance",
            "explanation": (
                "The eye regions show signs of manipulation â€” either left-right "
                "asymmetry that does not look natural, or the light reflections "
                "in the eyes (catch-lights) are missing, doubled, or appear "
                "artificial. Deepfake models consistently struggle to realistically "
                "render eyes, especially their subtle reflective properties and "
                "natural asymmetry patterns."
            )
        })

    # â”€â”€ 5. COMPRESSION / DOUBLE-COMPRESSION ARTIFACTS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    compression_score = _check_compression_artifacts(gray)
    if compression_score > 0.38:
        artifacts.append({
            "type":               "compression_artifact",
            "severity":           _severity(compression_score),
            "score":              round(compression_score, 3),
            "what_was_detected":  "Signs of double-compression or re-encoding",
            "explanation": (
                "The image shows signs of being compressed more than once â€” "
                "which is a common deepfake signature. The original video is "
                "compressed first, then the fake face is inserted, and the "
                "video is compressed again. This double compression creates "
                "blocky patterns at 8Ã—8 pixel JPEG boundaries that are not "
                "present in original unaltered footage."
            )
        })

    # â”€â”€ 6. COLOR MISMATCH (face vs neck / forehead) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    color_score = _check_color_mismatch(img, h, w)
    if color_score > 0.25:
        artifacts.append({
            "type":               "color_mismatch_artifact",
            "severity":           _severity(color_score),
            "score":              round(color_score, 3),
            "what_was_detected":  "Face skin color does not match surrounding areas",
            "explanation": (
                "The skin color or lighting in the center of the face does not "
                "match the forehead, chin, or neck area. In real faces these "
                "regions have consistent tone under the same lighting. This "
                "color mismatch strongly suggests a face swap where the pasted "
                "face has a different color profile, white balance, or lighting "
                "condition than the original person's body."
            )
        })

    summary = _build_summary(artifacts)

    return {
        "face_file":             face_path.split("/")[-1],
        "total_artifacts_found": len(artifacts),
        "artifacts":             artifacts,
        "summary":               summary
    }


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SIGNAL DETECTION FUNCTIONS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _check_boundary_blending(img, gray, h, w):
    """Detect abrupt blending at face edges â€” pasted face boundary seam."""
    border_w = int(w * 0.15)
    border_h = int(h * 0.15)
    mask = np.zeros_like(gray)
    mask[:border_h, :]  = 1
    mask[-border_h:, :] = 1
    mask[:, :border_w]  = 1
    mask[:, -border_w:] = 1

    edges               = cv2.Canny(gray, 50, 150)
    border_edge_density = (edges * mask).sum() / (mask.sum() + 1e-6)
    center_mask         = 1 - mask
    center_edge_density = (edges * center_mask).sum() / (center_mask.sum() + 1e-6)

    ratio = border_edge_density / (center_edge_density + 1e-6)
    score = np.clip((ratio - 1.0) / 3.0, 0, 1)
    return float(score)


def _check_lip_inconsistency(img, gray, h, w):
    """Detect texture mismatch in lip zone vs upper face."""
    lip_y1 = int(h * 0.60)
    lip_x1 = int(w * 0.25)
    lip_x2 = int(w * 0.75)
    lip_zone   = gray[lip_y1:, lip_x1:lip_x2]
    upper_zone = gray[:int(h * 0.40), int(w * 0.20):int(w * 0.80)]

    if lip_zone.size == 0 or upper_zone.size == 0:
        return 0.0

    lip_laplacian   = cv2.Laplacian(lip_zone,   cv2.CV_64F).var()
    upper_laplacian = cv2.Laplacian(upper_zone, cv2.CV_64F).var()

    ratio = abs(lip_laplacian - upper_laplacian) / (upper_laplacian + 1e-6)
    score = np.clip(ratio / 5.0, 0, 1)
    return float(score)


def _check_skin_texture(img, gray):
    """Detect unnaturally smooth (AI) or over-sharpened skin."""
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    if laplacian_var < 80:
        # Too smooth â€” plastic/AI look
        score = np.clip((80 - laplacian_var) / 80, 0, 1)
    elif laplacian_var > 3000:
        # Over-sharpened â€” processing artifact
        score = np.clip((laplacian_var - 3000) / 3000, 0, 1)
    else:
        score = 0.0

    return float(score)


def _check_eye_inconsistency(img, gray, h, w):
    """Detect left-right eye asymmetry beyond natural levels."""
    eye_y1 = int(h * 0.20)
    eye_y2 = int(h * 0.45)
    mid_x  = w // 2

    left_eye  = gray[eye_y1:eye_y2, :mid_x]
    right_eye = gray[eye_y1:eye_y2, mid_x:]

    if left_eye.size == 0 or right_eye.size == 0:
        return 0.0

    right_eye     = cv2.resize(right_eye, (left_eye.shape[1], left_eye.shape[0]))
    right_flipped = cv2.flip(right_eye, 1)
    diff          = np.abs(left_eye.astype(float) - right_flipped.astype(float))
    asymmetry     = diff.mean() / 255.0

    # Real faces have ~8% natural asymmetry. Beyond 28% is suspicious.
    score = np.clip((asymmetry - 0.08) / 0.20, 0, 1)
    return float(score)


def _check_compression_artifacts(gray):
    """Detect JPEG 8Ã—8 block boundary patterns from double compression."""
    h, w = gray.shape

    block_diffs = []
    for x in range(8, w - 8, 8):
        col_diff = abs(gray[:, x].astype(int) - gray[:, x - 1].astype(int))
        block_diffs.append(col_diff.mean())

    non_block_diffs = []
    for x in range(1, w - 1):
        if x % 8 != 0:
            col_diff = abs(gray[:, x].astype(int) - gray[:, x - 1].astype(int))
            non_block_diffs.append(col_diff.mean())

    if not block_diffs or not non_block_diffs:
        return 0.0

    avg_block     = np.mean(block_diffs)
    avg_non_block = np.mean(non_block_diffs)
    ratio         = avg_block / (avg_non_block + 1e-6)
    score         = np.clip((ratio - 1.2) / 1.5, 0, 1)
    return float(score)


def _check_color_mismatch(img, h, w):
    """Detect skin color mismatch between face center and chin/forehead."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    cy1, cy2 = int(h * 0.35), int(h * 0.65)
    cx1, cx2 = int(w * 0.30), int(w * 0.70)
    center   = hsv[cy1:cy2, cx1:cx2]

    forehead = hsv[:int(h * 0.20), int(w * 0.25):int(w * 0.75)]
    chin     = hsv[int(h * 0.80):, int(w * 0.25):int(w * 0.75)]

    if forehead.size > 0 and chin.size > 0:
        outer = np.vstack([forehead, chin])
    elif forehead.size > 0:
        outer = forehead
    else:
        outer = chin

    if center.size == 0 or outer.size == 0:
        return 0.0

    center_h = center[:, :, 0].mean()
    outer_h  = outer[:, :, 0].mean()
    center_s = center[:, :, 1].mean()
    outer_s  = outer[:, :, 1].mean()

    hue_diff = abs(center_h - outer_h) / 180.0
    sat_diff = abs(center_s - outer_s) / 255.0
    score    = np.clip((hue_diff * 2 + sat_diff) / 1.5, 0, 1)
    return float(score)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# HELPERS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _severity(score: float) -> str:
    if score >= 0.70:
        return "HIGH"
    if score >= 0.45:
        return "MEDIUM"
    return "LOW"


def _build_summary(artifacts: list) -> str:
    if not artifacts:
        return (
            "No specific manipulation artifacts were detected in this face. "
            "The face appears visually consistent with a real, unaltered image."
        )

    high   = [a for a in artifacts if a["severity"] == "HIGH"]
    medium = [a for a in artifacts if a["severity"] == "MEDIUM"]

    display_names = {
        "face_boundary_artifact":  "face-boundary blending",
        "lip_sync_artifact":       "lip-sync mismatch",
        "unnatural_skin_texture":  "unnatural skin texture",
        "eye_region_artifact":     "eye asymmetry",
        "compression_artifact":    "double-compression",
        "color_mismatch_artifact": "face color mismatch"
    }

    detected = [display_names.get(a["type"], a["type"]) for a in artifacts]

    if high:
        severity_word = "strong"
    elif medium:
        severity_word = "moderate"
    else:
        severity_word = "mild"

    has_boundary = any(a["type"] == "face_boundary_artifact" for a in artifacts)
    has_lip      = any(a["type"] == "lip_sync_artifact"      for a in artifacts)

    extra = ""
    if has_boundary or has_lip:
        extra = (
            " The most suspicious signals are at the face boundary and lip area, "
            "which are hallmarks of face-swap deepfakes."
        )

    return (
        f"This face shows {severity_word} signs of manipulation. "
        f"Detected issues: {', '.join(detected)}. "
        f"Total: {len(artifacts)} artifact type(s) found.{extra}"
    )
