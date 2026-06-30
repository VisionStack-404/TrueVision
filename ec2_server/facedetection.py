import cv2
import os

# Load  the Haar Cascade safely
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    raise RuntimeError(f"âŒ Failed to load Haar Cascade from {cascade_path}")


def extract_faces(frames_folder, faces_folder="faces", max_faces=10):
    """
    Extracts faces from frames.
    - Detects one face per frame (largest face).
    - Stops after collecting max_faces.
    """

    os.makedirs(faces_folder, exist_ok=True)

    saved_faces = []
    frame_files = sorted(os.listdir(frames_folder))

    if not frame_files:
        print("âŒ No frames found in:", frames_folder)
        return []

    print(f"ðŸ“¦ Processing {len(frame_files)} frames (max {max_faces} faces)...")

    for frame_name in frame_files:
        if len(saved_faces) >= max_faces:
            print(f"âœ… Max faces ({max_faces}) reached â€” stopping early")
            break

        frame_path = os.path.join(frames_folder, frame_name)
        img = cv2.imread(frame_path)

        if img is None:
            print(f"âš ï¸ Skipping unreadable frame: {frame_name}")
            continue

        # Improve detection using histogram equalization
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # Detect faces with improved sensitivity
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,   # More sensitive
            minNeighbors=3,     # Balanced detection
            minSize=(20, 20),   # Detect smaller faces
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 0:
            print(f"âš ï¸ No face detected in {frame_name}")
            continue

        # Select the largest detected face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        # Add padding for better context
        pad = int(0.1 * w)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img.shape[1], x + w + pad)
        y2 = min(img.shape[0], y + h + pad)

        face = img[y1:y2, x1:x2]

        if face.size == 0:
            continue

        filename = f"face_{len(saved_faces):04d}.jpg"
        face_path = os.path.join(faces_folder, filename)
        cv2.imwrite(face_path, face)

        saved_faces.append(face_path)
        print(f"  {frame_name} â†’ face saved ({len(saved_faces)}/{max_faces})")

    print(f"âœ… Total faces detected: {len(saved_faces)}")
    return saved_faces
