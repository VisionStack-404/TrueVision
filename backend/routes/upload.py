from fastapi import APIRouter, UploadFile, File
import shutil
import os
from services.preprocessing import extract_frames   # 👈 IMPORTANT

router = APIRouter()

UPLOAD_FOLDER = "uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@router.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    # Save video
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 🔥 CALL YOUR OPENCV FUNCTION
    frames = extract_frames(file_path)

    return {
        "message": "Video processed",
        "frames_extracted": frames
    }