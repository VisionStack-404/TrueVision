import cv2, os

def extract_frames(video_path, frames_folder, max_frames=30):
    """
    Smart frame skipping based on video duration.
    Stops at max_frames regardless of video length.
    """
    os.makedirs(frames_folder, exist_ok=True)

    cap          = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25
    duration_sec = total_frames / fps

    # Smart skip â€” longer video = skip more frames
    if duration_sec > 120:
        frame_skip = 30
    elif duration_sec > 60:
        frame_skip = 20
    else:
        frame_skip = 15

    print(f"ðŸ“¹ Video: {duration_sec:.1f}s | fps:{fps:.0f} "
          f"| skip:every {frame_skip} frames | max:{max_frames}")

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if saved_count >= max_frames:
            print(f"âœ… Max frames ({max_frames}) reached â€” stopping early")
            break

        if frame_count % frame_skip == 0:
            frame      = cv2.resize(frame, (640, 360))
            frame_path = os.path.join(
                frames_folder, f"frame_{saved_count:04d}.jpg"
            )
            cv2.imwrite(frame_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"âœ… Frames extracted: {saved_count}")
