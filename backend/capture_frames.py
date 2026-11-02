import cv2
import os
import sys

# This extracts the frames from a video and adds it to a directory
def extract_frames(video, out_dir, every=2):
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(video))[0]
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {video}")
    
    
    i = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if i % every == 0:
            cv2.imwrite(f"{out_dir}/{name}_{i:05d}.jpg", frame)
            saved += 1
        i+=1

    cap.release()
    print(f"Read {i} frames, saved {saved}")


path = "./extracted_frames"
video = sys.argv[1]
extract_frames(f"{video}.mp4", path)