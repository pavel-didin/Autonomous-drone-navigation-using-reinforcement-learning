import os
import cv2
import re
from natsort import natsorted

# ===== SETTINGS =====
IMAGE_FOLDER = "results/recording_03.26.2026_23.38.44"   # folder with frames
OUTPUT_VIDEO = "flight.mp4"                               # name of the output video
FPS = 4                                                  # frame rate
# =====================

def create_video_from_frames(output_video, image_folder, fps=30):
    """
    Creates a video from PNG frames in the specified folder.
    """
    images = [f for f in os.listdir(image_folder) if f.endswith('.png') and re.match(r'frame_\d+\.png', f)]
    if not images:
        print("No frames found in the folder frame_*.png")
        return

    images = natsorted(images)
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        print("The first frame could not be read.")
        return

    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for img_name in images:
        img_path = os.path.join(image_folder, img_name)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"A corrupted frame was skipped: {img_name}")
            continue
        out.write(frame)

    out.release()
    print(f"Video saved: {output_video}")

if __name__ == "__main__":
    create_video_from_frames(OUTPUT_VIDEO, IMAGE_FOLDER, FPS)
