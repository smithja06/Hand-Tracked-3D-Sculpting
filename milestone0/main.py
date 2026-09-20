from pathlib import Path
import time
from urllib.request import urlretrieve

import cv2  # OpenCV's Python module (installed by opencv-python).
import mediapipe as mp


# MediaPipe's pre-trained hand-landmark model, downloaded once on first run.
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")


def main():
    # The model is separate from the Python package.
    if not MODEL_PATH.exists():
        urlretrieve(MODEL_URL, MODEL_PATH)

    # Open camera 0 (normally the built-in webcam).
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open camera 0")

    # Configure MediaPipe to follow up to two hands in a video stream.
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
    )
    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as hands:
            while True:
                # Read one image (a frame) from the webcam.
                ok, frame = camera.read()
                if not ok:
                    break
                # Mirror the preview so it behaves like a selfie camera.
                frame = cv2.flip(frame, 1)
                # OpenCV uses BGR; MediaPipe expects RGB images.
                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                )
                # Find the 21 normalized landmarks for each visible hand.
                result = hands.detect_for_video(image, int(time.monotonic() * 1000))
                height, width = frame.shape[:2]
                for hand in result.hand_landmarks:
                    # Turn 0-to-1 landmark coordinates into screen pixels.
                    points = [(int(point.x * width), int(point.y * height)) for point in hand]
                    # Draw the finger and palm bones, then the landmark dots.
                    for connection in mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS:
                        cv2.line(frame, points[connection.start], points[connection.end], (0, 255, 0), 2)
                    for point in points:
                        cv2.circle(frame, point, 3, (0, 0, 255), -1)
                # Show the annotated frame; press q to stop.
                cv2.imshow("MediaPipe Hands", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        # Always give the camera and preview window back to the operating system.
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
