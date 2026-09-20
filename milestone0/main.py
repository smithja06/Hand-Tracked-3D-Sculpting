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
        urlretrieve(
            MODEL_URL,  # Official internet address for the model file.
            MODEL_PATH,  # Local path where the download is saved.
        )

    # Open camera 0 (normally the built-in webcam).
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open camera 0")

    # Configure MediaPipe to follow up to two hands in a video stream.
    options = mp.tasks.vision.HandLandmarkerOptions(
        # BaseOptions tells MediaPipe where its downloaded model lives.
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        # VIDEO reuses tracking information between consecutive webcam frames.
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        # Detect at most two hands in each frame.
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
                frame = cv2.flip(
                    frame,  # Source webcam frame.
                    1,  # Flip code: 1 means horizontal mirroring.
                )
                # OpenCV uses BGR; MediaPipe expects RGB images.
                image = mp.Image(
                    # Declare the colour order of the image data below.
                    image_format=mp.ImageFormat.SRGB,
                    # Convert OpenCV's BGR frame into RGB pixels.
                    data=cv2.cvtColor(
                        frame,  # OpenCV frame to convert.
                        cv2.COLOR_BGR2RGB,  # Requested BGR-to-RGB conversion.
                    ),
                )
                # Find the 21 normalized landmarks for each visible hand.
                result = hands.detect_for_video(
                    image,  # RGB MediaPipe image for this frame.
                    int(time.monotonic() * 1000),  # Increasing timestamp in milliseconds.
                )
                height, width = frame.shape[:2]
                for hand in result.hand_landmarks:
                    # Turn 0-to-1 landmark coordinates into screen pixels.
                    points = [(int(point.x * width), int(point.y * height)) for point in hand]
                    # Draw the finger and palm bones, then the landmark dots.
                    for connection in mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS:
                        cv2.line(
                            frame,  # Image to draw on.
                            points[connection.start],  # Start landmark pixel.
                            points[connection.end],  # End landmark pixel.
                            (0, 255, 0),  # Green line colour in OpenCV's BGR order.
                            2,  # Line thickness in pixels.
                        )
                    for point in points:
                        cv2.circle(
                            frame,  # Image to draw on.
                            point,  # Landmark centre pixel.
                            3,  # Circle radius in pixels.
                            (0, 0, 255),  # Red dot colour in OpenCV's BGR order.
                            -1,  # Negative thickness fills the circle.
                        )
                # Show the annotated frame; press q to stop.
                cv2.imshow(
                    "MediaPipe Hands",  # Preview-window title.
                    frame,  # Image to show in that window.
                )
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        # Always give the camera and preview window back to the operating system.
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
