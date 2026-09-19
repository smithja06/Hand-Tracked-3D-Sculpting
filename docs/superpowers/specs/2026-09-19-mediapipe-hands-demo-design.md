# MediaPipe Hands Demo

## Goal

Provide a minimal local Python demo that shows webcam frames with MediaPipe hand landmarks.

## Design

- `milestone0/main.py` opens the default camera with OpenCV.
- Each frame is mirrored, converted to RGB, and processed by MediaPipe Hands.
- The window draws landmarks and hand connections for up to two hands.
- Pressing `q` closes the window and releases the camera.
- If camera 0 cannot open, the program raises a clear error.
- `requirements.txt` contains only `mediapipe` and `opencv-python`.

## Validation

- A small test verifies the file exposes a runnable entry point without opening a camera.
- Manual run: `python milestone0/main.py`; confirm landmarks follow a hand and `q` exits.

## Out of Scope

No 3D mesh editing, gesture recognition, recording, UI controls, or model downloads beyond package installation.
