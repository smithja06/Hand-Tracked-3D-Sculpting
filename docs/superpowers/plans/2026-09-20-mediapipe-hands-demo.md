# MediaPipe Hands Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local webcam window that overlays MediaPipe hand landmarks.

**Architecture:** One executable module owns camera setup, frame processing, landmark drawing, and cleanup. A dependency-free structural test confirms the module keeps a runnable `main()` entry point without opening a camera.

**Tech Stack:** Python, OpenCV, MediaPipe

**Implementation adjustment:** MediaPipe 1.0.1 for Windows exposes the Tasks API rather than `mp.solutions`. The demo therefore uses `mp.tasks.vision.HandLandmarker` and downloads its official `hand_landmarker.task` model on first run; this replaces the legacy Solutions code in Task 1, Step 3.

## Global Constraints

- Use only `mediapipe` and `opencv-python` as runtime dependencies.
- Use camera index `0`, mirror displayed frames, process at most two hands, and exit on `q`.
- Raise `RuntimeError("Could not open camera 0")` when the camera cannot open.

---

### Task 1: Runnable landmark demo

**Files:**
- Create: `milestone0/test_main.py`
- Create: `milestone0/main.py`
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: OpenCV camera frames and `mediapipe.solutions.hands.Hands` results.
- Produces: `main() -> None`, invoked by `if __name__ == "__main__": main()`.

- [ ] **Step 1: Write the failing entry-point test**

Create `milestone0/test_main.py`:

```python
from pathlib import Path


source = Path(__file__).with_name("main.py")
assert source.exists(), "main.py is missing"
assert "def main()" in source.read_text(encoding="utf-8")
assert 'if __name__ == "__main__":' in source.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python milestone0/test_main.py`

Expected: `AssertionError: main.py is missing`

- [ ] **Step 3: Write the minimal demo and dependency list**

Create `milestone0/main.py`:

```python
import cv2
import mediapipe as mp


def main():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open camera 0")

    hands = mp.solutions.hands.Hands(max_num_hands=2)
    drawing = mp.solutions.drawing_utils
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if result.multi_hand_landmarks:
                for hand in result.multi_hand_landmarks:
                    drawing.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
            cv2.imshow("MediaPipe Hands", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        hands.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

Replace `requirements.txt` with:

```text
mediapipe
opencv-python
```

- [ ] **Step 4: Run the structural test to verify it passes**

Run: `python milestone0/test_main.py`

Expected: exits with status `0`.

- [ ] **Step 5: Install and manually validate the demo**

Run: `python -m pip install -r requirements.txt`

Then run: `python milestone0/main.py`

Expected: a `MediaPipe Hands` camera window opens, up to two hands have 21-point landmark overlays, and `q` exits cleanly.

- [ ] **Step 6: Commit**

```bash
git add milestone0/main.py milestone0/test_main.py requirements.txt
git commit -m "feat: add mediapipe hands demo"
```
