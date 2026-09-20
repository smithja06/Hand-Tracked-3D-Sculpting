from pathlib import Path
import time
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np
import pyrender
import trimesh


MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
MODEL_PATH = Path(__file__).parent.parent / "hand_landmarker.task"
CAMERA_INDEX = 1
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
PREVIEW_SCALE = 0.75


def fingertip_pose(landmark, width, height):
    """Place a 3D object on a fixed plane directly in front of the camera."""
    focal_length = width
    depth = 1.0
    pose = np.eye(4)
    pose[0, 3] = (landmark.x * width - width / 2) * depth / focal_length
    pose[1, 3] = -(landmark.y * height - height / 2) * depth / focal_length
    pose[2, 3] = -depth
    return pose


def make_renderer(width, height):
    """Create the transparent 3D scene once, then move its mesh nodes each frame."""
    scene = pyrender.Scene(bg_color=(0, 0, 0, 0), ambient_light=(0.25, 0.25, 0.25))
    camera = pyrender.IntrinsicsCamera(width, width, width / 2, height / 2)
    scene.add(camera)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), pose=np.eye(4))

    cube = trimesh.creation.box(extents=(0.12, 0.12, 0.12))
    cube.visual.face_colors = (0, 160, 255, 255)
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.07)
    sphere.visual.face_colors = (255, 80, 0, 255)
    cube_node = scene.add(pyrender.Mesh.from_trimesh(cube, smooth=False))
    sphere_node = scene.add(pyrender.Mesh.from_trimesh(sphere, smooth=False))
    return scene, cube_node, sphere_node, pyrender.OffscreenRenderer(width, height)


def main():
    if not MODEL_PATH.exists():
        urlretrieve(MODEL_URL, MODEL_PATH)

    camera = cv2.VideoCapture(CAMERA_INDEX)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera {CAMERA_INDEX}")

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
    )
    renderer = None
    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as hands:
            while True:
                ok, frame = camera.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                if renderer is None:
                    scene, cube_node, sphere_node, renderer = make_renderer(width, height)

                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                )
                result = hands.detect_for_video(image, int(time.monotonic() * 1000))

                show_cube = show_sphere = False
                for hand in result.hand_landmarks:
                    scene.set_pose(cube_node, fingertip_pose(hand[8], width, height))
                    scene.set_pose(sphere_node, fingertip_pose(hand[12], width, height))
                    show_cube = show_sphere = True
                cube_node.mesh.is_visible = show_cube
                sphere_node.mesh.is_visible = show_sphere

                rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
                alpha = rgba[:, :, 3:4] / 255.0
                frame = (rgba[:, :, :3][:, :, ::-1] * alpha + frame * (1 - alpha)).astype(np.uint8)
                preview = cv2.resize(frame, None, fx=PREVIEW_SCALE, fy=PREVIEW_SCALE, interpolation=cv2.INTER_AREA)
                cv2.imshow("MediaPipe Finger AR", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        if renderer:
            renderer.delete()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
