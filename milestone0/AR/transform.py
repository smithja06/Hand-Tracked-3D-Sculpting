from pathlib import Path
import time
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np
import pyrender
import trimesh


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

MODEL_PATH = Path(__file__).parent.parent / "hand_landmarker.task"

CAMERA_INDEX = 0
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
PREVIEW_SCALE = 0.75

# Transform tuning
MIN_SCALE = 0.5
MAX_SCALE = 2.0

MIN_PINCH_DISTANCE = 0.03
MAX_PINCH_DISTANCE = 0.25


def landmark_distance(a, b):
    """Euclidean distance between two MediaPipe landmarks."""
    return np.sqrt(
        (a.x - b.x) ** 2
        + (a.y - b.y) ** 2
        + (a.z - b.z) ** 2
    )


def get_scale(hand):
    """
    Map thumb-index distance to object scale.

    Thumb tip = landmark 4
    Index tip = landmark 8
    """
    pinch_distance = landmark_distance(hand[4], hand[8])

    scale = np.interp(
        pinch_distance,
        [MIN_PINCH_DISTANCE, MAX_PINCH_DISTANCE],
        [MIN_SCALE, MAX_SCALE],
    )

    return np.clip(scale, MIN_SCALE, MAX_SCALE)


def get_hand_rotation(hand, width, height):
    """
    Estimate rotation around the camera's Z axis using the palm orientation.

    Index MCP = landmark 5
    Pinky MCP = landmark 17
    """
    index_mcp = hand[5]
    pinky_mcp = hand[17]

    dx = (pinky_mcp.x - index_mcp.x) * width

    # Image coordinates point downward, while our 3D y-axis points upward.
    dy = -(pinky_mcp.y - index_mcp.y) * height

    angle = np.arctan2(dy, dx)

    return angle


def fingertip_translation(landmark, width, height):
    """
    Convert a MediaPipe image landmark to an approximate 3D position.

    For now the object stays on a fixed Z plane.
    """
    focal_length = width
    depth = 1.0

    x = (
        (landmark.x * width - width / 2)
        * depth
        / focal_length
    )

    y = -(
        (landmark.y * height - height / 2)
        * depth
        / focal_length
    )

    z = -depth

    return np.array([x, y, z])


def make_transform(position, scale=1.0, rotation_z=0.0):
    """
    Create a 4x4 transformation matrix containing:

    translation
    rotation
    uniform scaling
    """

    # Translation matrix
    T = np.eye(4)
    T[:3, 3] = position

    # Rotation around Z axis
    cos_a = np.cos(rotation_z)
    sin_a = np.sin(rotation_z)

    R = np.array(
        [
            [cos_a, -sin_a, 0, 0],
            [sin_a, cos_a, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    # Uniform scale matrix
    S = np.diag([scale, scale, scale, 1.0])

    # Apply scale, then rotation, then translation.
    return T @ R @ S


def object_pose(hand, landmark_index, width, height):
    """
    Build the object's full transformation from the detected hand.
    """

    position = fingertip_translation(
        hand[landmark_index],
        width,
        height,
    )

    scale = get_scale(hand)

    rotation = get_hand_rotation(
        hand,
        width,
        height,
    )

    return make_transform(
        position=position,
        scale=scale,
        rotation_z=rotation,
    )


def make_renderer(width, height):
    """
    Create the transparent 3D scene once,
    then transform mesh nodes each frame.
    """

    scene = pyrender.Scene(
        bg_color=(0, 0, 0, 0),
        ambient_light=(0.25, 0.25, 0.25),
    )

    camera = pyrender.IntrinsicsCamera(
        width,
        width,
        width / 2,
        height / 2,
    )

    scene.add(camera)

    scene.add(
        pyrender.DirectionalLight(
            color=np.ones(3),
            intensity=3.0,
        ),
        pose=np.eye(4),
    )

    cube = trimesh.creation.box(
        extents=(0.12, 0.12, 0.12)
    )

    cube.visual.face_colors = (
        0,
        160,
        255,
        255,
    )

    sphere = trimesh.creation.icosphere(
        subdivisions=2,
        radius=0.07,
    )

    sphere.visual.face_colors = (
        255,
        80,
        0,
        255,
    )

    cube_node = scene.add(
        pyrender.Mesh.from_trimesh(
            cube,
            smooth=False,
        )
    )

    sphere_node = scene.add(
        pyrender.Mesh.from_trimesh(
            sphere,
            smooth=False,
        )
    )

    renderer = pyrender.OffscreenRenderer(
        width,
        height,
    )

    return (
        scene,
        cube_node,
        sphere_node,
        renderer,
    )


def draw_debug_info(frame, hand):
    """Display the current transformation values."""

    scale = get_scale(hand)

    rotation = np.degrees(
        get_hand_rotation(
            hand,
            frame.shape[1],
            frame.shape[0],
        )
    )

    cv2.putText(
        frame,
        f"Scale: {scale:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"Rotation: {rotation:.1f} deg",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )


def main():

    if not MODEL_PATH.exists():
        urlretrieve(
            MODEL_URL,
            MODEL_PATH,
        )

    camera = cv2.VideoCapture(
        CAMERA_INDEX
    )

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAPTURE_WIDTH,
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAPTURE_HEIGHT,
    )

    if not camera.isOpened():
        raise RuntimeError(
            f"Could not open camera {CAMERA_INDEX}"
        )

    options = (
        mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(MODEL_PATH)
            ),
            running_mode=(
                mp.tasks.vision.RunningMode.VIDEO
            ),
            num_hands=2,
        )
    )

    renderer = None

    try:

        with (
            mp.tasks.vision.HandLandmarker
            .create_from_options(options)
        ) as hands:

            while True:

                ok, frame = camera.read()

                if not ok:
                    break

                # Mirror webcam view
                frame = cv2.flip(frame, 1)

                height, width = frame.shape[:2]

                if renderer is None:
                    (
                        scene,
                        cube_node,
                        sphere_node,
                        renderer,
                    ) = make_renderer(
                        width,
                        height,
                    )

                image = mp.Image(
                    image_format=(
                        mp.ImageFormat.SRGB
                    ),
                    data=cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB,
                    ),
                )

                result = hands.detect_for_video(
                    image,
                    int(
                        time.monotonic()
                        * 1000
                    ),
                )

                cube_node.mesh.is_visible = False
                sphere_node.mesh.is_visible = False

                if result.hand_landmarks:

                    # For now, use the first detected hand
                    hand = result.hand_landmarks[0]

                    # Cube follows index fingertip
                    cube_pose = object_pose(
                        hand,
                        landmark_index=8,
                        width=width,
                        height=height,
                    )

                    # Sphere follows middle fingertip
                    sphere_pose = object_pose(
                        hand,
                        landmark_index=12,
                        width=width,
                        height=height,
                    )

                    scene.set_pose(
                        cube_node,
                        cube_pose,
                    )

                    scene.set_pose(
                        sphere_node,
                        sphere_pose,
                    )

                    cube_node.mesh.is_visible = True
                    sphere_node.mesh.is_visible = True

                    draw_debug_info(
                        frame,
                        hand,
                    )

                rgba, _ = renderer.render(
                    scene,
                    flags=(
                        pyrender.RenderFlags.RGBA
                    ),
                )

                alpha = (
                    rgba[:, :, 3:4]
                    / 255.0
                )

                frame = (
                    rgba[:, :, :3][:, :, ::-1]
                    * alpha
                    + frame
                    * (1 - alpha)
                ).astype(np.uint8)

                preview = cv2.resize(
                    frame,
                    None,
                    fx=PREVIEW_SCALE,
                    fy=PREVIEW_SCALE,
                    interpolation=cv2.INTER_AREA,
                )

                cv2.imshow(
                    "MediaPipe Finger AR",
                    preview,
                )

                if (
                    cv2.waitKey(1)
                    & 0xFF
                    == ord("q")
                ):
                    break

    finally:

        if renderer:
            renderer.delete()

        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()