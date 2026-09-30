from pathlib import Path
import threading

import av
import cv2
import mediapipe as mp
import numpy as np
from streamlit_webrtc import VideoProcessorBase
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from detectors.squat import SquatDetector
from detectors.pushup import PushUpDetector
from detectors.biceps_curl import BicepsCurlDetector
from detectors.shoulder_press import ShoulderPressDetector
from detectors.lunges import LungesDetector
from services.config.workout_config import POSE_CONNECTIONS


class VideoProcessorClass(VideoProcessorBase):

    def __init__(self):
        self._lock = threading.Lock()
        self._latest_metrics = None
        self._exercise_type = "Squats"

        # --------------------------------------------------
        # CORRECT MODEL PATH
        # File is:
        # Main App/ml_models/pose_landmarker_full.task
        # --------------------------------------------------

        app_dir = Path(__file__).resolve().parents[2]

        model_path = (
            app_dir
            / "ml_models"
            / "pose_landmarker_full.task"
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe model not found at: {model_path}"
            )

        # --------------------------------------------------
        # MEDIAPIPE POSE LANDMARKER
        # --------------------------------------------------

        base_option = python.BaseOptions(
            model_asset_path=str(model_path)
        )

        options = vision.PoseLandmarkerOptions(
            base_options=base_option,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.7,
            min_pose_presence_confidence=0.7,
            min_tracking_confidence=0.7,
            output_segmentation_masks=False,
        )

        self._landmarker = (
            vision.PoseLandmarker.create_from_options(options)
        )

        # --------------------------------------------------
        # EXERCISE DETECTORS
        # --------------------------------------------------

        self._detectors = {
            "Squats": SquatDetector(),
            "Push-ups": PushUpDetector(),
            "Biceps Curls (Dumbbell)": BicepsCurlDetector(),
            "Shoulder Press": ShoulderPressDetector(),
            "Lunges": LungesDetector(),
        }

        # MediaPipe VIDEO mode requires increasing timestamps
        self._frame_timestamps_ms = 0

    # ======================================================
    # METRICS
    # ======================================================

    def set_latest_metrics(self, metrics):
        with self._lock:
            self._latest_metrics = metrics.copy()

    def get_latest_metrics(self):
        with self._lock:
            if self._latest_metrics is None:
                return None

            return self._latest_metrics.copy()

    # ======================================================
    # EXERCISE
    # ======================================================

    def set_exercise(self, exercise_type):
        with self._lock:
            self._exercise_type = exercise_type

    def get_exercise(self):
        with self._lock:
            return self._exercise_type

    # ======================================================
    # DRAW SKELETON
    # ======================================================

    def _draw_skeleton(self, img, landmarks):
        h, w = img.shape[:2]

        # Draw connections
        for start_idx, end_idx in POSE_CONNECTIONS:

            if (
                start_idx >= len(landmarks)
                or end_idx >= len(landmarks)
            ):
                continue

            p1 = landmarks[start_idx]
            p2 = landmarks[end_idx]

            if (
                getattr(p1, "visibility", 1.0) > 0.7
                and getattr(p2, "visibility", 1.0) > 0.7
            ):
                cv2.line(
                    img,
                    (
                        int(p1.x * w),
                        int(p1.y * h),
                    ),
                    (
                        int(p2.x * w),
                        int(p2.y * h),
                    ),
                    (0, 255, 0),
                    4,
                )

        # Draw landmarks
        for lm in landmarks:

            if getattr(lm, "visibility", 1.0) > 0.7:

                cv2.circle(
                    img,
                    (
                        int(lm.x * w),
                        int(lm.y * h),
                    ),
                    6,
                    (255, 0, 0),
                    -1,
                )

    # ======================================================
    # NO POSE WARNING
    # ======================================================

    def _draw_no_pose_warnings(self, img):

        cv2.putText(
            img,
            "NO POSE DETECTED",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            img,
            "PLEASE FACE THE CAMERA",
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # ======================================================
    # OVERLAYS
    # ======================================================

    def _draw_overlays(self, img, metrics, ex_type):

        if ex_type == "Squats":
            self._draw_squats_overlays(img, metrics)

        elif ex_type == "Push-ups":
            self._draw_pushup_overlays(img, metrics)

        elif ex_type == "Biceps Curls (Dumbbell)":
            self._draw_curl_overlays(img, metrics)

        elif ex_type == "Shoulder Press":
            self._draw_press_overlays(img, metrics)

        elif ex_type == "Lunges":
            self._draw_lunge_overlays(img, metrics)

    # ======================================================
    # SQUAT OVERLAY
    # ======================================================

    def _draw_squats_overlays(self, img, metrics):

        h, _ = img.shape[:2]

        cv2.putText(
            img,
            f"DEPTH: {metrics.get('depth_status', 'N/A')}",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # ======================================================
    # PUSH-UP OVERLAY
    # ======================================================

    def _draw_pushup_overlays(self, img, metrics):

        h, _ = img.shape[:2]

        body = metrics.get(
            "body_alignment",
            "N/A",
        )

        hip = metrics.get(
            "hip_status",
            "N/A",
        )

        cv2.putText(
            img,
            f"BODY: {body} | HIP: {hip}",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # ======================================================
    # BICEPS CURL OVERLAY
    # ======================================================

    def _draw_curl_overlays(self, img, metrics):

        h, _ = img.shape[:2]

        swing = metrics.get(
            "swing_status",
            "N/A",
        )

        cv2.putText(
            img,
            f"SWING: {swing}",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # ======================================================
    # SHOULDER PRESS OVERLAY
    # ======================================================

    def _draw_press_overlays(self, img, metrics):

        h, _ = img.shape[:2]

        extension = metrics.get(
            "extension_status",
            "N/A",
        )

        back = metrics.get(
            "back_arch_status",
            "N/A",
        )

        cv2.putText(
            img,
            f"EXT: {extension} | BACK: {back}",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # ======================================================
    # LUNGE OVERLAY
    # ======================================================

    def _draw_lunge_overlays(self, img, metrics):

        h, _ = img.shape[:2]

        balance = metrics.get(
            "balance_status",
            "N/A",
        )

        cv2.putText(
            img,
            f"BALANCE: {balance}",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # ======================================================
    # RECEIVE CAMERA FRAME
    # ======================================================

    def recv(self, frame):

        # --------------------------------------------------
        # CAMERA FRAME
        # --------------------------------------------------

        image = np.asarray(
            cv2.flip(
                frame.to_ndarray(
                    format="bgr24"
                ),
                1,
            ),
            dtype=np.uint8,
        )

        # --------------------------------------------------
        # BGR -> RGB
        # MediaPipe expects RGB
        # --------------------------------------------------

        rgb_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_image,
        )

        # --------------------------------------------------
        # TIMESTAMP
        # --------------------------------------------------

        self._frame_timestamps_ms += 30

        # --------------------------------------------------
        # MEDIAPIPE INFERENCE
        # --------------------------------------------------

        result = self._landmarker.detect_for_video(
            mp_image,
            self._frame_timestamps_ms,
        )

        # ==================================================
        # POSE DETECTED
        # ==================================================

        if result.pose_landmarks:

            landmarks = result.pose_landmarks[0]

            self._draw_skeleton(
                image,
                landmarks,
            )

            ex_type = self.get_exercise()

            detector = self._detectors.get(
                ex_type
            )

            if detector:

                metrics = detector.process(
                    landmarks
                )

                metrics["pose_detected"] = True

                self._draw_overlays(
                    image,
                    metrics,
                    ex_type,
                )

                self.set_latest_metrics(
                    metrics
                )

        # ==================================================
        # NO POSE DETECTED
        # ==================================================

        else:

            self._draw_no_pose_warnings(
                image
            )

            with self._lock:

                if self._latest_metrics is not None:

                    self._latest_metrics[
                        "pose_detected"
                    ] = False

                else:

                    self._latest_metrics = {
                        "pose_detected": False
                    }

        # --------------------------------------------------
        # RETURN FRAME
        # --------------------------------------------------

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24",
        )