"""Face detection with YuNet (a small deep-learning model built into OpenCV)."""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Face:
    raw: np.ndarray  # the 15 numbers YuNet returns (SFace needs them for alignment)

    # YuNet row layout:
    # [x, y, w, h, right_eye(x,y), left_eye(x,y), nose(x,y),
    #  mouth_right(x,y), mouth_left(x,y), score]

    @property
    def box(self):
        x, y, w, h = self.raw[:4].astype(int)
        return x, y, w, h

    @property
    def score(self) -> float:
        return float(self.raw[14])

    @property
    def landmarks(self) -> np.ndarray:
        return self.raw[4:14].reshape(5, 2)

    @property
    def area(self) -> int:
        _, _, w, h = self.box
        return w * h

    @property
    def yaw(self) -> float:
        """Rough head-turn estimate from 3 landmarks.

        Looking straight: the nose sits between the eyes -> yaw ~ 0.
        Turning: the nose moves toward one side -> yaw becomes + or -.
        Dividing by eye distance makes it independent of how close you are.
        """
        eye_a, eye_b, nose = self.landmarks[0], self.landmarks[1], self.landmarks[2]
        eyes_mid_x = (eye_a[0] + eye_b[0]) / 2
        eye_dist = abs(eye_a[0] - eye_b[0]) + 1e-6
        return float((nose[0] - eyes_mid_x) / eye_dist)


class FaceDetector:
    def __init__(self, model_path, score_threshold: float = 0.8):
        self.model = cv2.FaceDetectorYN.create(
            str(model_path), "", (320, 320), score_threshold, 0.3, 5000
        )

    def detect(self, frame) -> list[Face]:
        h, w = frame.shape[:2]
        self.model.setInputSize((w, h))  # YuNet must know the image size
        _, faces = self.model.detect(frame)
        if faces is None:
            return []
        return [Face(row.astype(np.float32)) for row in faces]
