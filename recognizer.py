"""Face recognition with SFace: turn a face into 128 numbers and compare them."""
import cv2
import numpy as np

from database import Database
from detector import Face


class FaceRecognizer:
    def __init__(self, model_path, db: Database, threshold: float = 0.363):
        self.model = cv2.FaceRecognizerSF.create(str(model_path), "")
        self.db = db
        self.threshold = threshold
        self.reload()

    def reload(self):
        """Read all students' embeddings from the database into memory."""
        self.students, self.embeddings = self.db.load_embeddings()

    def embed(self, frame, face: Face) -> np.ndarray:
        """Face -> 128-number vector ("face fingerprint"), length normalized to 1."""
        aligned = self.model.alignCrop(frame, face.raw)  # rotate/crop face to 112x112
        vec = self.model.feature(aligned).flatten()
        return vec / np.linalg.norm(vec)

    def identify(self, embedding: np.ndarray):
        """Compare against EVERY student at once.

        Returns (student dict or None, similarity). Similarity is cosine: 1.0 = identical.
        """
        if len(self.students) == 0:
            return None, 0.0
        sims = self.embeddings @ embedding   # one matrix multiplication = all comparisons
        best = int(np.argmax(sims))
        score = float(sims[best])
        return (self.students[best] if score >= self.threshold else None), score

    @staticmethod
    def average(samples: list[np.ndarray]) -> np.ndarray:
        """Average several samples into one stable embedding."""
        mean = np.mean(samples, axis=0)
        return (mean / np.linalg.norm(mean)).astype(np.float32)
