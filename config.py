"""All tunable settings live here, so you can experiment without touching the logic."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"

# Pretrained models (downloaded by download_models.py)
YUNET_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"

# SQLite database: students + attendance.
# NOTE: no photos are ever saved, only 128-number embeddings.
DB_PATH = DATA_DIR / "facegate.db"

# --- Detection -------------------------------------------------------------
DETECTION_SCORE = 0.8      # YuNet confidence (0..1) needed to accept a face

# --- Recognition -----------------------------------------------------------
# Cosine similarity between two embeddings. 0.363 is the threshold OpenCV
# recommends for SFace. Higher = stricter (fewer false matches).
MATCH_THRESHOLD = 0.363

# --- Liveness (head-turn challenge) ----------------------------------------
# "yaw" = how far the nose is from the middle of the eyes, divided by the
# distance between the eyes. ~0 when looking straight, grows when turning.
# Both are measured relative to the yaw at the start of the challenge.
# The live yaw value is shown on screen, so you can tune these for your camera.
YAW_TURNED = 0.30          # must go beyond this to count as "turned"
YAW_CENTER = 0.15          # must come back under this to count as "facing camera"
CHALLENGE_TIMEOUT = 6.0    # seconds to complete the challenge


def check_models():
    """Stop with a clear message instead of a scary OpenCV error."""
    for path in (YUNET_PATH, SFACE_PATH):
        if not path.exists() or path.stat().st_size < 100_000:
            raise SystemExit(f"Model missing: models/{path.name}\nRun: python download_models.py")
