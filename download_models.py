"""Download the two pretrained ONNX models from the official OpenCV model zoo."""
import shutil
import subprocess
import urllib.request

import config

BASE = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"
MODELS = {
    config.YUNET_PATH: f"{BASE}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    config.SFACE_PATH: f"{BASE}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}

config.MODELS_DIR.mkdir(exist_ok=True)
for path, url in MODELS.items():
    if path.exists() and path.stat().st_size > 100_000:
        print(f"already have {path.name}")
        continue
    print(f"downloading {path.name} ...")
    try:
        urllib.request.urlretrieve(url, path)
    except Exception as e:
        # Python from python.org on macOS often lacks SSL certificates.
        # curl uses the system certificates, so try it instead.
        print(f"  python download failed ({type(e).__name__}), trying curl ...")
        if shutil.which("curl"):
            subprocess.run(["curl", "-L", "--fail", "-o", str(path), url], check=False)
    if path.exists() and path.stat().st_size > 100_000:
        print(f"  done ({path.stat().st_size / 1e6:.1f} MB)")
    else:
        path.unlink(missing_ok=True)
        print(f"  failed. Download manually:\n  curl -L -o models/{path.name} {url}")
