"""Compare the old detector (Haar cascade) with the new one (YuNet) on the same frames.

    python benchmark.py               # 200 webcam frames
    python benchmark.py --images dir  # a folder of photos

Prints speed and how often each detector found a face, and saves a
side-by-side image to data/benchmark.jpg for your README.
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

import config
from detector import FaceDetector


def load_haar():
    local = config.ROOT / "haarcascade_frontalface_default.xml"
    path = local if local.exists() else Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(str(path))


def frames_from(args):
    if args.images:
        for p in sorted(Path(args.images).iterdir()):
            img = cv2.imread(str(p))
            if img is not None:
                yield img
        return
    cap = cv2.VideoCapture(args.camera)
    for _ in range(args.frames):
        ok, frame = cap.read()
        if not ok:
            break
        yield cv2.flip(frame, 1)
    cap.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--frames", type=int, default=200)
    args = parser.parse_args()

    config.check_models()
    haar = load_haar()
    yunet = FaceDetector(config.YUNET_PATH, config.DETECTION_SCORE)
    stats = {"Haar cascade": [[], 0], "YuNet (DNN)": [[], 0]}  # [times_ms, frames_with_face]
    n, sample = 0, None

    for frame in frames_from(args):
        n += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        t = time.perf_counter()
        h_faces = haar.detectMultiScale(gray, 1.1, 5)
        stats["Haar cascade"][0].append((time.perf_counter() - t) * 1000)
        stats["Haar cascade"][1] += len(h_faces) > 0

        t = time.perf_counter()
        y_faces = yunet.detect(frame)
        stats["YuNet (DNN)"][0].append((time.perf_counter() - t) * 1000)
        stats["YuNet (DNN)"][1] += len(y_faces) > 0

        if sample is None or n == 30:  # keep one frame for the comparison image
            left, right = frame.copy(), frame.copy()
            for (x, y, w, h) in h_faces:
                cv2.rectangle(left, (x, y), (x + w, y + h), (70, 70, 235), 2)
            for f in y_faces:
                x, y, w, h = f.box
                cv2.rectangle(right, (x, y), (x + w, y + h), (80, 220, 100), 2)
                cv2.putText(right, f"{f.score:.0%}", (x, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 220, 100), 2)
            cv2.putText(left, "Haar cascade", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (70, 70, 235), 2)
            cv2.putText(right, "YuNet", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 220, 100), 2)
            sample = np.hstack([left, right])

    if n == 0:
        raise SystemExit("No frames read.")

    print(f"\n{n} frames\n")
    print(f"| {'Detector':<14} | {'avg ms':>7} | {'FPS':>6} | {'face found':>10} |")
    print(f"|{'-'*16}|{'-'*9}|{'-'*8}|{'-'*12}|")
    for name, (times, found) in stats.items():
        avg = float(np.mean(times))
        print(f"| {name:<14} | {avg:7.1f} | {1000/avg:6.0f} | {found/n:10.0%} |")

    config.DATA_DIR.mkdir(exist_ok=True)
    out = config.DATA_DIR / "benchmark.jpg"
    cv2.imwrite(str(out), sample)
    print(f"\nSide-by-side image saved to {out}")


if __name__ == "__main__":
    main()
