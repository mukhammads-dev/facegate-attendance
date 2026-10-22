"""Register ONE student: capture face samples and store their average embedding.

From the webcam:   python register.py --name "Ali Valiyev" --class 7-B
From photos:       python register.py --name "Ali Valiyev" --class 7-B --images photos/ali
Remove a student:  python register.py --name "Ali Valiyev" --class 7-B --delete

Only the 128-number embedding is saved - never the photo itself.
For a whole school at once, use enroll_folder.py.
"""
import argparse
import time
from pathlib import Path

import cv2

import config
import ui
from database import Database
from detector import FaceDetector
from recognizer import FaceRecognizer

SAMPLES = 15
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def samples_from_images(folder, detector, recognizer, verbose=True):
    samples = []
    for path in sorted(Path(folder).iterdir()):
        if path.suffix.lower() not in IMAGE_TYPES:
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        faces = detector.detect(img)
        if len(faces) != 1:
            if verbose:
                print(f"    skip {path.name}: found {len(faces)} faces (need exactly 1)")
            continue
        samples.append(recognizer.embed(img, faces[0]))
    return samples


def samples_from_webcam(camera, detector, recognizer, title):
    cap = cv2.VideoCapture(camera)
    samples, last_shot = [], 0.0
    try:
        while len(samples) < SAMPLES:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            faces = detector.detect(frame)
            view = frame.copy()

            if len(faces) == 1:
                ui.draw_face(view, faces[0], ui.GREEN, f"{title}  {faces[0].score:.0%}")
                # take a sample every 0.3 s; moving your head a little gives variety
                if time.time() - last_shot > 0.3:
                    samples.append(recognizer.embed(frame, faces[0]))
                    last_shot = time.time()
                status, color = f"Capturing {len(samples)}/{SAMPLES}", ui.GREEN
            else:
                status, color = ("No face found" if not faces else "Only one person please"), ui.RED

            ui.draw_panel(view, status, color, ["Move your head slightly and change expression"],
                          progress=len(samples) / SAMPLES)
            cv2.imshow("FaceGate - Register", view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--class", dest="class_name", required=True, help='e.g. 7-B')
    parser.add_argument("--images", help="folder of photos instead of the webcam")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--delete", action="store_true", help="remove this student")
    args = parser.parse_args()

    db = Database(config.DB_PATH)
    if args.delete:
        ok = db.delete_student(args.name, args.class_name)
        print("Deleted." if ok else "No such student.")
        return

    config.check_models()
    detector = FaceDetector(config.YUNET_PATH, config.DETECTION_SCORE)
    recognizer = FaceRecognizer(config.SFACE_PATH, db, config.MATCH_THRESHOLD)

    if args.images:
        samples = samples_from_images(args.images, detector, recognizer)
    else:
        samples = samples_from_webcam(args.camera, detector, recognizer, args.name)

    if len(samples) < 3 and not args.images:
        raise SystemExit("Not enough good samples. Try again with better light.")
    if not samples:
        raise SystemExit("No usable photos (each photo needs exactly one face).")

    sid = db.upsert_student(args.name, args.class_name, FaceRecognizer.average(samples))
    print(f"Registered '{args.name}' ({args.class_name}) as id {sid} from {len(samples)} samples.")


if __name__ == "__main__":
    main()
