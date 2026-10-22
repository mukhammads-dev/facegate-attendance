"""Register a whole school from a folder of photos, in one command.

Expected layout (one folder per class, one folder per student):

    students/
      7-A/
        Ali Valiyev/      1.jpg  2.jpg  3.jpg
        Madina Karimova/  1.jpg  2.jpg
      7-B/
        Jasur Toshmatov/  photo.png

    python enroll_folder.py students

2-3 photos per student (different angles/light) work better than one.
Running it again updates existing students instead of duplicating them.
"""
import argparse
from pathlib import Path

import config
from database import Database
from detector import FaceDetector
from recognizer import FaceRecognizer
from register import samples_from_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="folder with class/student subfolders")
    args = parser.parse_args()

    config.check_models()
    db = Database(config.DB_PATH)
    detector = FaceDetector(config.YUNET_PATH, config.DETECTION_SCORE)
    recognizer = FaceRecognizer(config.SFACE_PATH, db, config.MATCH_THRESHOLD)

    ok, failed = 0, []
    for class_dir in sorted(p for p in Path(args.root).iterdir() if p.is_dir()):
        print(f"[{class_dir.name}]")
        for student_dir in sorted(p for p in class_dir.iterdir() if p.is_dir()):
            samples = samples_from_images(student_dir, detector, recognizer)
            if not samples:
                print(f"  FAIL {student_dir.name}: no usable photo")
                failed.append(f"{class_dir.name}/{student_dir.name}")
                continue
            db.upsert_student(student_dir.name, class_dir.name, FaceRecognizer.average(samples))
            print(f"  ok   {student_dir.name}  ({len(samples)} photos)")
            ok += 1

    print(f"\nRegistered {ok} students.")
    if failed:
        print("Need new photos for:\n  " + "\n  ".join(failed))


if __name__ == "__main__":
    main()
