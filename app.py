"""FaceGate - spoof-resistant face attendance.

Flow for the person in front of the camera:
  detect face (YuNet) -> recognize which student it is (SFace)
  -> random head-turn challenge (liveness) -> save attendance (SQLite)

Keys:  q = quit   b = toggle privacy blur   l = toggle landmarks
"""
import argparse
import time

import cv2

import config
import ui
from database import Database
from detector import FaceDetector
from liveness import FAILED, PASSED, LivenessChallenge
from recognizer import FaceRecognizer


def label(student) -> str:
    return f"{student['name']} ({student['class_name']})"


class AttendanceApp:
    def __init__(self, db: Database):
        config.check_models()
        self.db = db
        self.detector = FaceDetector(config.YUNET_PATH, config.DETECTION_SCORE)
        self.recognizer = FaceRecognizer(config.SFACE_PATH, db, config.MATCH_THRESHOLD)
        self.challenge = None
        self.challenge_id = None     # which student the current challenge is for
        self.message = None          # (title, color, show_until)
        self.cooldown_until = 0.0
        self.blur_unknown = True
        self.show_landmarks = True

    def _start_challenge(self, student_id):
        self.challenge = LivenessChallenge(
            config.YAW_TURNED, config.YAW_CENTER, config.CHALLENGE_TIMEOUT
        )
        self.challenge_id = student_id

    def _flash(self, title, color, seconds):
        self.message = (title, color, time.time() + seconds)

    def process(self, frame):
        """Takes one camera frame, returns the annotated frame."""
        now = time.time()
        faces = self.detector.detect(frame)

        # Recognize every face; the biggest (closest) one is the one checking in.
        results = []
        for face in faces:
            student, sim = self.recognizer.identify(self.recognizer.embed(frame, face))
            results.append((face, student, sim))
        results.sort(key=lambda r: r[0].area, reverse=True)
        active = results[0] if results else None

        # ---- state machine for the active person -------------------------
        status, status_color, lines, progress = "Look at the camera", ui.WHITE, [], None
        if active is None:
            self.challenge = None
        else:
            face, student, sim = active
            if student is None:
                self.challenge = None
                status, status_color = "Unknown face", ui.RED
                lines = ["Not registered. Run: python register.py --name \"Name\" --class 7-B"]
            elif self.db.checked_in(student["id"]):
                self.challenge = None
                status, status_color = f"{label(student)}: already checked in", ui.GREEN
                lines = [f"Today at {self.db.checked_in(student['id'])}"]
            elif now < self.cooldown_until:
                status, status_color = "Liveness failed - try again", ui.RED
            else:
                if self.challenge is None or self.challenge_id != student["id"]:
                    self._start_challenge(student["id"])
                state = self.challenge.update(face.yaw)
                if state == PASSED:
                    t = self.db.mark(student["id"], sim)
                    self._flash(f"Welcome, {student['name']}!  Checked in {t}", ui.GREEN, 3)
                    self.challenge = None
                elif state == FAILED:
                    self.cooldown_until = now + 2
                    self.challenge = None
                else:
                    status, status_color = self.challenge.instruction, ui.YELLOW
                    lines = [f"Hi {student['name']}! Quick liveness check   (yaw {face.yaw:+.2f})"]
                    progress = self.challenge.time_left / config.CHALLENGE_TIMEOUT

        # ---- draw ----------------------------------------------------------
        for face, student, sim in results:
            if student is None:
                if self.blur_unknown:
                    ui.blur_face(frame, face)
                ui.draw_face(frame, face, ui.RED, f"Unknown  {face.score:.0%}",
                             f"best match {sim:.2f}", self.show_landmarks)
            else:
                ui.draw_face(frame, face, ui.GREEN, f"{label(student)}  {sim:.2f}",
                             f"detect {face.score:.0%}  yaw {face.yaw:+.2f}", self.show_landmarks)

        if self.message and now < self.message[2]:
            ui.draw_panel(frame, self.message[0], self.message[1])
        else:
            ui.draw_panel(frame, status, status_color, lines, progress)
        return frame


def main():
    parser = argparse.ArgumentParser(description="FaceGate face attendance")
    parser.add_argument("--camera", default="0", help="camera index or video file path")
    args = parser.parse_args()

    source = int(args.camera) if args.camera.isdigit() else args.camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera/video: {args.camera}")

    app = AttendanceApp(Database(config.DB_PATH))
    print(f"Registered students: {len(app.recognizer.students)}")
    fps, last = 0.0, time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera stopped sending frames.")
                break
            frame = cv2.flip(frame, 1)  # mirror view, feels natural like a selfie
            frame = app.process(frame)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1 / max(now - last, 1e-6))
            last = now
            blur = "on" if app.blur_unknown else "off"
            ui.draw_footer(frame, f"FPS {fps:4.1f}  |  q quit  b blur({blur})  l landmarks")

            cv2.imshow("FaceGate", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("b"):
                app.blur_unknown = not app.blur_unknown
            if key == ord("l"):
                app.show_landmarks = not app.show_landmarks
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
