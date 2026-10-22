# FaceGate — spoof-resistant face attendance

Most webcam attendance demos can be fooled by holding up a photo on a phone.
**FaceGate** recognizes who you are **and** checks that you are a live person
with a random head-turn challenge before it marks you present. Built for a
school setting: students are organized by class, a whole school can be
enrolled from a photo folder, and teachers get a daily *present / absent*
report. It never stores face photos — only 128-number embeddings.

<!-- Add your demo GIF here: docs/demo.gif -->

## Features

| | |
|---|---|
| **Face detection** | YuNet (CNN, OpenCV DNN) with a real confidence score and 5 facial landmarks |
| **Face recognition** | SFace embeddings (128-d), matched against all students with one matrix multiplication |
| **Liveness check** | Random LEFT/RIGHT head-turn challenge measured from landmarks; a flat photo cannot pass |
| **School mode** | Students by class, bulk enrollment from folders, daily present/absent report, CSV export |
| **Privacy by design** | Only embeddings are stored; unknown faces are blurred on screen |
| **Storage** | SQLite (`students`, `attendance`), one check-in per student per day |
| **Benchmark** | Classic Haar cascade vs YuNet on the same frames |

## How it works

```
camera frame
   │
   ▼
YuNet detector ──► box + 5 landmarks + confidence
   │
   ▼
SFace ──► 128-d embedding ──► cosine similarity vs ALL students (one matmul)
   │                                   │
   │                          known? ──┴── unknown → blur + "Unknown"
   ▼
Liveness: "Turn your head LEFT/RIGHT" → "look back"   (yaw from nose vs eyes)
   │
   ▼
SQLite attendance (student, date, time, similarity) ──► report.py
```

**Liveness idea:** `yaw = (nose_x − eyes_center_x) / eye_distance`.
When a real head turns, the nose moves sideways relative to the eyes. A photo is
flat, so tilting or moving it does not change that ratio. The direction is
random, so it can't be prepared in advance.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python download_models.py                          # ~39 MB, official OpenCV model zoo

python register.py --name "Ali Valiyev" --class 7-B   # webcam enrollment
python enroll_folder.py students                    # or: whole school from students/<class>/<name>/*.jpg

python app.py                                       # check-in station
python report.py --class 7-B --csv                  # who came, who didn't
python benchmark.py                                 # Haar vs YuNet
```

Keys in `app.py`: `q` quit · `b` toggle blur · `l` toggle landmarks

## Example report

```
=== 7-B:  24/26 present ===
   [+] Ali Valiyev               07:58:12
   [+] Jasur Toshmatov           08:01:45
   [-] Madina Karimova           ABSENT
```

## Benchmark

Classic Haar cascade vs YuNet on the same 200 live webcam frames
(MacBook Air, Apple Silicon, CPU only, `python benchmark.py`):

| Detector | avg ms / frame | FPS | frames with a face found |
|---|---|---|---|
| Haar cascade | 60.0 | 17 | 88% |
| **YuNet (DNN)** | **52.9** | **19** | **100%** |

YuNet found the face in **every** frame, while Haar missed it in roughly 1 of
every 8 frames, at slightly better speed. YuNet also returns a real confidence
score and 5 landmarks, which the recognition and liveness steps depend on.

## Project structure

```
app.py             check-in station: camera loop + state machine
register.py        enroll / delete one student (webcam or photos)
enroll_folder.py   enroll a whole school from class/student folders
report.py          present / absent report, CSV export
database.py        SQLite schema and queries
detector.py        YuNet wrapper, landmarks, yaw estimate
recognizer.py      SFace embeddings + matching
liveness.py        head-turn challenge
ui.py              drawing helpers
benchmark.py       Haar vs YuNet
config.py          all thresholds in one place
```

## Limitations

- The head-turn check stops printed/phone photos, but a replayed video of the
  person turning could still pass. Production systems add depth/IR cameras or
  a trained anti-spoofing model.
- Recognition quality drops in poor lighting and at large angles; identical twins may be confused.
- Children's faces change quickly, so re-enrollment once a year is recommended.
- Face data of minors is biometric personal data: real deployments need parental
  consent and must follow local data-protection law.

## Models

- [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) — face detection
- [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) — face recognition
