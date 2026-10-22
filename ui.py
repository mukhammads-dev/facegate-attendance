"""Drawing helpers: face boxes, privacy blur and the status panel."""
import cv2

GREEN = (80, 220, 100)
RED = (70, 70, 235)
YELLOW = (40, 200, 240)
WHITE = (240, 240, 240)
GRAY = (150, 150, 150)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _clip(frame, face):
    x, y, w, h = face.box
    H, W = frame.shape[:2]
    return max(0, x), max(0, y), min(W, x + w), min(H, y + h)


def blur_face(frame, face):
    """Privacy: hide faces we don't know."""
    x1, y1, x2, y2 = _clip(frame, face)
    if x2 > x1 and y2 > y1:
        roi = frame[y1:y2, x1:x2]
        k = max(15, ((x2 - x1) // 3) | 1)  # kernel size must be odd
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)


def draw_face(frame, face, color, label, sublabel="", show_landmarks=True):
    x1, y1, x2, y2 = _clip(frame, face)
    L = max(12, (x2 - x1) // 5)
    # corner brackets instead of a full rectangle
    for (cx, cy, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(frame, (cx, cy), (cx + dx * L, cy), color, 2, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * L), color, 2, cv2.LINE_AA)
    if show_landmarks:
        for (px, py) in face.landmarks.astype(int):
            cv2.circle(frame, (int(px), int(py)), 2, color, -1, cv2.LINE_AA)

    (tw, th), _ = cv2.getTextSize(label, FONT, 0.6, 2)
    ty = max(th + 8, y1 - 8)
    cv2.rectangle(frame, (x1, ty - th - 6), (x1 + tw + 10, ty + 4), color, -1)
    cv2.putText(frame, label, (x1 + 5, ty), FONT, 0.6, (20, 20, 20), 2, cv2.LINE_AA)
    if sublabel:
        cv2.putText(frame, sublabel, (x1, y2 + 20), FONT, 0.5, color, 1, cv2.LINE_AA)


def draw_panel(frame, title, color, lines=(), progress=None):
    """Status bar at the top of the window."""
    W = frame.shape[1]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, 70), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.putText(frame, title, (16, 32), FONT, 0.8, color, 2, cv2.LINE_AA)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (16, 58 + i * 20), FONT, 0.5, GRAY, 1, cv2.LINE_AA)
    if progress is not None:  # e.g. time left in the challenge
        cv2.rectangle(frame, (0, 66), (int(W * progress), 70), color, -1)


def draw_footer(frame, text):
    H = frame.shape[0]
    cv2.putText(frame, text, (16, H - 14), FONT, 0.5, GRAY, 1, cv2.LINE_AA)
