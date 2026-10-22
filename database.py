"""SQLite database: students (with their face embedding) and daily attendance.

SQLite is a whole database in a single file (data/facegate.db). It ships with
Python, so there is nothing to install or run.

Tables
  students   : id, name, class_name, embedding (128 float32 numbers as bytes)
  attendance : student_id, date, time, similarity   (one row per student per day)
"""
import sqlite3
from datetime import datetime

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    class_name  TEXT NOT NULL,
    embedding   BLOB NOT NULL,
    created_at  TEXT NOT NULL,
    UNIQUE (name, class_name)
);
CREATE TABLE IF NOT EXISTS attendance (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id  INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    date        TEXT NOT NULL,
    time        TEXT NOT NULL,
    similarity  REAL NOT NULL,
    UNIQUE (student_id, date)
);
"""


def today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


class Database:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row      # rows behave like dicts
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)

    # ---- students ------------------------------------------------------------
    def upsert_student(self, name: str, class_name: str, embedding: np.ndarray) -> int:
        """Add a student, or replace the embedding if (name, class) already exists."""
        blob = embedding.astype(np.float32).tobytes()
        now = datetime.now().isoformat(timespec="seconds")
        self.conn.execute(
            """INSERT INTO students (name, class_name, embedding, created_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT (name, class_name) DO UPDATE SET embedding = excluded.embedding""",
            (name, class_name, blob, now),
        )
        self.conn.commit()
        row = self.conn.execute(
            "SELECT id FROM students WHERE name = ? AND class_name = ?", (name, class_name)
        ).fetchone()
        return row["id"]

    def all_students(self, class_name=None) -> list[dict]:
        sql, args = "SELECT id, name, class_name FROM students", ()
        if class_name:
            sql, args = sql + " WHERE class_name = ?", (class_name,)
        return [dict(r) for r in self.conn.execute(sql + " ORDER BY class_name, name", args)]

    def load_embeddings(self):
        """Return (list of student dicts, N x 128 matrix) for fast comparison."""
        rows = self.conn.execute("SELECT id, name, class_name, embedding FROM students").fetchall()
        students = [{"id": r["id"], "name": r["name"], "class_name": r["class_name"]} for r in rows]
        if not rows:
            return students, np.empty((0, 128), dtype=np.float32)
        matrix = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        return students, matrix

    def delete_student(self, name: str, class_name: str) -> bool:
        cur = self.conn.execute(
            "DELETE FROM students WHERE name = ? AND class_name = ?", (name, class_name)
        )
        self.conn.commit()
        return cur.rowcount > 0

    # ---- attendance ----------------------------------------------------------
    def checked_in(self, student_id: int, date=None):
        row = self.conn.execute(
            "SELECT time FROM attendance WHERE student_id = ? AND date = ?",
            (student_id, date or today()),
        ).fetchone()
        return row["time"] if row else None

    def mark(self, student_id: int, similarity: float) -> str:
        now = datetime.now().strftime("%H:%M:%S")
        self.conn.execute(
            "INSERT OR IGNORE INTO attendance (student_id, date, time, similarity) VALUES (?, ?, ?, ?)",
            (student_id, today(), now, round(similarity, 3)),
        )
        self.conn.commit()
        return self.checked_in(student_id)

    def day_report(self, date=None, class_name=None) -> list[dict]:
        """Every student with their check-in time for that day (None = absent)."""
        sql = """SELECT s.id, s.name, s.class_name, a.time, a.similarity
                 FROM students s
                 LEFT JOIN attendance a ON a.student_id = s.id AND a.date = ?"""
        args = [date or today()]
        if class_name:
            sql += " WHERE s.class_name = ?"
            args.append(class_name)
        sql += " ORDER BY s.class_name, s.name"
        return [dict(r) for r in self.conn.execute(sql, args)]
