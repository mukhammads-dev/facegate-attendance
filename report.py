"""Attendance report: who came and who did not.

    python report.py                         # today, all classes
    python report.py --class 7-B             # one class
    python report.py --date 2026-10-21       # another day
    python report.py --class 7-B --csv       # also save a CSV you can open in Excel
    python report.py --students              # list registered students
"""
import argparse
import csv
from itertools import groupby

import config
from database import Database, today


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class", dest="class_name")
    parser.add_argument("--date", default=today(), help="YYYY-MM-DD")
    parser.add_argument("--csv", action="store_true", help="save to data/report_<class>_<date>.csv")
    parser.add_argument("--students", action="store_true", help="list registered students")
    args = parser.parse_args()

    db = Database(config.DB_PATH)

    if args.students:
        rows = db.all_students(args.class_name)
        for cls, group in groupby(rows, key=lambda r: r["class_name"]):
            group = list(group)
            print(f"{cls}  ({len(group)} students)")
            for r in group:
                print(f"   #{r['id']:<4} {r['name']}")
        if not rows:
            print("No students registered yet.")
        return

    rows = db.day_report(args.date, args.class_name)
    if not rows:
        raise SystemExit("No students found.")

    print(f"\nAttendance for {args.date}\n")
    for cls, group in groupby(rows, key=lambda r: r["class_name"]):
        group = list(group)
        present = [r for r in group if r["time"]]
        absent = [r for r in group if not r["time"]]
        print(f"=== {cls}:  {len(present)}/{len(group)} present ===")
        for r in present:
            print(f"   [+] {r['name']:<25} {r['time']}")
        for r in absent:
            print(f"   [-] {r['name']:<25} ABSENT")
        print()

    if args.csv:
        out = config.DATA_DIR / f"report_{args.class_name or 'all'}_{args.date}.csv"
        with open(out, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig: Excel-friendly
            writer = csv.writer(f)
            writer.writerow(["class", "name", "status", "time", "similarity"])
            for r in rows:
                writer.writerow([r["class_name"], r["name"],
                                 "present" if r["time"] else "absent",
                                 r["time"] or "", r["similarity"] or ""])
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
