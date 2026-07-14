import os
import sqlite3
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "students.db")

ATTENDANCE_SLOTS = (
    ("hour_1_2", "1-2 Hour", 9, 11),
    ("hour_3_4", "3-4 Hour", 11, 13),
    ("hour_5_6", "5-6 Hour", 13, 15),
    ("hour_7_8", "7-8 Hour", 15, 17),
)


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def resolve_attendance_slot(marked_at: str):
    try:
        current = datetime.strptime((marked_at or "").strip(), "%H:%M:%S").time()
        minute_of_day = current.hour * 60 + current.minute
    except ValueError:
        minute_of_day = None

    if minute_of_day is not None:
        for slot_key, slot_label, start_hour, end_hour in ATTENDANCE_SLOTS:
            start_minute = start_hour * 60
            end_minute = end_hour * 60
            if start_minute <= minute_of_day < end_minute:
                return slot_key, slot_label

    if minute_of_day is None:
        return ATTENDANCE_SLOTS[0][0], ATTENDANCE_SLOTS[0][1]

    nearest = min(
        ATTENDANCE_SLOTS,
        key=lambda slot: abs(minute_of_day - (slot[2] * 60)),
    )
    return nearest[0], nearest[1]


def ensure_attendance_schema(cursor):
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'attendance'"
    )
    has_attendance_table = cursor.fetchone() is not None

    if not has_attendance_table:
        cursor.execute(
            """
            CREATE TABLE attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_name TEXT NOT NULL,
                student_id TEXT DEFAULT '',
                attendance_date TEXT NOT NULL,
                marked_at TEXT NOT NULL,
                confidence REAL NOT NULL,
                liveness_mode TEXT NOT NULL,
                camera_label TEXT DEFAULT '',
                slot_key TEXT NOT NULL,
                slot_label TEXT NOT NULL,
                UNIQUE(student_name, attendance_date, slot_key)
            )
            """
        )
        return

    cursor.execute("PRAGMA table_info(attendance)")
    columns = {row[1] for row in cursor.fetchall()}
    if {"slot_key", "slot_label"}.issubset(columns):
        return

    cursor.execute("ALTER TABLE attendance RENAME TO attendance_legacy")
    cursor.execute(
        """
        CREATE TABLE attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT NOT NULL,
            student_id TEXT DEFAULT '',
            attendance_date TEXT NOT NULL,
            marked_at TEXT NOT NULL,
            confidence REAL NOT NULL,
            liveness_mode TEXT NOT NULL,
            camera_label TEXT DEFAULT '',
            slot_key TEXT NOT NULL,
            slot_label TEXT NOT NULL,
            UNIQUE(student_name, attendance_date, slot_key)
        )
        """
    )

    cursor.execute(
        """
        SELECT
            id, student_name, student_id, attendance_date, marked_at,
            confidence, liveness_mode, camera_label
        FROM attendance_legacy
        ORDER BY attendance_date ASC, marked_at ASC, id ASC
        """
    )
    legacy_rows = cursor.fetchall()

    for row in legacy_rows:
        slot_key, slot_label = resolve_attendance_slot(row["marked_at"])
        cursor.execute(
            """
            INSERT OR REPLACE INTO attendance (
                id, student_name, student_id, attendance_date, marked_at,
                confidence, liveness_mode, camera_label, slot_key, slot_label
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["id"],
                row["student_name"],
                row["student_id"] or "",
                row["attendance_date"],
                row["marked_at"],
                row["confidence"],
                row["liveness_mode"],
                row["camera_label"] or "",
                slot_key,
                slot_label,
            ),
        )

    cursor.execute("DROP TABLE attendance_legacy")


def ensure_students_schema(cursor):
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'students'"
    )
    has_students_table = cursor.fetchone() is not None

    if not has_students_table:
        cursor.execute(
            """
            CREATE TABLE students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                name TEXT UNIQUE NOT NULL,
                dob TEXT DEFAULT '',
                class_name TEXT DEFAULT '',
                dept TEXT DEFAULT '',
                batch TEXT DEFAULT '',
                status TEXT DEFAULT 'active',
                on_duty INTEGER DEFAULT 0,
                image TEXT DEFAULT '',
                profile_json TEXT DEFAULT '',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(student_id, dept)
            )
            """
        )
        return

    cursor.execute("PRAGMA index_list(students)")
    indexes = cursor.fetchall()
    has_student_id_dept_unique = False
    has_legacy_student_id_unique = False

    for index in indexes:
        index_name = index[1]
        is_unique = bool(index[2])
        if not is_unique:
            continue
        cursor.execute(f"PRAGMA index_info({index_name})")
        columns = [row[2] for row in cursor.fetchall()]
        if columns == ["student_id", "dept"]:
            has_student_id_dept_unique = True
        if columns == ["student_id"]:
            has_legacy_student_id_unique = True

    if has_student_id_dept_unique and not has_legacy_student_id_unique:
        return

    cursor.execute("ALTER TABLE students RENAME TO students_legacy")
    cursor.execute(
        """
        CREATE TABLE students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            name TEXT UNIQUE NOT NULL,
            dob TEXT DEFAULT '',
            class_name TEXT DEFAULT '',
            dept TEXT DEFAULT '',
            batch TEXT DEFAULT '',
            status TEXT DEFAULT 'active',
            on_duty INTEGER DEFAULT 0,
            image TEXT DEFAULT '',
            profile_json TEXT DEFAULT '',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(student_id, dept)
        )
        """
    )
    cursor.execute(
        """
        INSERT INTO students (
            id, student_id, name, dob, class_name, dept, batch, status,
            on_duty, image, profile_json, created_at, updated_at
        )
        SELECT
            id, student_id, name, dob, class_name, dept, batch, status,
            on_duty, image, profile_json, created_at, updated_at
        FROM students_legacy
        ORDER BY id ASC
        """
    )
    cursor.execute("DROP TABLE students_legacy")


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cursor.execute(
        """
        INSERT OR IGNORE INTO users (username, password)
        VALUES (?, ?)
        """,
        ("staff", "staff123"),
    )

    ensure_students_schema(cursor)

    ensure_attendance_schema(cursor)

    conn.commit()
    conn.close()
