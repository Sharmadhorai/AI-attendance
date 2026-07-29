import csv
import json
import os
import pickle
import re
import sqlite3
from datetime import datetime
from typing import Optional
from ultralytics import YOLO
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from insightface.app import FaceAnalysis
from pydantic import BaseModel
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

try:
    from database import get_connection, init_db
except ImportError:
    from .database import get_connection, init_db


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
FRONTEND_DIR = os.path.join(PROJECT_DIR, "frontend")
STUDENT_IMAGES_DIR = os.path.join(BASE_DIR, "student_images")
STUDENT_PROFILES_DIR = os.path.join(BASE_DIR, "student_profiles")
ATTENDANCE_FILE = os.path.join(BASE_DIR, "attendance.csv")
ATTENDANCE_PDF = os.path.join(BASE_DIR, "attendance_report.pdf")
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "embeddings.pkl")
MATCH_THRESHOLD = 0.55
TEMP_ID_PREFIX = "__TMP__"
LIVENESS_MODE = "single_blink_plus_motion"
ATTENDANCE_SLOTS = (
    ("hour_1_2", "1-2 Hour", 9, 11),
    ("hour_3_4", "3-4 Hour", 11, 13),
    ("hour_5_6", "5-6 Hour", 13, 15),
    ("hour_7_8", "7-8 Hour", 15, 17),
)
DEPARTMENT_OPTIONS = (
    "B.tech IT",
    "B.tech AI&DS",
    "B.E CSE",
    "B.E CSE (CS)",
    "B.E ECE",
    "B.E R&A",
)
DEPARTMENT_ORDER = {name: index for index, name in enumerate(DEPARTMENT_OPTIONS)}
DEPARTMENT_ALIASES = {
    "b.tech it": "B.tech IT",
    "btech it": "B.tech IT",
    "b.tech ai&ds": "B.tech AI&DS",
    "b.tech ai and ds": "B.tech AI&DS",
    "btech ai&ds": "B.tech AI&DS",
    "b.e cse": "B.E CSE",
    "be cse": "B.E CSE",
    "b.e cse (cs)": "B.E CSE (CS)",
    "be cse (cs)": "B.E CSE (CS)",
    "b.e ece": "B.E ECE",
    "be ece": "B.E ECE",
    "b.e r&a": "B.E R&A",
    "b.e r and a": "B.E R&A",
    "be r&a": "B.E R&A",
}

LEFT_EYE_POINTS = slice(33, 43)
RIGHT_EYE_POINTS = slice(87, 97)

os.makedirs(STUDENT_IMAGES_DIR, exist_ok=True)
os.makedirs(STUDENT_PROFILES_DIR, exist_ok=True)
init_db()

app = FastAPI(title="AI Attendance System", version="4.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.isdir(FRONTEND_DIR):
    app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")
app.mount("/student-images", StaticFiles(directory=STUDENT_IMAGES_DIR), name="student-images")


class LoginPayload(BaseModel):
    username: str
    password: str


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    return cleaned.strip("_") or "student"


def image_url(path: str) -> str:
    return f"/student-images/{os.path.basename(path)}" if path else ""


def normalize_student_id(value: str) -> str:
    cleaned = re.sub(r"\D+", "", (value or "").strip())
    if not cleaned:
        raise HTTPException(status_code=400, detail="Student ID must contain numbers only")
    return cleaned.zfill(2)


def validate_department(value: str) -> str:
    dept = (value or "").strip()
    if dept in DEPARTMENT_OPTIONS:
        return dept

    normalized_key = re.sub(r"\s+", " ", dept).strip().lower()
    normalized_key = normalized_key.replace("b.tech", "b.tech").replace("b.tech", "b.tech")
    mapped = DEPARTMENT_ALIASES.get(normalized_key)
    if mapped:
        return mapped

    title_key = dept.lower()
    for option in DEPARTMENT_OPTIONS:
        if option.lower() == title_key:
            return option

    if dept not in DEPARTMENT_OPTIONS:
        raise HTTPException(status_code=400, detail="Please select a valid department")
    return dept


def department_sort_key(value: str):
    dept = (value or "").strip()
    return (DEPARTMENT_ORDER.get(dept, len(DEPARTMENT_ORDER)), dept.lower())


def resolve_attendance_slot(marked_at: Optional[str] = None):
    time_value = marked_at or datetime.now().strftime("%H:%M:%S")
    try:
        current = datetime.strptime(time_value.strip(), "%H:%M:%S").time()
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


def load_embeddings():
    if not os.path.exists(EMBEDDINGS_FILE):
        return []
    with open(EMBEDDINGS_FILE, "rb") as handle:
        return pickle.load(handle)


KNOWN_FACES = load_embeddings()
FACE_APP = FaceAnalysis(name="buffalo_s", allowed_modules=["detection", "recognition", "landmark_2d_106"])
PHONE_MODEL = YOLO(os.path.join(BASE_DIR, "yolov8n.pt"))
try:
    FACE_APP.prepare(ctx_id=0, det_size=(320, 320))
except Exception:
    FACE_APP.prepare(ctx_id=-1, det_size=(320, 320))


def save_embeddings():
    with open(EMBEDDINGS_FILE, "wb") as handle:
        pickle.dump(KNOWN_FACES, handle)


def replace_embedding(name: str, embedding):
    global KNOWN_FACES
    key = name.strip().lower()
    KNOWN_FACES = [item for item in KNOWN_FACES if item["name"].strip().lower() != key]
    KNOWN_FACES.append({"name": name.strip(), "embedding": embedding})
    save_embeddings()


def remove_embedding(name: str):
    global KNOWN_FACES
    key = name.strip().lower()
    original_count = len(KNOWN_FACES)
    KNOWN_FACES = [item for item in KNOWN_FACES if item["name"].strip().lower() != key]
    if len(KNOWN_FACES) != original_count:
        save_embeddings()


def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return -1.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def best_face_match(embedding):
    best_score = -1.0
    best_name = "Unknown"
    for known in KNOWN_FACES:
        score = cosine_similarity(embedding, known["embedding"])
        if score > best_score:
            best_score = score
            best_name = known["name"]
    return best_name, best_score


def decode_image(image_bytes: bytes):
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    return image


def extract_embedding(image_bytes: bytes):
    image = decode_image(image_bytes)
    faces = FACE_APP.get(image)
    if not faces:
        raise HTTPException(status_code=400, detail="No face found in uploaded image")
    return max(
        faces,
        key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]),
    ).embedding


def save_student_image(name: str, image_bytes: bytes):
    image = decode_image(image_bytes)
    path = os.path.join(STUDENT_IMAGES_DIR, f"{sanitize_name(name)}.png")
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise HTTPException(status_code=400, detail="Unable to save student image")
    with open(path, "wb") as handle:
        handle.write(encoded.tobytes())
    return path

def detect_mobile(frame):

    results = PHONE_MODEL(frame)

    mobile_boxes = []

    for result in results:

        for box in result.boxes:

            cls = int(box.cls[0])

            label = PHONE_MODEL.names[cls]

            confidence = float(box.conf[0])

            print("Detected:", label, confidence)

            if label in ["cell phone", "tv"] and confidence > 0.15:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                mobile_boxes.append({
                    "label": "Mobile",
                    "confidence": round(confidence * 100, 2),
                    "bbox": [x1, y1, x2, y2]
                })

    return mobile_boxes



def eye_state(face):
    landmarks = getattr(face, "landmark_2d_106", None)
    if landmarks is None:
        return "unclear"

    try:
        points = np.asarray(landmarks, dtype=np.float32)
        if points.shape[0] < 106:
            return "unclear"

        def open_ratio(eye_points):
            x_coords = eye_points[:, 0]
            y_coords = eye_points[:, 1]
            width = float(np.max(x_coords) - np.min(x_coords))
            height = float(np.max(y_coords) - np.min(y_coords))
            if width <= 0:
                return 0.0
            return height / width

        left_ratio = open_ratio(points[LEFT_EYE_POINTS])
        right_ratio = open_ratio(points[RIGHT_EYE_POINTS])
        blink_threshold = 0.20
        return "closed" if min(left_ratio, right_ratio) < 0.35 else "open"
    except Exception:
        return "unclear"


def build_face_payload(face, name, confidence):
    bbox = [int(value) for value in face.bbox]
    width = max(bbox[2] - bbox[0], 1)
    height = max(bbox[3] - bbox[1], 1)
    return {
        "name": name,
        "confidence": round(confidence * 100, 2),
        "bbox": bbox,
        "eye_state": eye_state(face),
        "face_size": {"width": width, "height": height},
    }


def serialize_student(row):
    return {
        "id": row["id"],
        "student_id": row["student_id"] or "",
        "name": row["name"] or "",
        "dob": row["dob"] or "",
        "class_name": row["class_name"] or "",
        "dept": row["dept"] or "",
        "batch": row["batch"] or "",
        "status": row["status"] or "active",
        "on_duty": bool(row["on_duty"]),
        "image_url": image_url(row["image"] or ""),
    }


def ensure_attendance_csv():
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                ["Name", "Student ID", "Class", "Department", "Date", "Hour Slot", "Time", "Confidence"]
            )


def get_attendance_records(date: Optional[str] = None):
    conn = get_connection()
    cursor = conn.cursor()
    if date:
        cursor.execute(
            """
            SELECT
                attendance.id,
                attendance.student_name,
                attendance.student_id,
                attendance.attendance_date,
                attendance.marked_at,
                attendance.confidence,
                attendance.liveness_mode,
                attendance.camera_label,
                attendance.slot_key,
                attendance.slot_label,
                students.class_name,
                students.dept,
                students.batch,
                students.on_duty
            FROM attendance
            LEFT JOIN students
                ON LOWER(students.name) = LOWER(attendance.student_name)
            WHERE attendance.attendance_date = ?
            ORDER BY attendance.marked_at DESC
            """,
            (date,),
        )
    else:
        cursor.execute(
            """
            SELECT
                attendance.id,
                attendance.student_name,
                attendance.student_id,
                attendance.attendance_date,
                attendance.marked_at,
                attendance.confidence,
                attendance.liveness_mode,
                attendance.camera_label,
                attendance.slot_key,
                attendance.slot_label,
                students.class_name,
                students.dept,
                students.batch,
                students.on_duty
            FROM attendance
            LEFT JOIN students
                ON LOWER(students.name) = LOWER(attendance.student_name)
            ORDER BY attendance.attendance_date DESC, attendance.marked_at DESC
            """
        )
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "id": row["id"],
            "name": row["student_name"],
            "student_id": row["student_id"] or "",
            "date": row["attendance_date"],
            "time": row["marked_at"],
            "confidence": round(float(row["confidence"]) * 100, 2),
            "slot_key": row["slot_key"] or "",
            "slot_label": row["slot_label"] or "",
            "class_name": row["class_name"] or "",
            "dept": row["dept"] or "",
            "batch": row["batch"] or "",
            "mode": row["liveness_mode"] or "",
            "camera_label": row["camera_label"] or "",
            "on_duty": bool(row["on_duty"]),
        }
        for row in rows
    ]


def refresh_attendance_csv():
    ensure_attendance_csv()
    rows = get_attendance_records()
    with open(ATTENDANCE_FILE, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Name", "Student ID", "Class", "Department", "Date", "Hour Slot", "Time", "Confidence"])
        for row in rows:
            writer.writerow(
                [
                    row["name"],
                    row["student_id"],
                    row["class_name"],
                    row["dept"],
                    row["date"],
                    row["slot_label"],
                    row["time"],
                    f"{row['confidence']:.2f}%",
                ]
            )


def get_student_by_name(name: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students WHERE LOWER(name) = LOWER(?) LIMIT 1", (name.strip(),))
    row = cursor.fetchone()
    conn.close()
    return row


def mark_attendance(name: str, confidence: float, camera_label: str):
    student = get_student_by_name(name)
    student_id = student["student_id"] if student else ""
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%H:%M:%S")
    slot_key, slot_label = resolve_attendance_slot(now_time)

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT marked_at, confidence, slot_label
        FROM attendance
        WHERE LOWER(student_name) = LOWER(?) AND attendance_date = ? AND slot_key = ?
        LIMIT 1
        """,
        (name, today, slot_key),
    )
    existing = cursor.fetchone()
    if existing:
        conn.close()
        return {
            "status": "already_marked",
            "name": name,
            "student_id": student_id,
            "time": existing["marked_at"],
            "slot_label": existing["slot_label"] or slot_label,
            "confidence": round(float(existing["confidence"]) * 100, 2),
        }

    cursor.execute(
        """
        INSERT INTO attendance (
            student_name, student_id, attendance_date, marked_at,
            confidence, liveness_mode, camera_label, slot_key, slot_label
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (name, student_id, today, now_time, confidence, LIVENESS_MODE, camera_label, slot_key, slot_label),
    )
    conn.commit()
    conn.close()
    refresh_attendance_csv()
    return {
        "status": "marked",
        "name": name,
        "student_id": student_id,
        "time": now_time,
        "slot_label": slot_label,
        "confidence": round(confidence * 100, 2),
    }


def get_students():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students ORDER BY LOWER(name) ASC")
    rows = cursor.fetchall()
    conn.close()
    return [serialize_student(row) for row in rows]


def build_summary():
    today = datetime.now().strftime("%Y-%m-%d")
    students = get_students()
    attendance_today = get_attendance_records(today)
    present_names = {row["name"].strip().lower() for row in attendance_today}
    total_students = len(students)
    present_students = len(present_names)
    absent_students = max(total_students - present_students, 0)
    on_duty_students = sum(1 for student in students if student["on_duty"])
    return {
        "date": today,
        "total_students": total_students,
        "present_students": present_students,
        "absent_students": absent_students,
        "on_duty_students": on_duty_students,
        "attendance": attendance_today,
        "students": students,
    }


def build_hour_wise_report_rows_by_department(selected_department: Optional[str] = None):
    students = get_students()
    attendance_rows = get_attendance_records()
    attendance_dates = sorted({row["date"] for row in attendance_rows}, reverse=True)
    if not attendance_dates:
        attendance_dates = [datetime.now().strftime("%Y-%m-%d")]

    slot_order = [slot[0] for slot in ATTENDANCE_SLOTS]
    slot_labels = {slot[0]: slot[1] for slot in ATTENDANCE_SLOTS}
    grouped_records = {}

    for row in attendance_rows:
        grouped_records[(row["date"], row["student_id"] or row["name"].strip().lower(), row["slot_key"])] = row

    header = [
        "Date",
        "Name",
        "Student ID",
        "Class",
        "Department",
        slot_labels["hour_1_2"],
        slot_labels["hour_3_4"],
        slot_labels["hour_5_6"],
        slot_labels["hour_7_8"],
    ]

    departments = {}
    for student in students:
        department_name = (student["dept"] or "No Department").strip() or "No Department"
        departments.setdefault(department_name, []).append(student)

    department_sections = []
    for department_name in sorted(departments, key=department_sort_key):
        if selected_department and department_name != selected_department:
            continue
        rows = [header]
        for attendance_date in attendance_dates:
            for student in sorted(departments[department_name], key=lambda item: item["name"].lower()):
                student_key = student["student_id"] or student["name"].strip().lower()
                row = [
                    attendance_date,
                    student["name"],
                    student["student_id"],
                    student["class_name"] or "-",
                    student["dept"] or "-",
                ]

                for slot_key in slot_order:
                    record = grouped_records.get((attendance_date, student_key, slot_key))
                    if record:
                        row.append("OD" if record["on_duty"] else "Present")
                    else:
                        row.append("OD" if student["on_duty"] else "Absent")

                rows.append(row)

        if len(rows) == 1:
            rows.append(["-", "No students found", "-", "-", "-", "-", "-", "-", "-"])

        department_sections.append(
            {
                "department": department_name,
                "rows": rows,
            }
        )

    if not department_sections:
        department_sections.append(
            {
                "department": selected_department or "No Department",
                "rows": [header, ["-", "No students found", "-", "-", "-", "-", "-", "-", "-"]],
            }
        )

    return department_sections


ensure_attendance_csv()
refresh_attendance_csv()


@app.get("/")
def root():
    login_file = os.path.join(FRONTEND_DIR, "login.html")
    if not os.path.exists(login_file):
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(login_file)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "students": len(KNOWN_FACES),
        "match_threshold": MATCH_THRESHOLD,
        "liveness_mode": LIVENESS_MODE,
        "camera_policy": "continuous_until_stop_button",
    }


@app.post("/auth/login")
def login(payload: LoginPayload):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT username FROM users WHERE username = ? AND password = ? LIMIT 1",
        (payload.username.strip(), payload.password.strip()),
    )
    user = cursor.fetchone()
    conn.close()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return {"success": True, "staff": user["username"]}


@app.post("/auth/register")
def register(payload: LoginPayload):
    username = payload.username.strip()
    password = payload.password.strip()
    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password are required")

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, password),
        )
        conn.commit()
    except Exception as exc:
        conn.close()
        raise HTTPException(status_code=400, detail=f"Unable to create account: {exc}")
    conn.close()
    return {"success": True, "staff": username}


@app.get("/dashboard/summary")
def dashboard_summary():
    return build_summary()


@app.get("/students")
def students():
    return get_students()


@app.post("/students")
async def add_student(
    student_id: str = Form(...),
    name: str = Form(...),
    class_name: str = Form(...),
    dept: str = Form(...),
    batch: str = Form(...),
    image: UploadFile = File(...),
):
    student_id = normalize_student_id(student_id)
    dept = validate_department(dept)
    image_bytes = await image.read()
    embedding = extract_embedding(image_bytes)
    image_path = save_student_image(name, image_bytes)

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO students (
                student_id, name, dob, class_name, dept, batch, on_duty, image,
                profile_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """,
            (
                student_id,
                name.strip(),
                "",
                class_name.strip(),
                dept,
                batch.strip(),
                0,
                image_path,
                json.dumps(
                    {
                        "student_id": student_id,
                        "name": name.strip(),
                        "dob": "",
                        "class_name": class_name.strip(),
                        "dept": dept,
                        "batch": batch.strip(),
                        "on_duty": False,
                        "image": image_path,
                    }
                ),
            ),
        )
        conn.commit()
        row_id = cursor.lastrowid
        cursor.execute("SELECT * FROM students WHERE id = ?", (row_id,))
        row = cursor.fetchone()
    except Exception as exc:
        conn.close()
        if isinstance(exc, sqlite3.IntegrityError) and "students.student_id" in str(exc):
            raise HTTPException(status_code=400, detail=f"ID {student_id} already exists in {dept}")
        raise HTTPException(status_code=400, detail=f"Unable to register student: {exc}")
    conn.close()

    replace_embedding(name, embedding)
    return {"message": "Student registered successfully", "student": serialize_student(row)}


@app.put("/students/{student_row_id}")
async def update_student(
    student_row_id: int,
    student_id: str = Form(...),
    name: str = Form(...),
    class_name: str = Form(...),
    dept: str = Form(...),
    batch: str = Form(...),
    image: Optional[UploadFile] = File(None),
):
    student_id = normalize_student_id(student_id)
    dept = validate_department(dept)
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students WHERE id = ? LIMIT 1", (student_row_id,))
    existing = cursor.fetchone()
    if not existing:
        conn.close()
        raise HTTPException(status_code=404, detail="Student not found")

    image_path = existing["image"] or ""
    if image is not None and image.filename:
        image_bytes = await image.read()
        embedding = extract_embedding(image_bytes)
        image_path = save_student_image(name, image_bytes)
        replace_embedding(name, embedding)
        if existing["name"].strip().lower() != name.strip().lower():
            remove_embedding(existing["name"])
    elif existing["name"].strip().lower() != name.strip().lower():
        existing_embedding = next(
            (item["embedding"] for item in KNOWN_FACES if item["name"].strip().lower() == existing["name"].strip().lower()),
            None,
        )
        if existing_embedding is not None:
            remove_embedding(existing["name"])
            replace_embedding(name, existing_embedding)

    profile_json = json.dumps(
        {
            "student_id": student_id,
            "name": name.strip(),
            "dob": "",
            "class_name": class_name.strip(),
            "dept": dept,
            "batch": batch.strip(),
            "on_duty": bool(existing["on_duty"]),
            "image": image_path,
        }
    )

    try:
        cursor.execute(
            """
            UPDATE students
            SET student_id = ?, name = ?, dob = ?, class_name = ?, dept = ?,
                batch = ?, image = ?, profile_json = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                student_id,
                name.strip(),
                "",
                class_name.strip(),
                dept,
                batch.strip(),
                image_path,
                profile_json,
                student_row_id,
            ),
        )
        conn.commit()
        cursor.execute("SELECT * FROM students WHERE id = ?", (student_row_id,))
        row = cursor.fetchone()
    except Exception as exc:
        conn.close()
        if isinstance(exc, sqlite3.IntegrityError) and "students.student_id" in str(exc):
            raise HTTPException(status_code=400, detail=f"ID {student_id} already exists in {dept}")
        raise HTTPException(status_code=400, detail=f"Unable to update student: {exc}")

    conn.close()
    return {"message": "Student updated successfully", "student": serialize_student(row)}


@app.delete("/students/{student_row_id}")
def delete_student(student_row_id: int):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students WHERE id = ? LIMIT 1", (student_row_id,))
    existing = cursor.fetchone()
    if not existing:
        conn.close()
        raise HTTPException(status_code=404, detail="Student not found")

    cursor.execute("DELETE FROM students WHERE id = ?", (student_row_id,))
    conn.commit()
    conn.close()
    remove_embedding(existing["name"])
    return {"message": "Student deleted successfully"}


@app.post("/students/reassign-ids")
def reassign_student_ids(dept: Optional[str] = None):
    selected_dept = validate_department(dept) if dept else None
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM students")
        rows = cursor.fetchall()

        if not rows:
            return {"message": "No students available for rearranging", "students": []}

        rows = sorted(
            rows,
            key=lambda row: (
                department_sort_key(row["dept"] or ""),
                (row["name"] or "").lower(),
                row["id"],
            ),
        )
        if selected_dept:
            rows = [row for row in rows if validate_department(row["dept"] or "") == selected_dept]

        if not rows:
            return {"message": f"No students found in {selected_dept}", "students": []}

        width = max(2, len(str(len(rows))))

        updated_students = []
        for row in rows:
            temp_student_id = f"{TEMP_ID_PREFIX}{row['id']}"
            cursor.execute(
                "UPDATE students SET student_id = ? WHERE id = ?",
                (temp_student_id, row["id"]),
            )

        department_counters = {}
        for row in rows:
            dept_key = validate_department(row["dept"] or "")
            department_counters[dept_key] = department_counters.get(dept_key, 0) + 1
            new_student_id = f"{department_counters[dept_key]:0{width}d}"
            profile = {}
            if row["profile_json"]:
                try:
                    profile = json.loads(row["profile_json"])
                except Exception:
                    profile = {}
            profile["student_id"] = new_student_id
            profile["name"] = row["name"]
            profile["dob"] = ""
            profile["class_name"] = row["class_name"] or ""
            profile["dept"] = dept_key
            profile["batch"] = row["batch"] or ""
            profile["on_duty"] = bool(row["on_duty"])
            profile["image"] = row["image"] or ""

            cursor.execute(
                """
                UPDATE students
                SET student_id = ?, dept = ?, profile_json = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (new_student_id, dept_key, json.dumps(profile), row["id"]),
            )
            cursor.execute(
                "UPDATE attendance SET student_id = ? WHERE LOWER(student_name) = LOWER(?)",
                (new_student_id, row["name"]),
            )
            updated_students.append(
                {
                    "id": row["id"],
                    "student_id": new_student_id,
                    "name": row["name"],
                    "dept": dept_key,
                }
            )

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    refresh_attendance_csv()
    if selected_dept:
        return {
            "message": f"Student IDs rearranged for {selected_dept} starting from 01",
            "students": updated_students,
        }
    return {
        "message": "Student IDs rearranged for all departments, each department starting from 01",
        "students": updated_students,
    }


@app.get("/attendance")
def attendance(date: Optional[str] = None):
    return get_attendance_records(date)


@app.delete("/attendance/{record_id}")
def delete_attendance(record_id: int):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM attendance WHERE id = ?", (record_id,))
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    refresh_attendance_csv()
    if not deleted:
        raise HTTPException(status_code=404, detail="Attendance record not found")
    return {"message": "Attendance record deleted successfully"}


@app.post("/attendance/{record_id}/toggle-od")
def toggle_attendance_od(record_id: int):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT attendance.id, attendance.student_name, students.id AS student_row_id, students.on_duty
        FROM attendance
        LEFT JOIN students ON LOWER(students.name) = LOWER(attendance.student_name)
        WHERE attendance.id = ?
        LIMIT 1
        """,
        (record_id,),
    )
    row = cursor.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Attendance record not found")
    if row["student_row_id"] is None:
        conn.close()
        raise HTTPException(status_code=404, detail="Student not found for this attendance record")

    next_value = 0 if row["on_duty"] else 1
    cursor.execute(
        "UPDATE students SET on_duty = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (next_value, row["student_row_id"]),
    )
    cursor.execute("SELECT * FROM students WHERE id = ?", (row["student_row_id"],))
    student = cursor.fetchone()
    profile = {}
    if student["profile_json"]:
        try:
            profile = json.loads(student["profile_json"])
        except Exception:
            profile = {}
    profile["student_id"] = student["student_id"] or ""
    profile["name"] = student["name"] or ""
    profile["dob"] = student["dob"] or ""
    profile["class_name"] = student["class_name"] or ""
    profile["dept"] = student["dept"] or ""
    profile["batch"] = student["batch"] or ""
    profile["on_duty"] = bool(next_value)
    profile["image"] = student["image"] or ""
    cursor.execute("UPDATE students SET profile_json = ? WHERE id = ?", (json.dumps(profile), row["student_row_id"]))
    conn.commit()
    conn.close()
    return {
        "message": "On duty enabled" if next_value else "On duty removed",
        "student_name": row["student_name"],
        "on_duty": bool(next_value),
    }


@app.post("/recognize")
async def recognize(
    file: UploadFile = File(...),
    blink_verified: bool = Form(False),
    motion_verified: bool = Form(False),
    camera_label: str = Form(""),
):
    # CHECK IF FACE DATABASE EXISTS
    if not KNOWN_FACES:
        return {
            "status": "searching",
            "reason": "No student face profiles are available.",
            "results": [],
            "low_confidence": [],
        }

    # READ IMAGE
    contents = await file.read()

    frame = cv2.imdecode(
        np.frombuffer(contents, np.uint8),
        cv2.IMREAD_COLOR
    )

    if frame is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image frame"
        )

    # ==============================
    # MOBILE PHONE DETECTION
    # ==============================

    mobile_boxes = detect_mobile(frame)

    if mobile_boxes:
        return {
            "status": "blocked",
            "reason": "Mobile phone detected. Attendance blocked.",
            "results": [],
            "low_confidence": [],
            "mobile_detections": mobile_boxes,
        }

    # ==============================
    # FACE DETECTION
    # ==============================

    faces = FACE_APP.get(frame)

    if not faces:
        return {
            "status": "searching",
            "reason": "Searching face...",
            "results": [],
            "low_confidence": [],
        }

    # ==============================
    # FACE RECOGNITION
    # ==============================

    recognized = []
    low_confidence = []

    for face in faces:

        best_name, best_score = best_face_match(face.embedding)

        payload = build_face_payload(
            face,
            best_name if best_score >= MATCH_THRESHOLD else "Unknown",
            best_score
        )

        if best_score >= MATCH_THRESHOLD:
            recognized.append(payload)
        else:
            low_confidence.append(payload)

    # ==============================
    # NO MATCH FOUND
    # ==============================

    if not recognized:
        return {
            "status": "no_match",
            "reason": "No match found. Low confidence or unknown face.",
            "results": [],
            "low_confidence": low_confidence,
        }

    # ==============================
    # BEST MATCH
    # ==============================

    recognized.sort(
        key=lambda item: item["confidence"],
        reverse=True
    )

    strongest = max(
        recognized,
        key=lambda item: item["confidence"]
    )

    # ==============================
    # LIVENESS CHECK
    # ==============================

    live_verified = bool(
        blink_verified and motion_verified
    )

    if not live_verified:
        return {
            "status": "awaiting_blink",
            "reason": (
                f"Face matched for {strongest['name']}. "
                f"Complete 2 eye blinks and 70% motion for live check."
            ),
            "results": recognized,
            "low_confidence": low_confidence,
        }

    # ==============================
    # MARK ATTENDANCE
    # ==============================

    strongest["attendance"] = mark_attendance(
        strongest["name"],
        strongest["confidence"] / 100,
        camera_label,
    )

    return {
        "status": strongest["attendance"]["status"],
        "reason": (
            f"{strongest['name']} verified "
            f"with 2 blinks and 70% motion."
        ),
        "results": recognized,
        "low_confidence": low_confidence,
    }


@app.get("/export-pdf")
def export_pdf(dept: Optional[str] = None):
    selected_dept = validate_department(dept) if dept and dept.strip().lower() != "all" else None
    sections = build_hour_wise_report_rows_by_department(selected_dept)
    styles = getSampleStyleSheet()

    pdf = SimpleDocTemplate(
        ATTENDANCE_PDF,
        pagesize=landscape(A4),
        leftMargin=24,
        rightMargin=24,
        topMargin=24,
        bottomMargin=24,
    )

    story = []
    for index, section in enumerate(sections):
        if index > 0:
            story.append(PageBreak())

        story.append(Paragraph(f"Department: {section['department']}", styles["Title"]))
        story.append(Spacer(1, 12))

        table = Table(section["rows"], repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#173d89")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.75, colors.black),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#eaf1ff")]),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.append(table)

    pdf.build(story)
    return FileResponse(ATTENDANCE_PDF, media_type="application/pdf", filename="attendance_report.pdf")
