from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response, FileResponse
from fastapi.middleware.cors import CORSMiddleware

import cv2
import numpy as np
import os
import pickle
from datetime import datetime

from insightface.app import FaceAnalysis
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors

app = FastAPI()

# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📁 Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
attendance_file = os.path.join(BASE_DIR, "attendance.csv")
embeddings_file = os.path.join(BASE_DIR, "embeddings.pkl")

# 🧠 Load embeddings
with open(embeddings_file, "rb") as f:
    known_faces = pickle.load(f)

# 🧠 Initialize InsightFace (SAFE)
face_app = FaceAnalysis(name="buffalo_l")

try:
    face_app.prepare(ctx_id=0, det_size=(640, 640))  # GPU
    print("🚀 Using GPU")
except:
    face_app.prepare(ctx_id=-1, det_size=(640, 640))  # CPU fallback
    print("⚠️ Using CPU")

# 📝 Attendance memory
marked = set()

# 📄 Create CSV if not exists
if not os.path.exists(attendance_file):
    with open(attendance_file, "w") as f:
        f.write("Name,Time\n")

# 📝 Mark attendance
def mark_attendance(name):
    if name not in marked and name != "Unknown":
        with open(attendance_file, "a") as f:
            time_now = datetime.now().strftime("%H:%M:%S")
            f.write(f"{name},{time_now}\n")
        marked.add(name)
        print(f"✅ Marked: {name}")

# 📐 Cosine similarity
def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return -1  # invalid similarity

    return np.dot(a, b) / (norm_a * norm_b)

# 🎥 FACE RECOGNITION API
@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 🔥 Detect faces
    faces = face_app.get(frame)

    print("Faces detected:", len(faces))

    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        emb = face.embedding

        best_score = -1
        best_name = "Unknown"

        # 🔍 Compare with known faces
        for data in known_faces:
            score = cosine_similarity(emb, data["embedding"])

            if score > best_score:
                best_score = score
                best_name = data["name"]

        print("DEBUG:", best_name, best_score)

        # 🎯 Threshold tuning
        if best_score > 0.45:
            name = best_name
            mark_attendance(name)
        else:
            name = "Unknown"

        # 🎨 Draw UI
        if name == "Unknown":
            color = (0, 0, 255)
            label = "Unknown"
        else:
            color = (0, 255, 0)
            label = f"{name} - Present"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(frame, (x1, y1 - 30), (x2, y1), color, -1)

        cv2.putText(
            frame,
            label,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )

    # 📤 Return processed frame
    _, img_encoded = cv2.imencode(".jpg", frame)

    return Response(
        content=img_encoded.tobytes(),
        media_type="image/jpeg"
    )

# 📋 ATTENDANCE API
@app.get("/attendance")
def get_attendance():
    data = []

    if os.path.exists(attendance_file):
        with open(attendance_file, "r") as f:
            lines = f.readlines()[1:]

            for line in lines:
                try:
                    name, time = line.strip().split(",")
                    data.append({"name": name, "time": time})
                except:
                    continue

    return data

# 📄 EXPORT PDF
@app.get("/export-pdf")
def export_pdf():

    file_path = os.path.join(BASE_DIR, "attendance_report.pdf")

    data = [["Name", "Time"]]

    if os.path.exists(attendance_file):
        with open(attendance_file, "r") as f:
            lines = f.readlines()[1:]

            for line in lines:
                try:
                    name, time = line.strip().split(",")
                    data.append([name, time])
                except:
                    continue

    pdf = SimpleDocTemplate(file_path)
    table = Table(data)

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 1, colors.black)
    ]))

    pdf.build([table])

    return FileResponse(
        file_path,
        media_type="application/pdf",
        filename="attendance.pdf"
    )

