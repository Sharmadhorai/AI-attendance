import cv2
import pickle
import numpy as np
import os
import time
from datetime import datetime
from insightface.app import FaceAnalysis

# 📁 Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
attendance_file = os.path.join(BASE_DIR, "attendance.csv")
embeddings_file = os.path.join(BASE_DIR, "embeddings.pkl")

# 🧠 Load embeddings
with open(embeddings_file, "rb") as f:
    known_faces = pickle.load(f)

# 📝 Attendance
marked = set()
last_seen = {}

COOLDOWN = 5

# 📄 CSV
if not os.path.exists(attendance_file):
    with open(attendance_file, "w") as f:
        f.write("Name,Time\n")

def mark_attendance(name):
    if name not in marked and name != "Unknown":
        with open(attendance_file, "a") as f:
            time_now = datetime.now().strftime("%H:%M:%S")
            f.write(f"{name},{time_now}\n")
        marked.add(name)
        print(f"✅ Marked: {name}")

# 📐 Similarity
def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return -1
    return np.dot(a, b) / (norm_a * norm_b)

# 🚀 InsightFace (FAST)
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=-1, det_size=(320, 320))

# 🎥 Camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 640)
cap.set(4, 480)

frame_skip = 5
frame_count = 0

# 🔥 STORE LAST RESULTS (ANTI-BLINK)
last_faces = []

print("🚀 ULTRA SMOOTH AI ATTENDANCE STARTED")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # 🔥 Only detect every few frames
    if frame_count % frame_skip == 0:

        frame_small = cv2.resize(frame, (480, 360))
        faces = app.get(frame_small)

        faces = faces[:2]

        current_faces = []

        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)

            if (x2 - x1) < 80 or (y2 - y1) < 80:
                continue

            # Scale
            scale_x = frame.shape[1] / 480
            scale_y = frame.shape[0] / 360

            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            # Padding
            pad = 15
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(frame.shape[1], x2 + pad)
            y2 = min(frame.shape[0], y2 + pad)

            emb = face.embedding

            best_score = -1
            best_name = "Unknown"

            for data in known_faces:
                score = cosine_similarity(emb, data["embedding"])
                if score > best_score:
                    best_score = score
                    best_name = data["name"]

            if best_score > 0.45:
                name = best_name

                current_time = time.time()
                if name not in last_seen or (current_time - last_seen[name]) > COOLDOWN:
                    mark_attendance(name)
                    last_seen[name] = current_time
            else:
                name = "Unknown"

            current_faces.append((x1, y1, x2, y2, name))

        # 🔥 Save last faces (for smooth display)
        last_faces = current_faces

    # 🔥 DRAW FROM LAST (NO BLINK)
    for (x1, y1, x2, y2, name) in last_faces:
        color = (0,255,0) if name != "Unknown" else (0,0,255)

        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, name, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("ULTRA SMOOTH ATTENDANCE", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()