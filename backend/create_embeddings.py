import os
import pickle
import cv2
from insightface.app import FaceAnalysis

# 📁 Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, "dataset")
embeddings_file = os.path.join(BASE_DIR, "embeddings.pkl")

# 🚀 Initialize InsightFace (SAFE)
app = FaceAnalysis(name="buffalo_l")

try:
    app.prepare(ctx_id=0, det_size=(640, 640))  # GPU
    print("🚀 Using GPU")
except:
    app.prepare(ctx_id=-1, det_size=(640, 640))  # CPU fallback
    print("⚠️ Using CPU")

embeddings = []

print("📁 Dataset path:", dataset_path)

# 🔁 Loop through dataset
for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)

    if not os.path.isdir(person_path):
        continue

    print(f"\n👤 Processing: {person}")

    for img_name in os.listdir(person_path):

        # ✅ Skip non-image files
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(person_path, img_name)

        try:
            img = cv2.imread(img_path)

            if img is None:
                print(f"❌ Failed to load: {img_name}")
                continue

            faces = app.get(img)

            if len(faces) > 0:
                # ✅ Select largest face
                face = max(
                    faces,
                    key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
                )

                emb = face.embedding

                embeddings.append({
                    "name": person,
                    "embedding": emb
                })

                print(f"✅ Processed: {img_name}")

            else:
                print(f"⚠️ No face found: {img_name}")

        except Exception as e:
            print(f"❌ Error processing: {img_name}")
            print("Reason:", e)

# 💾 Save embeddings
with open(embeddings_file, "wb") as f:
    pickle.dump(embeddings, f)

print(f"\n🔥 Total embeddings saved: {len(embeddings)}")