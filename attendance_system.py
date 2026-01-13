import os
import cv2
import csv
import pickle
import numpy as np
from datetime import datetime
from scipy.spatial.distance import cosine
from insightface.app import FaceAnalysis

# =========================
# CONFIG
# =========================
ENROLL_DIR = "enroll"               # enrollment folder
FACE_DB_PATH = "face_db.pkl"
ATTENDANCE_CSV = "attendance.csv"

CAMERA_SOURCE = r"C:\Users\Anjali Rajora\OneDrive\Documents\attendance_system\video-1\WIN_20260110_19_40_35_Pro.mp4"                 # 0 = webcam, or RTSP URL
OUTPUT_VIDEO = "output_attendance.mp4"

SIM_THRESHOLD = 0.45               # cosine similarity threshold
MIN_INTERVAL_SEC = 300             # 5 minutes cooldown
DET_SIZE = (640, 640)
SAVE_FPS = 20

# =========================
# INIT INSIGHTFACE (BUFFALO)
# =========================
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=DET_SIZE)

# =========================
# UTILS
# =========================
def cosine_similarity(a, b):
    return 1 - cosine(a, b)

# =========================
# ENROLLMENT
# =========================
def enroll_faces():
    face_db = {}
    print("🔹 Starting enrollment...")

    for person in os.listdir(ENROLL_DIR):
        person_path = os.path.join(ENROLL_DIR, person)
        if not os.path.isdir(person_path):
            continue

        embeddings = []

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            faces = app.get(img)
            if len(faces) == 1:
                embeddings.append(faces[0].embedding)

        if embeddings:
            mean_emb = np.mean(embeddings, axis=0)
            mean_emb /= np.linalg.norm(mean_emb)
            face_db[person] = mean_emb
            print(f"✅ Enrolled {person} ({len(embeddings)} images)")
        else:
            print(f"⚠️ No valid face found for {person}")

    with open(FACE_DB_PATH, "wb") as f:
        pickle.dump(face_db, f)

    print("✅ Enrollment completed")
    return face_db

# =========================
# LOAD FACE DATABASE
# =========================
def load_face_db():
    if not os.path.exists(FACE_DB_PATH):
        return enroll_faces()

    with open(FACE_DB_PATH, "rb") as f:
        return pickle.load(f)

# =========================
# FACE RECOGNITION
# =========================
def recognize_face(query_emb, db):
    best_name = "Unknown"
    best_score = 0.0

    for name, ref_emb in db.items():
        score = cosine_similarity(query_emb, ref_emb)
        if score > best_score:
            best_score = score
            best_name = name

    if best_score > SIM_THRESHOLD:
        return best_name, best_score
    return "Unknown", best_score

# =========================
# ATTENDANCE LOGIC
# =========================
last_seen = {}

def mark_attendance(name, confidence):
    now = datetime.now()

    if name in last_seen:
        if (now - last_seen[name]).seconds < MIN_INTERVAL_SEC:
            return

    last_seen[name] = now
    entry_type = "IN" if now.hour < 15 else "OUT"

    file_exists = os.path.exists(ATTENDANCE_CSV)

    with open(ATTENDANCE_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["name", "date", "time", "type", "confidence"])

        writer.writerow([
            name,
            now.date(),
            now.strftime("%H:%M:%S"),
            entry_type,
            round(confidence, 3)
        ])

    print(f"📝 Attendance marked: {name} | {entry_type} | {confidence:.2f}")

# =========================
# MAIN
# =========================
def main():
    face_db = load_face_db()
    print(f"📦 Loaded {len(face_db)} identities")

    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        print("❌ Camera not accessible")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = SAVE_FPS

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        OUTPUT_VIDEO,
        fourcc,
        fps,
        (width, height)
    )

    print("🎥 Recording started →", OUTPUT_VIDEO)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = app.get(frame)

        for face in faces:
            name, score = recognize_face(face.embedding, face_db)

            if name != "Unknown":
                mark_attendance(name, score)

            x1, y1, x2, y2 = face.bbox.astype(int)
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{name} {score:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        out.write(frame)

    cap.release()
    out.release()
    print("✅ Video saved successfully")

# =========================
# RUN
# =========================
if __name__ == "__main__":
    main()