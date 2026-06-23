import os
import cv2
import numpy as np
import face_recognition
 
# ── CONFIG ────────────────────────────────────────────────────────────────────
 
KNOWN_FACES_DIR = "known_faces"   # folder of reference photos
THRESHOLD = 0.55                  # lower = stricter matching (0.4–0.6 is typical)
SCALE = 0.5                       # resize frame before detection (speeds things up)
MODEL = "hog"                     # "hog" (fast, CPU) or "cnn" (accurate, needs GPU)
 
# ── COLOURS (BGR) ─────────────────────────────────────────────────────────────
 
COL_KNOWN   = (0, 220, 100)   # green  — recognised face
COL_UNKNOWN = (0, 80, 220)    # red    — unknown face
COL_TEXT_BG = (20, 20, 20)    # dark label background
FONT        = cv2.FONT_HERSHEY_SIMPLEX
 
# ── LOAD KNOWN FACES ──────────────────────────────────────────────────────────
 
def load_known_faces(directory: str):
    """
    Scan a directory for images. For each image:
      - detect the first face
      - compute its 128-d embedding
      - store under the filename (minus extension) as the person's name
 
    Returns:
        known_encodings : list of np.ndarray (128,)
        known_names     : list of str
    """
    known_encodings = []
    known_names = []
 
    if not os.path.isdir(directory):
        print(f"[WARN] '{directory}' not found — running in detection-only mode.")
        return known_encodings, known_names
 
    supported = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = [f for f in os.listdir(directory) if f.lower().endswith(supported)]
 
    if not files:
        print(f"[WARN] No images found in '{directory}'.")
        return known_encodings, known_names
 
    print(f"Loading {len(files)} reference image(s)...")
 
    for filename in files:
        path = os.path.join(directory, filename)
        name = os.path.splitext(filename)[0].replace("_", " ").title()
 
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
 
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(name)
            print(f"  ✓ Loaded: {name}")
        else:
            print(f"  ✗ No face found in {filename}, skipping.")
 
    return known_encodings, known_names
 
 
# ── RECOGNITION ───────────────────────────────────────────────────────────────
 
def identify(encoding, known_encodings, known_names):
    """
    Compare a face encoding against all known encodings.
 
    face_recognition.face_distance returns a Euclidean distance in
    128-dimensional space. Lower = more similar.
 
    Returns (name, distance) — name is "Unknown" if no match under threshold.
    """
    if not known_encodings:
        return "Unknown", None
 
    distances = face_recognition.face_distance(known_encodings, encoding)
    best_idx = np.argmin(distances)
    best_dist = distances[best_idx]
 
    if best_dist < THRESHOLD:
        return known_names[best_idx], best_dist
    return "Unknown", best_dist
 
 
# ── DRAWING ───────────────────────────────────────────────────────────────────
 
def draw_face_box(frame, top, right, bottom, left, name, distance, colour):
    """Draw a bounding box and label on the frame."""
    # Box
    cv2.rectangle(frame, (left, top), (right, bottom), colour, 2)
 
    # Label text
    if distance is not None:
        label = f"{name}  ({distance:.2f})"
    else:
        label = name
 
    # Label background pill
    (text_w, text_h), baseline = cv2.getTextSize(label, FONT, 0.55, 1)
    label_top = bottom + 4
    cv2.rectangle(
        frame,
        (left, label_top),
        (left + text_w + 8, label_top + text_h + baseline + 6),
        COL_TEXT_BG,
        cv2.FILLED,
    )
    cv2.putText(
        frame,
        label,
        (left + 4, label_top + text_h + 3),
        FONT,
        0.55,
        colour,
        1,
        cv2.LINE_AA,
    )
 
 
def draw_hud(frame, fps, face_count):
    """Overlay FPS and face count in the top-left corner."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (200, 48), COL_TEXT_BG, cv2.FILLED)
    cv2.putText(frame, f"FPS: {fps:.1f}", (8, 18), FONT, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Faces: {face_count}", (8, 38), FONT, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
 
 
# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
 
def main():
    known_encodings, known_names = load_known_faces(KNOWN_FACES_DIR)
 
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check it's connected and not in use.")
 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
 
    print("\nRunning — press Q to quit.\n")
 
    prev_time = cv2.getTickCount()
 
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame grab failed — exiting.")
            break
 
        # ── FPS ───────────────────────────────────────────────────────────────
        now = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (now - prev_time)
        prev_time = now
 
        # ── DETECTION ─────────────────────────────────────────────────────────
        # Downscale for speed, then convert BGR → RGB (face_recognition wants RGB)
        small = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
 
        # Detect bounding boxes and compute embeddings
        locations = face_recognition.face_locations(rgb_small, model=MODEL)
        encodings = face_recognition.face_encodings(rgb_small, locations)
 
        # ── RECOGNITION + DRAWING ─────────────────────────────────────────────
        for (top, right, bottom, left), encoding in zip(locations, encodings):
            # Scale coordinates back up to original frame size
            top    = int(top    / SCALE)
            right  = int(right  / SCALE)
            bottom = int(bottom / SCALE)
            left   = int(left   / SCALE)
 
            name, distance = identify(encoding, known_encodings, known_names)
            colour = COL_KNOWN if name != "Unknown" else COL_UNKNOWN
            draw_face_box(frame, top, right, bottom, left, name, distance, colour)
 
        draw_hud(frame, fps, len(locations))
 
        cv2.imshow("Facial Recognition  |  Q to quit", frame)
 
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
 
    cap.release()
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()