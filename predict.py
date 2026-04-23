import cv2
import numpy as np
import tensorflow as tf
import os
import pyttsx3
import time

# ======================================
# 1️⃣ SETTINGS
# ======================================
MODEL_PATH = 'model_trained\\final_sign_model.keras'
DATASET_DIR = 'dataset/images'
IMG_SIZE = 128
CONFIDENCE_THRESHOLD = 0.85   # require higher confidence
STABLE_THRESHOLD = 6
SPEAK_DELAY = 2.0
ROI_SIZE = 300
MIN_MOTION_DIFF = 40000       # minimum frame difference to consider ROI “active”

# ======================================
# 2️⃣ LOAD MODEL & LABELS
# ======================================
model = tf.keras.models.load_model(MODEL_PATH)
labels = sorted(os.listdir(DATASET_DIR))
print(f"✅ Loaded model with {len(labels)} classes.")

# ======================================
# 3️⃣ CAMERA & TTS
# ======================================
cap = cv2.VideoCapture(0)
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

# ======================================
# 4️⃣ STATE VARIABLES
# ======================================
word = ''
last_letter_time = time.time()
spoken = False
stable_letter = None
stable_count = 0
prev_roi_gray = None

print("\n🎯 Instructions:")
print(" - Keep your hand in the green box.")
print(" - Press 'c' to clear the word.")
print(" - Press 's' to insert SPACE.")
print(" - Press 'q' to quit.\n")

# ======================================
# 5️⃣ MAIN LOOP
# ======================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Define fixed ROI in center
    cx, cy = w // 2, h // 2 - 50
    x1, y1 = cx - ROI_SIZE // 2, cy - ROI_SIZE // 2
    x2, y2 = cx + ROI_SIZE // 2, cy + ROI_SIZE // 2
    roi = frame[y1:y2, x1:x2]

    # Draw ROI
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Motion check — detect if hand is moving
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.GaussianBlur(roi_gray, (7, 7), 0)
    motion_detected = False

    if prev_roi_gray is not None:
        diff = cv2.absdiff(roi_gray, prev_roi_gray)
        motion_score = np.sum(diff)
        if motion_score > MIN_MOTION_DIFF:
            motion_detected = True
    prev_roi_gray = roi_gray

    predicted_letter = ''
    confidence = 0.0

    if motion_detected:  # Only predict if ROI changes
        # Preprocess ROI
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        img = cv2.resize(roi_rgb, (IMG_SIZE, IMG_SIZE))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict
        pred = model.predict(img, verbose=0)[0]
        confidence = float(np.max(pred))
        index = int(np.argmax(pred))

        if confidence >= CONFIDENCE_THRESHOLD:
            candidate_letter = labels[index]

            # Stability logic
            if stable_letter == candidate_letter:
                stable_count += 1
            else:
                stable_letter = candidate_letter
                stable_count = 1

            # Confirm when stable
            if stable_count >= STABLE_THRESHOLD:
                if len(word) == 0 or word[-1] != stable_letter:
                    word += stable_letter
                    last_letter_time = time.time()
                    spoken = False
                    print(f"✅ Added Letter: {stable_letter} | Conf: {confidence:.2f}")
                stable_letter = None
                stable_count = 0
        else:
            stable_letter = None
            stable_count = 0
    else:
        stable_letter = None
        stable_count = 0

    # =======================
    # 🖼 DISPLAY INFO
    # =======================
    cv2.putText(frame, f"Letter: {stable_letter if stable_letter else '-'}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    cv2.putText(frame, f"Conf: {confidence:.2f}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Word: {word}", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    cv2.imshow("Sign Detection", frame)

    # =======================
    # 🔊 SPEAK WORD
    # =======================
    if word and (time.time() - last_letter_time > SPEAK_DELAY) and not spoken:
        print(f"🔊 Speaking: {word}")
        engine.say(word)
        engine.runAndWait()
        spoken = True

    # =======================
    # ⌨️ KEYS
    # =======================
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        word = ''
        spoken = False
        stable_letter = None
        stable_count = 0
        print("🧹 Word cleared\n")
    elif key == ord('s'):
        word += ' '
        spoken = False
        print("➕ Added space\n")

cap.release()
cv2.destroyAllWindows()
