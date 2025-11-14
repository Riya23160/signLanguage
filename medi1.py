# ==========================
# PHASE 2 — REAL-TIME PREDICTION (MULTI-HAND FIXED)
# ==========================
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pyttsx3
import os
import time

# -----------------------------
# SETTINGS
# -----------------------------
MODEL_PATH = "final_sign_model.keras"
DATASET_DIR = "data_processed"
IMG_SIZE = 128
CONFIDENCE_THRESHOLD = 0.80
STABLE_THRESHOLD = 5
SPEAK_DELAY = 2.5

# -----------------------------
# LOAD MODEL & LABELS
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH)
labels = sorted(os.listdir(DATASET_DIR))
print(f"✅ Loaded model with {len(labels)} classes.")

# -----------------------------
# INITIALIZE TOOLS
# -----------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
engine = pyttsx3.init()
engine.setProperty('rate', 150)

word = ''
stable_letter = None
stable_count = 0
spoken = False
last_letter_time = time.time()

print("\n🖐 Instructions:")
print(" - Show your hand clearly in front of camera (avoid face in view).")
print(" - Supports detection for both hands.")
print(" - Press 's' to add a SPACE.")
print(" - Press 'c' to clear.")
print(" - Press 'q' to quit.\n")

# -----------------------------
# MAIN LOOP
# -----------------------------
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # 🔹 Track both hands
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        h, w, _ = frame.shape
        predicted_letter = ""
        confidence = 0.0

        # Process each detected hand
        if res.multi_hand_landmarks:
            for hand_landmarks in res.multi_hand_landmarks:
                # Get bounding box for the hand
                x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
                y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
                x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
                y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)
                margin = 30
                x_min, y_min = max(0, x_min - margin), max(0, y_min - margin)
                x_max, y_max = min(w, x_max + margin), min(h, y_max + margin)

                roi = frame[y_min:y_max, x_min:x_max]
                if roi.size == 0:
                    continue

                roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
                roi_norm = roi_resized.astype("float32") / 255.0
                pred = model.predict(np.expand_dims(roi_norm, axis=0), verbose=0)[0]
                conf = float(np.max(pred))
                label = labels[int(np.argmax(pred))]

                # Pick the hand with higher confidence
                if conf > confidence:
                    confidence = conf
                    predicted_letter = label

                # Draw bounding box and landmarks
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Stabilization and confidence logic
        if predicted_letter and confidence > CONFIDENCE_THRESHOLD:
            if stable_letter == predicted_letter:
                stable_count += 1
            else:
                stable_letter = predicted_letter
                stable_count = 1

            if stable_count >= STABLE_THRESHOLD:
                if len(word) == 0 or word[-1] != stable_letter:
                    word += stable_letter
                    print(f"✅ Added {stable_letter} (Conf: {confidence:.2f})")
                    last_letter_time = time.time()
                    spoken = False
                stable_count = 0
                stable_letter = None
        else:
            stable_count = 0

        # Display predictions
        cv2.putText(frame, f"Letter: {predicted_letter if predicted_letter else '-'}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Conf: {confidence:.2f}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Word: {word}", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Sign Detection (Both Hands)", frame)

        # Auto TTS with cooldown
        if word and (time.time() - last_letter_time > SPEAK_DELAY) and not spoken:
            print(f"🔊 Speaking: {word}")
            engine.say(word)
            engine.runAndWait()
            spoken = True

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            word = ''
            print("🧹 Cleared word")
        elif key == ord('s'):
            word += ' '
            print("➕ Space added")

cap.release()
cv2.destroyAllWindows()
