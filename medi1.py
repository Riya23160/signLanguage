# ==========================
# PHASE 2 — REAL-TIME PREDICTION (MULTI-HAND + RELIABLE TTS USING SUBPROCESS)
# Copy-paste this file and run it. Tested approach: uses PowerShell on Windows.
# ==========================
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os
import time
import subprocess
import sys
import json

# -----------------------------
# SETTINGS
# -----------------------------
MODEL_PATH = "final_sign_model.keras"   # path to your trained model
DATASET_DIR = "data_processed"          # folder with classes subfolders
IMG_SIZE = 128
CONFIDENCE_THRESHOLD = 0.80
STABLE_THRESHOLD = 5
LETTER_COOLDOWN = 0.25   # small guard to avoid immediate repeats (seconds)

# -----------------------------
# SIMPLE CROSS-PLATFORM TTS (NO pyttsx3)
# - launches a system process to speak text
# - kills previous TTS process (if still running) before starting new one
# -----------------------------
current_tts_proc = None

def speak_nonblocking(text, urgent=True):
    """
    Speak `text` using platform TTS. Non-blocking: returns subprocess object.
    If a previous TTS process is still running, terminate it first (for urgent).
    This avoids pyttsx3's hang-in-loop problems.
    """
    global current_tts_proc

    # sanitize text for embedding in PowerShell / shell
    if not isinstance(text, str):
        text = str(text)

    # terminate old process if still running
    try:
        if current_tts_proc is not None:
            # check if still running
            if current_tts_proc.poll() is None:
                try:
                    current_tts_proc.terminate()
                except Exception:
                    pass
            current_tts_proc = None
    except Exception:
        current_tts_proc = None

    # Platform-specific commands
    if sys.platform.startswith("win"):
        # Use PowerShell System.Speech with proper escaping using json.dumps
        # json.dumps returns a double-quoted, escaped string safe to embed.
        quoted = json.dumps(text)
        cmd = [
            "powershell",
            "-NoProfile",
            "-Command",
            f"Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Speak({quoted})"
        ]
        # CREATE_NO_WINDOW prevents new console windows on Windows
        creationflags = 0
        try:
            creationflags = subprocess.CREATE_NO_WINDOW
        except Exception:
            creationflags = 0
        try:
            current_tts_proc = subprocess.Popen(cmd, creationflags=creationflags)
            return current_tts_proc
        except Exception as e:
            print("TTS start error (PowerShell):", e)
            current_tts_proc = None
            return None

    elif sys.platform == "darwin":
        # macOS 'say' command
        cmd = ["say", text]
        try:
            current_tts_proc = subprocess.Popen(cmd)
            return current_tts_proc
        except Exception as e:
            print("TTS start error (macOS say):", e)
            current_tts_proc = None
            return None

    else:
        # Linux fallback: try 'spd-say' then 'espeak'
        for prog in (["spd-say", text], ["espeak", text]):
            try:
                current_tts_proc = subprocess.Popen(prog)
                return current_tts_proc
            except FileNotFoundError:
                continue
            except Exception as e:
                print("TTS start error (linux):", e)
                current_tts_proc = None
                return None
        print("No system TTS found. Install 'spd-say' or 'espeak' or use Windows PowerShell.")
        return None

# -----------------------------
# LOAD MODEL & LABELS
# -----------------------------
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
labels = sorted(os.listdir(DATASET_DIR))
print(f"✅ Loaded model with {len(labels)} classes.")

# -----------------------------
# INIT MEDIAPIPE, CAMERA
# -----------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not detected. Check your camera index.")
    sys.exit(1)

# -----------------------------
# STATE
# -----------------------------
stable_letter = None
stable_count = 0
last_spoken_letter = ""
last_spoken_time = 0.0
word = ""

print("\n🖐 Instructions:")
print(" - Show your hand clearly in view (supports both hands).")
print(" - Press 's' to add a SPACE.")
print(" - Press 'c' to clear the current word (and reset TTS).")
print(" - Press 'q' to quit.\n")

# -----------------------------
# MAIN LOOP
# -----------------------------
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Camera read failed.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        h, w, _ = frame.shape
        predicted_letter = ""
        confidence = 0.0

        # Process detected hands (choose highest-confidence prediction across hands)
        if res.multi_hand_landmarks:
            for hand_landmarks in res.multi_hand_landmarks:
                # compute bounding box
                xs = [lm.x for lm in hand_landmarks.landmark]
                ys = [lm.y for lm in hand_landmarks.landmark]
                x_min = int(max(0, min(xs) * w) - 30)
                y_min = int(max(0, min(ys) * h) - 30)
                x_max = int(min(w, max(xs) * w) + 30)
                y_max = int(min(h, max(ys) * h) + 30)

                # crop ROI
                roi = frame[y_min:y_max, x_min:x_max]
                if roi.size == 0:
                    continue

                roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
                roi_norm = roi_resized.astype("float32") / 255.0
                pred = model.predict(np.expand_dims(roi_norm, axis=0), verbose=0)[0]
                conf = float(np.max(pred))
                label = labels[int(np.argmax(pred))]

                if conf > confidence:
                    confidence = conf
                    predicted_letter = label

                # draw bounding box & landmarks
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Stabilize predicted_letter
        if predicted_letter and confidence > CONFIDENCE_THRESHOLD:
            if stable_letter == predicted_letter:
                stable_count += 1
            else:
                stable_letter = predicted_letter
                stable_count = 1

            if stable_count >= STABLE_THRESHOLD:
                # Append to word (avoid duplicates repeating)
                if len(word) == 0 or word[-1] != stable_letter:
                    word += stable_letter
                    print(f"✅ Added {stable_letter} (Conf: {confidence:.2f})")

                    # Determine if we should speak this new letter now
                    now = time.time()
                    # Simple cooldown to avoid immediate repeats
                    if stable_letter != last_spoken_letter or (now - last_spoken_time > LETTER_COOLDOWN):
                        # speak non-blocking (kills any previous TTS process)
                        speak_nonblocking(stable_letter, urgent=True)
                        last_spoken_letter = stable_letter
                        last_spoken_time = now

                # reset stability for next letter
                stable_count = 0
                stable_letter = None
        else:
            stable_count = 0

        # Display
        cv2.putText(frame, f"Letter: {predicted_letter if predicted_letter else '-'}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Conf: {confidence:.2f}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Word: {word}", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Sign Detection (Both Hands)", frame)

        # Key controls
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            word = ''
            last_spoken_letter = ''
            # terminate tts process if running
            try:
                if current_tts_proc is not None and current_tts_proc.poll() is None:
                    current_tts_proc.terminate()
            except Exception:
                pass
            print("🧹 Cleared word and stopped TTS")
        elif key == ord('s'):
            word += ' '
            print("➕ Space added")

# Cleanup
cap.release()
cv2.destroyAllWindows()
# Ensure TTS subprocess is terminated
try:
    if current_tts_proc is not None and current_tts_proc.poll() is None:
        current_tts_proc.terminate()
except Exception:
    pass

print("Program exited cleanly.")
