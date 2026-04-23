import cv2
import numpy as np
import tensorflow as tf
import pyttsx3
import time
import mediapipe as mp
import threading
from ui_layout import ResponsiveUI


MODEL_PATH = 'model_trained/final_sign_model.keras'
IMG_SIZE = 128
CONFIDENCE_THRESHOLD = 0.75
STABLE_THRESHOLD = 4
SPEAK_DELAY = 2.0
EMA_ALPHA = 0.2

model = tf.keras.models.load_model(MODEL_PATH)
label_map = np.load("label_map.npy", allow_pickle=True).item()
labels = {v: k for k, v in label_map.items()}

print(f"✅ Model loaded with {len(labels)} classes.")


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6
)

LANDMARK_SPEC = mp_drawing.DrawingSpec(
    color=(210, 130, 255),   
    thickness=2,
    circle_radius=4
)
CONNECTION_SPEC = mp_drawing.DrawingSpec(
    color=(240, 240, 255),   
    thickness=1
)


def speak_async(text):
    try:
        _engine = pyttsx3.init()
        _engine.setProperty('rate', 150)
        _engine.setProperty('volume', 1.0)
        _engine.say(text)
        _engine.runAndWait()
        _engine.stop()
    except Exception as e:
        print(f"TTS error: {e}")


stable_letter    = None
stable_count     = 0
word             = ''
last_letter_time = time.time()
spoken           = False
smoothed_pred    = None
gap              = 0.0


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("Sign Language Detector", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Sign Language Detector",
                       cv2.WND_PROP_FULLSCREEN,
                       cv2.WINDOW_FULLSCREEN)

print("\n🎯 Controls:  C=clear   S=space   Q=quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()
    H, W, _ = frame.shape

    # ── Hand detection ─────────────────────────────────────
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands_detector.process(rgb)

    img        = None
    index      = None
    confidence = 0.0
    gap        = 0.0
    hand_count = 0

    if res.multi_hand_landmarks:
        hand_count = len(res.multi_hand_landmarks)

        for hand_lm in res.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                display_frame, hand_lm,
                mp_hands.HAND_CONNECTIONS,
                LANDMARK_SPEC, CONNECTION_SPEC
            )

        crops = []
        for hand_lm in res.multi_hand_landmarks:
            x_min = int(min(lm.x for lm in hand_lm.landmark) * W) - 30
            x_max = int(max(lm.x for lm in hand_lm.landmark) * W) + 30
            y_min = int(min(lm.y for lm in hand_lm.landmark) * H) - 30
            y_max = int(max(lm.y for lm in hand_lm.landmark) * H) + 30

            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(W, x_max), min(H, y_max)

            crop = frame[y_min:y_max, x_min:x_max]
            if crop.size > 0:
                crops.append(cv2.resize(crop, (IMG_SIZE, IMG_SIZE)))

        if len(crops) == 2:
            img = cv2.resize(np.hstack(crops), (IMG_SIZE, IMG_SIZE))
        elif len(crops) == 1:
            img = crops[0]

    if img is not None:
        inp = img.astype('float32') / 255.0
        inp = np.expand_dims(inp, axis=0)

        pred = model.predict(inp, verbose=0)[0]

        if smoothed_pred is None:
            smoothed_pred = pred
        else:
            smoothed_pred = EMA_ALPHA * pred + (1 - EMA_ALPHA) * smoothed_pred

        sorted_pred = np.sort(smoothed_pred)
        confidence  = float(sorted_pred[-1])
        gap         = confidence - float(sorted_pred[-2])
        index       = int(np.argmax(smoothed_pred))
    else:
        smoothed_pred = None
        stable_letter = None
        stable_count  = 0


    if index is not None and confidence >= CONFIDENCE_THRESHOLD and gap > 0.2:
        candidate = labels[index]

        if stable_letter == candidate:
            stable_count += 1
        else:
            stable_letter = candidate
            stable_count  = 1

        if stable_count >= STABLE_THRESHOLD:
            if len(word) == 0 or word[-1] != stable_letter:
                word += stable_letter
                last_letter_time = time.time()
                spoken = False

            stable_letter = None
            stable_count  = 0
    else:
        stable_count = max(0, stable_count - 1)
        if stable_count == 0:
            stable_letter = None

    ui = ResponsiveUI(display_frame)
    ui.top_bar(hand_count=hand_count)
    ui.left_panel(
        letter       = stable_letter if stable_letter else "-",
        confidence   = confidence,
        stable_count = stable_count,
        stable_max   = STABLE_THRESHOLD
    )
    ui.bottom_bar(word=word, spoken=spoken)

    cv2.imshow("Sign Language Detector", display_frame)

    if word and (time.time() - last_letter_time > SPEAK_DELAY) and not spoken:
        threading.Thread(target=speak_async, args=(word,), daemon=True).start()
        spoken = True


    key = cv2.waitKey(5)
    if key == ord('q'):
        print("Quitting...")
        break
    elif key == ord('c'):
        print("Cleared")
        word = ''; stable_letter = None; stable_count = 0; spoken = False
    elif key == ord('s'):
        print("Space added")
        word += ' '; spoken = False


cap.release()
cv2.destroyAllWindows()