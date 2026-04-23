# ==========================
# PHASE 1 — TRAINING SCRIPT
# ==========================
import os
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# --------------------------------------
# PATHS & CONFIG
# --------------------------------------
INPUT_DIR = "dataset/images"
PROCESSED_DIR = "data_processed"
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 40

# --------------------------------------
# MEDIAPIPE HAND CROP (Supports 2 Hands)
# --------------------------------------
mp_hands = mp.solutions.hands

def extract_hands(img_path, img_size=128):
    """Extract one OR two hands from an image using MediaPipe."""
    img = cv2.imread(img_path)
    if img is None:
        return None

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
        res = hands.process(rgb)

        if not res.multi_hand_landmarks:
            return None

        crops = []
        h, w, _ = img.shape

        for hand_landmarks in res.multi_hand_landmarks:
            x_min = min([lm.x for lm in hand_landmarks.landmark]) * w
            y_min = min([lm.y for lm in hand_landmarks.landmark]) * h
            x_max = max([lm.x for lm in hand_landmarks.landmark]) * w
            y_max = max([lm.y for lm in hand_landmarks.landmark]) * h

            margin = 30
            x_min, y_min = max(0, int(x_min - margin)), max(0, int(y_min - margin))
            x_max, y_max = min(w, int(x_max + margin)), min(h, int(y_max + margin))

            crop = img[y_min:y_max, x_min:x_max]
            if crop.size == 0:
                continue

            resized = cv2.resize(crop, (img_size, img_size))
            crops.append(resized)

        if len(crops) == 0:
            return None

        # If two hands → concatenate side-by-side
        if len(crops) == 2:
            combined = np.hstack(crops)
            combined = cv2.resize(combined, (img_size, img_size))
            return combined

        # Single hand
        return crops[0]

# --------------------------------------
# CREATE PROCESSED DATASET
# --------------------------------------
os.makedirs(PROCESSED_DIR, exist_ok=True)

for label in tqdm(os.listdir(INPUT_DIR), desc="Preprocessing Dataset"):
    src_dir = os.path.join(INPUT_DIR, label)
    if not os.path.isdir(src_dir):
        continue

    dst_dir = os.path.join(PROCESSED_DIR, label)
    os.makedirs(dst_dir, exist_ok=True)

    for img_file in os.listdir(src_dir):
        img_path = os.path.join(src_dir, img_file)
        processed = extract_hands(img_path, IMG_SIZE)
        if processed is not None:
            cv2.imwrite(os.path.join(dst_dir, img_file), processed)

# --------------------------------------
# TRAINING
# --------------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    PROCESSED_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    PROCESSED_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    subset='validation'
)

# --------------------------------------
# MODEL
# --------------------------------------
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
out = Dense(train_data.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=out)

steps_per_epoch = len(train_data)

# ❗ FIXED LINE (no syntax error)
lr_sched = CosineDecay(
    initial_learning_rate=0.0003,
    decay_steps=EPOCHS * steps_per_epoch
)

model.compile(
    optimizer=Adam(learning_rate=lr_sched),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
    ModelCheckpoint("best_sign_model.keras", save_best_only=True, verbose=1)
]

model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks
)

model.save("final_sign_model.keras")

print("\n✅ Training complete. Model saved as final_sign_model.keras.")
