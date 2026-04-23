import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ======================================
# LOAD MODEL
# ======================================
MODEL_PATH = "model_trained/final_sign_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# ======================================
# LOAD DATA (VALIDATION SET)
# ======================================
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 128
BATCH_SIZE = 32

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_data = datagen.flow_from_directory(
    "data_processed",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    subset='validation',
    shuffle=False
)

# ======================================
# PREDICT
# ======================================
preds = model.predict(val_data, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = val_data.classes

# ======================================
# LABELS
# ======================================
labels = list(val_data.class_indices.keys())

# ======================================
# CONFUSION MATRIX
# ======================================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels,
            yticklabels=labels)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ======================================
# CLASSIFICATION REPORT
# ======================================
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=labels))