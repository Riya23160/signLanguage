🖐️ Real-Time Sign Language Detection System

A real-time computer vision application that detects hand signs using a deep learning model and converts them into text and speech. Built using TensorFlow, OpenCV, and MediaPipe, this system provides an interactive and responsive UI for seamless sign recognition.

🚀 Features
🔍 Real-time hand detection using MediaPipe
🧠 Deep learning model (MobileNetV2) for accurate sign classification
🎯 Stability + confidence filtering to reduce noise
🔊 Text-to-Speech (TTS) for spoken output
🎨 Responsive UI system (adaptive to screen size & fullscreen)
✌️ Supports single and dual-hand gestures
📊 Includes evaluation tools (confusion matrix, classification report)
🧠 System Architecture
Camera Input
   ↓
MediaPipe (Hand Detection)
   ↓
Hand Cropping & Preprocessing
   ↓
Deep Learning Model (MobileNetV2)
   ↓
Prediction + EMA Smoothing
   ↓
Stability Filtering
   ↓
Word Formation
   ↓
UI Rendering + Text-to-Speech
🛠️ Tech Stack
Python
TensorFlow / Keras
OpenCV
MediaPipe
NumPy
pyttsx3 (TTS)
📂 Project Structure
sign_language/
│
├── model_training.py        # Training pipeline
├── predict.py               # Real-time inference system
├── ui_layout.py             # Responsive UI module
├── confusion_matrix.py      # Model evaluation
├── label_map.npy            # Label mapping
├── requirements.txt
│
├── dataset/                 # Raw dataset (not included)
├── data_processed/          # Preprocessed dataset
├── model_trained/           # Saved models (ignored in Git)
⚙️ Installation
1. Clone the repository
git clone https://github.com/your-username/sign-language-detector.git
cd sign-language-detector
2. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # (Linux/Mac)
.venv\Scripts\activate      # (Windows)
3. Install dependencies
pip install -r requirements.txt
▶️ Usage
🔹 Run real-time detection
python predict.py
🎮 Controls
C → Clear word
S → Add space
Q → Quit
🧠 Model Details
Base Model: MobileNetV2 (pretrained on ImageNet)
Input Size: 128 × 128
Training Strategy:
Transfer Learning
Data Augmentation
Fine-tuning last layers
Accuracy: ~98% on validation set
📊 Evaluation

Run:

python confusion_matrix.py

Outputs:

Confusion Matrix
Precision / Recall / F1-score
⚠️ Important Notes
The model expects cropped hand images (same preprocessing as training)
Ensure good lighting conditions for best accuracy
Fullscreen UI works best when run outside VS Code terminal
