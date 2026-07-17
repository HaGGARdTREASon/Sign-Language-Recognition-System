import cv2
import torch
import torch.nn as nn
import mediapipe as mp
import numpy as np
import os
import threading
import subprocess
from collections import deque, Counter

# --- CONFIG ---
BASE_DIR = os.getcwd()
MODEL_PATH = os.path.join(BASE_DIR, "alphabet_model.pth")
CLASS_FILE = os.path.join(BASE_DIR, "classes.npy")

# --- LOAD CLASSES ---
try:
    CLASSES = np.load(CLASS_FILE).tolist()
    print(f"--- Loaded {len(CLASSES)} Classes ---")
except:
    print("Error: Could not load classes.npy. Run trainmodel.py first.")
    exit()

NUM_CLASSES = len(CLASSES)

# --- ROBUST VOICE (WINDOWS NATIVE) ---
def speak_windows(text):
    """
    Uses Windows built-in TTS via PowerShell.
    Runs in a separate process so it NEVER blocks or freezes the video.
    """
    def _run():
        # PowerShell command to speak text
        cmd = f'powershell -Command "Add-Type –AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{text}\')"'
        
        # Run silently (without popping up a black window)
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        
        subprocess.Popen(cmd, startupinfo=startupinfo, shell=False)

    # Fire and forget
    t = threading.Thread(target=_run)
    t.start()

# --- MODEL DEFINITION ---
class AlphabetNet(nn.Module):
    def __init__(self, num_classes):
        super(AlphabetNet, self).__init__()
        self.fc1 = nn.Linear(42, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# --- INITIALIZE SYSTEM ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AlphabetNet(NUM_CLASSES).to(device)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print("✔ Model Loaded.")
else:
    print("⚠ Model not found. Please train first.")
    exit()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# Prediction Buffer
buffer = deque(maxlen=15)
last_spoken_word = ""

print("--- Sign Language Voice Assistant Running (Press Q to Quit) ---")

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        prediction_text = "..."
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                lm_list = []
                for lm in hand_landmarks.landmark:
                    lm_list.extend([lm.x, lm.y])
                
                input_tensor = torch.tensor([lm_list], dtype=torch.float32).to(device)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.softmax(output, 1)
                    confidence, predicted_idx = torch.max(probabilities, 1)
                    
                    idx = predicted_idx.item()
                    conf = confidence.item()
                
                # Buffer Logic
                if conf > 0.6:
                    buffer.append(idx)
                
                if len(buffer) >= 5:
                    most_common_idx = Counter(buffer).most_common(1)[0][0]
                    word = CLASSES[most_common_idx]
                    
                    prediction_text = word
                    
                    # --- VOICE TRIGGER ---
                    if word != last_spoken_word:
                        print(f"Speaking: {word}")
                        speak_windows(word) # Call the new Windows function
                        last_spoken_word = word
                    
                    # UI
                    cv2.rectangle(frame, (0,0), (400, 60), (0,0,0), -1)
                    cv2.putText(frame, f"{word} ({conf:.2f})", (20, 45), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Sign Language Voice Assistant', frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
