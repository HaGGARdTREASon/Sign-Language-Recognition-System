import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import torch
import torch.nn as nn
import mediapipe as mp
import numpy as np
import av
import os
from collections import deque, Counter

# --- CONFIG ---
MODEL_FILE = "alphabet_model.pth"
CLASS_FILE = "classes.npy"

# --- MODEL DEFINITION (Must match your training) ---
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

# --- CACHED RESOURCES ---
@st.cache_resource
def load_resources():
    # 1. Load Classes
    try:
        classes = np.load(CLASS_FILE).tolist()
    except Exception as e:
        st.error(f"Error loading classes.npy: {e}")
        return None, None, None

    # 2. Load Model
    device = torch.device('cpu') # Use CPU for cloud hosting (cheaper/standard)
    model = AlphabetNet(len(classes)).to(device)
    
    try:
        model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
        model.eval()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

    # 3. MediaPipe
    mp_hands = mp.solutions.hands
    return model, classes, mp_hands

model, classes, mp_hands = load_resources()

# --- VIDEO PROCESSING CLASS ---
class SignLanguageProcessor:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False, 
            max_num_hands=1, 
            min_detection_confidence=0.7
        )
        self.buffer = deque(maxlen=15)
        self.device = torch.device('cpu')

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 1. Flip & Process
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        
        prediction_text = "..."
        conf_score = 0.0
        
        if results.multi_hand_landmarks and model is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw Landmarks
                mp.solutions.drawing_utils.draw_landmarks(
                    img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract Data
                lm_list = []
                for lm in hand_landmarks.landmark:
                    lm_list.extend([lm.x, lm.y])
                
                # Predict
                input_tensor = torch.tensor([lm_list], dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.softmax(output, 1)
                    confidence, predicted_idx = torch.max(probabilities, 1)
                    
                    idx = predicted_idx.item()
                    conf_score = confidence.item()

                # Buffer Logic
                if conf_score > 0.6:
                    self.buffer.append(idx)
                
                if len(self.buffer) >= 5:
                    most_common_idx = Counter(self.buffer).most_common(1)[0][0]
                    prediction_text = classes[most_common_idx]

                # Draw Text on Video
                cv2.rectangle(img, (0,0), (300, 60), (0,0,0), -1)
                cv2.putText(img, f"{prediction_text} ({conf_score:.2f})", (20, 45), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- STREAMLIT UI ---
st.title("Sign Language Recognizer 🖐️")
st.write("Allow camera access to start.")

if model is not None:
    webrtc_streamer(
        key="sign-language",
        video_processor_factory=SignLanguageProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
else:
    st.error("Model or Classes file missing. Please upload 'alphabet_model.pth' and 'classes.npy' to the repository.")
