import os
import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# --- CONFIG ---
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "numbers_model.pth")
CLASS_FILE = os.path.join(BASE_DIR, "classes.npy")
EPOCHS = 50
BATCH_SIZE = 32

# --- 1. DETECT CLASSES ---
if not os.path.exists(DATA_DIR):
    print(f"Error: Dataset not found at {DATA_DIR}")
    exit()

# Get folder names (A, B, HUNGRY, THIRSTY...)
CLASSES = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
NUM_CLASSES = len(CLASSES)
print(f"--- Detected {NUM_CLASSES} Classes: {CLASSES} ---")

# Save class list for Realtime script
np.save(CLASS_FILE, CLASSES)
label_map = {label: idx for idx, label in enumerate(CLASSES)}

# --- HELPER: LANDMARK EXTRACTION ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

def get_landmarks(image):
    """Converts image to 42 landmarks (21 x,y pairs)"""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0]
        # Flatten to [x1, y1, x2, y2, ...]
        return np.array([[p.x, p.y] for p in lm.landmark]).flatten()
    return None

# --- 2. PROCESS DATA (IMAGES & VIDEOS) ---
data = []
labels = []

print("--- Processing Data (This may take a moment) ---")

for class_name in CLASSES:
    folder_path = os.path.join(DATA_DIR, class_name)
    file_count = 0
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # A. Process Images
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = cv2.imread(file_path)
            if img is None: continue
            lms = get_landmarks(img)
            if lms is not None:
                data.append(lms)
                labels.append(label_map[class_name])
                file_count += 1
                
        # B. Process Videos (Frame extraction)
        elif filename.lower().endswith(('.mp4', '.avi', '.mov')):
            cap = cv2.VideoCapture(file_path)
            frame_skip = 2 # Process every 2nd frame
            curr = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                if curr % frame_skip == 0:
                    lms = get_landmarks(frame)
                    if lms is not None:
                        data.append(lms)
                        labels.append(label_map[class_name])
                        file_count += 1
                curr += 1
            cap.release()
            
    print(f"Processed '{class_name}': {file_count} samples.")

if not data:
    print("Error: No landmarks extracted. Check your dataset images/videos.")
    exit()

# Convert to Tensors
X = np.array(data, dtype=np.float32)
y = np.array(labels, dtype=np.int64)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 3. MODEL ARCHITECTURE ---
class AlphabetNet(nn.Module):
    def __init__(self, num_classes):
        super(AlphabetNet, self).__init__()
        self.fc1 = nn.Linear(42, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = AlphabetNet(NUM_CLASSES)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- 4. TRAIN LOOP ---
print("\n--- Training Started ---")
for epoch in range(EPOCHS):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 10 == 0:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        print(f"Epoch {epoch+1}/{EPOCHS} | Accuracy: {100 * correct / total:.2f}%")

# Save
torch.save(model.state_dict(), MODEL_PATH)
hands.close()
print(f"Model saved to: {MODEL_PATH}")
