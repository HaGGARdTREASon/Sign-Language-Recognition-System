import os
import cv2
import string
import pyttsx3
import time

# --- CONFIGURATION ---
DATA_DIR = "./dataset"

# 1. Classes: Alphabets + New Phrases
ALPHABET = list(string.ascii_uppercase)
PHRASES = ["FULL", "I HAVE SOMETHING", "THIRSTY", "HURT", 
           "EXHAUSTED", "HUNGRY", "PAIN", "ENJOY"]
CLASSES = ALPHABET + PHRASES 

IMAGES_PER_CLASS = 100

# --- VOICE SETUP ---
engine = pyttsx3.init()
engine.setProperty('rate', 150) # Speaking speed

def speak(text):
    """Speak text (blocking is okay for collection setup)"""
    try:
        engine.say(text)
        engine.runAndWait()
    except:
        pass

# Create main directory
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

cap = cv2.VideoCapture(0)

print("--- Data Collection ---")
speak("Starting data collection. Press Q to capture.")

for letter in CLASSES:
    # Create folder (e.g., dataset/HUNGRY)
    class_dir = os.path.join(DATA_DIR, letter)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
    
    # Check how many images are already there
    existing = len(os.listdir(class_dir))
    counter = existing
    
    print(f"\n>>> CURRENT CLASS: '{letter}' <<<")
    speak(f"Next class is {letter}")
    
    while True:
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.flip(frame, 1)
        
        # Display Info
        cv2.putText(frame, f"Class: {letter}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, f"Saved: {counter}", (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Hold 'Q' to Record | 'N' for Next", (50, 450), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        cv2.imshow('Collector', frame)
        
        key = cv2.waitKey(25)
        
        # 'Q' to Record (Hold for continuous capture)
        if key == ord('q'):
            save_path = os.path.join(class_dir, f"{letter}_{counter}.jpg")
            cv2.imwrite(save_path, frame)
            counter += 1
            print(f"Captured: {letter}_{counter}.jpg")
            
        # 'N' to go to Next Class
        if key == ord('n'):
            break
            
        # 'ESC' to Quit completely
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            exit()

speak("All classes collected.")
print("\n--- Collection Complete ---")
cap.release()
cv2.destroyAllWindows()
