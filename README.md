# 🤟 Real-Time Sign Language Recognition & Translation

**A patented, AI-driven assistive technology system designed to bridge communication gaps for the Deaf and Hard of Hearing (DHH) community.** 

This project utilizes a standard webcam to perform real-time hand tracking and translates sign language gestures into readable text and synthesized spoken audio. By heavily leveraging Computer Vision, Deep Learning, and Operating System-level multithreading, this system eliminates the need for expensive sensor gloves or specialized hardware, offering a highly accessible software solution. 

**Note:** A patent for the methodology and architecture of this system has been officially published by the Indian Patent Office.

## **Key Features**
* **Real-Time Computer Vision:** Extracts a 42-point 3D geometric skeletal mapping of the user's hand using Google's MediaPipe framework, filtering out background noise and lighting variations.
* **Deep Learning Inference:** Utilizes a custom PyTorch Feed-Forward Neural Network (Multi-Layer Perceptron) to classify complex spatial coordinate data with high accuracy.
* **Temporal Smoothing:** Implements a sliding window buffer and statistical mode algorithms to filter out high-frequency noise, ensuring flicker-free, stable text predictions on screen.
* **Multithreaded Voice Synthesis:** Dynamically hijacks the native Windows Text-to-Speech (SAPI) engine via PowerShell scripting to instantly vocalize predictions—including high-priority emergency distress signals—without dropping video frame rates.

## **Tech Stack**

| Domain | Technologies Used |
| :--- | :--- |
| **Deep Learning** | PyTorch, Neural Networks (MLP), Cross-Entropy Loss, Adam Optimizer |
| **Computer Vision** | OpenCV, MediaPipe Hands |
| **Data Engineering** | NumPy, Scikit-learn (Train/Test Splits) |
| **System Integrations** | Python `threading`, `subprocess`, Windows PowerShell, SAPI |


Installation & Setup

System Requirements

Operating System: Windows 10/11 (Required for the native SAPI Text-to-Speech integration). The vision/learning models will work on macOS/Linux, but the voice module requires modification.

Python Version: Python 3.9 - 3.12 is highly recommended. (Note: MediaPipe and PyTorch versions may conflict on Python 3.13+).

Hardware: A standard webcam. (GPU is not required for inference, CPU runs this easily).

Step-by-Step Guide

1. Clone the Repository

git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name


2. Create a Virtual Environment (Highly Recommended)
To avoid dependency conflicts, create a clean environment.

# Windows
python -m venv venv
venv\Scripts\activate


3. Install Dependencies
Install the required machine learning and computer vision libraries.

# Install PyTorch (CPU version is sufficient for inference)
pip install torch torchvision torchaudio

# Install Computer Vision & Data libraries
pip install opencv-python mediapipe numpy scikit-learn


4. Run the Application
Once installed, you can run the inference script.

python c.py
