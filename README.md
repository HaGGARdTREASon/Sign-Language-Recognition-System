Real-Time Sign Language Recognition & Translation

A patented, AI-driven assistive technology system designed to bridge communication gaps for the Deaf and Hard of Hearing (DHH) community.

This project utilizes a standard webcam to perform real-time hand tracking and translates sign language gestures into readable text and synthesized spoken audio. By heavily leveraging Computer Vision, Deep Learning, and Operating System-level multithreading, this system eliminates the need for expensive sensor gloves or specialized hardware, offering a highly accessible software solution. 

Note: A patent for the methodology and architecture of this system has been officially published by the Indian Patent Office.

Key Features
* Real-Time Computer Vision: Extracts a 42-point 3D geometric skeletal mapping of the user's hand using Google's MediaPipe framework, filtering out background noise and lighting variations.
* Deep Learning Inference: Utilizes a custom PyTorch Feed-Forward Neural Network (Multi-Layer Perceptron) to classify complex spatial coordinate data with high accuracy.
* Temporal Smoothing: Implements a sliding window buffer and statistical mode algorithms to filter out high-frequency noise, ensuring flicker-free, stable text predictions on screen.
* Multithreaded Voice Synthesis: Dynamically hijacks the native Windows Text-to-Speech (SAPI) engine via PowerShell scripting to instantly vocalize predictions—including high-priority emergency distress signals—without dropping video frame rates.

Tech Stack

| Domain | Technologies Used |
| :--- | :--- |
| Deep Learning | PyTorch, Neural Networks (MLP), Cross-Entropy Loss, Adam Optimizer |
| Computer Vision | OpenCV, MediaPipe Hands |
| Data Engineering | NumPy, Scikit-learn (Train/Test Splits) |
| System Integrations | Python `threading`, `subprocess`, Windows PowerShell, SAPI |
