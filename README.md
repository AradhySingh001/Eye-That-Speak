Project: Integrated Smart Glasses System

This repository contains a Python implementation of an Integrated Smart Glasses System that combines object detection and GPS navigation with unified voice control. The system leverages computer vision, speech recognition, and text-to-speech functionalities to provide an interactive, hands-free experience for users.

Key Features
Object Detection: Utilizes YOLOv8 for real-time object detection, allowing users to identify and count objects in the camera's view.
GPS Navigation: Provides location-based services, including current location retrieval, navigation to destinations, and saving locations.
Voice Control: Supports unified voice commands for seamless switching between detection and GPS modes, with commands like "find [object]," "navigate to [place]," and "where am I."
Camera Integration: Processes live camera feed with visual overlays for system status and detected objects.
Text-to-Speech: Announces system status, detected objects, and navigation updates using a customizable voice.
Error Handling: Robust error handling for camera, model, and speech recognition failures.
User Interface: Displays real-time video feed with annotated object detection boxes and system status overlays.
Dependencies
ultralytics (for YOLOv8)
opencv-python (for camera and image processing)
pyttsx3 (for text-to-speech)
speechrecognition (for voice command recognition)
pyaudio (for microphone input)
requests (for GPS functionality via IP-API)
Voice Commands
Mode Switching: "switch to detection", "switch to gps", "what mode"
Object Detection: "start detection", "stop detection", "what do you see", "find [object]", "count [objects]"
GPS Navigation: "where am I", "navigate to [place]", "save location as [name]", "distance to [place]", "stop navigation"
System Commands: "help", "system status", "test camera", "exit"
System Requirements
Python 3.6+
Compatible webcam
Microphone for voice input
Internet connection for GPS functionality
Notes
Ensure proper camera and microphone permissions.
The system supports fallback camera indices and ambient noise calibration for robust performance.
GPS functionality uses a basic IP-based location service; a real implementation would require more sophisticated GPS hardware integration.
The YOLOv8 model (yolov8n.pt) must be available in the working directory or downloaded automatically.
Future Improvements
Implement actual turn-by-turn navigation with GPS hardware.
Enhance object detection with custom-trained models for specific use cases.
Add augmented reality (AR) overlays for a richer user experience.
Optimize performance for low-power devices like smart glasses hardware.
