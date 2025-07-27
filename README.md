#Smart Glasses 


A comprehensive AI-powered smart glasses system that combines object detection, GPS navigation, text reading, and facial recognition with unified voice control.


Key Highlights:

🔍 Real-time Object Detection using YOLOv8

👤 Advanced Facial Recognition with persistent database

📖 Multi-language Text Reading via Tesseract OCR

🗺️ GPS Navigation with location management

🎤 Voice Control with natural language processing

🌐 Modern Web Interface with live camera streaming

🔊 Text-to-Speech feedback system


🎯 Features
🔍 Object Detection
Real-time Detection: Continuous object detection using YOLOv8

80+ Object Classes: Recognizes people, vehicles, animals, and everyday objects

Position Awareness: Announces object positions (left, right, center)

Confidence Filtering: Adjustable confidence thresholds

Detection Modes: Continuous, on-demand, and selective detection

👤 Facial Recognition
High Accuracy Recognition: Uses advanced face encoding algorithms

Persistent Database: Stores face encodings in pickle format with JSON names

Dynamic Learning: Add new faces with voice commands

Confidence Scoring: Recognition with confidence percentages

Face Management: Add, remove, and list known faces

📖 Text Reading (OCR)
Multi-language Support: English, Spanish, French, German

Real-time OCR: Extract text from live camera feed

Text Preprocessing: Image enhancement for better accuracy

Reading Speed Control: Adjustable speech rate

Continuous Reading: Automatic text monitoring mode

🗺️ GPS Navigation
Location Detection: Get current location via IP geolocation

Navigation Assistance: Start navigation to destinations

Location Saving: Save and manage custom locations

Distance Calculations: Calculate distances to destinations

Coordinate Display: Show precise GPS coordinates

🎤 Voice Control System
Natural Language Processing: Understand conversational commands

Real-time Recognition: Instant voice command processing

TTS Feedback: Clear audio responses

Background Listening: Continuous voice monitoring

Command Queue: Handles multiple commands efficiently

🌐 Web Interface
Modern UI: Responsive glassmorphism design

Live Camera Feed: Real-time video streaming

Interactive Controls: Click-to-execute commands

System Monitoring: Live status updates and metrics

Mobile Responsive: Works on all device sizes

🛠️ Installation
Prerequisites
Python 3.8 or higher

Webcam/Camera for video input

Microphone for voice commands

Speakers/Headphones for audio output

Internet connection for GPS features

____________________________________________________________________________________________________________________________________________________________________________________________________________________




Voice Commands Reference
Mode Switching
"switch to detection" - Object detection mode

"switch to gps" - GPS navigation mode

"switch to text reading" - Text reading mode

"switch to face recognition" - Facial recognition mode

"what mode" - Check current mode

Object Detection Commands
"what do you see" - Describe current view

"start detection" - Begin continuous detection

"stop detection" - Pause detection

"find [object]" - Look for specific object (e.g., "find person")

"count [objects]" - Count specific objects (e.g., "count cars")

Facial Recognition Commands
"who is this" - Identify person in view

"learn face as [name]" - Add new person (e.g., "learn face as John")

"forget [name]" - Remove person from database

"list known faces" - Show all known people

"start face recognition" - Begin continuous recognition

"stop face recognition" - Stop recognition

"face database status" - Show database info

Text Reading Commands
"read text" - Read text from camera view

"start reading" - Start continuous text reading

"stop reading" - Stop text reading

"read again" - Repeat last read text

"read slowly" - Set slow reading speed

"read fast" - Set fast reading speed

"change language to [lang]" - Change OCR language

GPS Navigation Commands
"where am i" - Get current location

"navigate to [place]" - Start navigation

"save location as [name]" - Save current location

"distance to [place]" - Get distance information

"stop navigation" - End navigation

"my coordinates" - Show GPS coordinates

System Commands
"help" - Show all available commands

"system status" - Display system information

"test camera" - Test camera functionality

"exit" or "quit" - Close application
____________________________________________________________________________________________________________________________________________________________________________________________________________________
Web Interface Features
Camera Feed
Live Streaming: Real-time camera feed with AI processing overlays

Mode Indicators: Visual status of current operating mode

Detection Boxes: Real-time object detection visualization

Face Recognition: Live face identification with confidence scores

Control Panel
Mode Cards: Click to switch between different AI modes

Quick Commands: One-click execution of common commands

Status Monitoring: Real-time system metrics and status

Voice Control: Start/stop voice recognition

System Output
Live Log: Real-time command execution and system messages

Color-coded Entries: Success, error, and info message types

Scrollable History: Keep track of all system activities

Clear Function: Reset log when needed

Keyboard Shortcuts
Desktop Application
q - Quit application

m - Toggle between modes

r - Instant text reading

f - Instant face recognition




____________________________________________________________________________________________________________________________________________________________________________________________________________________


#SMART GLASSES WITH FACIAL RECOGNITION - SETUP GUIDE
==================================================

STEP 1: SYSTEM REQUIREMENTS CHECK
---------------------------------
Before starting, ensure you have:
- Windows 10/11, Ubuntu 18.04+, or macOS 10.14+
- At least 4GB RAM (8GB recommended)
- 2GB free disk space
- Working webcam (built-in or USB)
- Working microphone
- Internet connection for initial setup

STEP 2: INSTALL SYSTEM DEPENDENCIES
-----------------------------------

FOR WINDOWS USERS:
------------------
1. Download and install Python 3.8+ from python.org
   - Make sure to check "Add Python to PATH" during installation

2. Install Tesseract OCR:
   - Go to: https://github.com/UB-Mannheim/tesseract/wiki
   - Download the Windows installer
   - Install to default location (usually C:\Program Files\Tesseract-OCR)
   - Add Tesseract to your PATH:
     * Open System Properties > Environment Variables
     * Add C:\Program Files\Tesseract-OCR to your PATH variable

3. Install Visual Studio Build Tools (for dlib/face_recognition):
   - Go to: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Download and install "Build Tools for Visual Studio"
   - Select "C++ build tools" workload during installation

FOR UBUNTU/DEBIAN USERS:
------------------------
1. Update system packages:
   sudo apt update && sudo apt upgrade -y

2. Install Python and development tools:
   sudo apt install python3 python3-pip python3-dev

3. Install system dependencies:
   sudo apt install tesseract-ocr tesseract-ocr-eng
   sudo apt install cmake build-essential
   sudo apt install portaudio19-dev python3-pyaudio
   sudo apt install libopencv-dev

FOR MACOS USERS:
----------------
1. Install Homebrew if not already installed:
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

2. Install system dependencies:
   brew install python3
   brew install tesseract
   brew install cmake
   brew install portaudio

STEP 3: CREATE PROJECT DIRECTORY
--------------------------------
1. Create a new folder for the project:
   mkdir smart_glasses
   cd smart_glasses

2. Create a virtual environment (recommended):
   python -m venv venv
   
   # Activate virtual environment:
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate

STEP 4: INSTALL PYTHON PACKAGES
-------------------------------
Install packages in this specific order to avoid conflicts:

1. Install basic packages first:
   pip install --upgrade pip
   pip install numpy
   pip install opencv-python

2. Install machine learning packages:
   pip install ultralytics
   pip install pillow

3. Install audio packages:
   pip install pyttsx3
   pip install speechrecognition
   
   # For Windows users who have issues with pyaudio:
   pip install pipwin
   pipwin install pyaudio
   
   # For Linux/macOS:
   pip install pyaudio

4. Install OCR package:
   pip install pytesseract

5. Install face recognition (this may take 10-15 minutes):
   pip install cmake
   pip install dlib
   pip install face_recognition

6. Install remaining packages:
   pip install requests

STEP 5: DOWNLOAD PROJECT FILES
-----------------------------
1. Copy the smart_glasses.py file to your project directory

2. Verify the file structure:
   smart_glasses/
   ├── smart_glasses.py
   └── venv/ (if using virtual environment)

STEP 6: TEST INSTALLATION
-------------------------
1. Test basic imports:
   python -c "import cv2, numpy, pyttsx3, speech_recognition; print('Basic packages OK')"

2. Test face recognition:
   python -c "import face_recognition; print('Face recognition OK')"

3. Test Tesseract:
   python -c "import pytesseract; print('Tesseract OK')"

4. Test YOLO:
   python -c "from ultralytics import YOLO; print('YOLO OK')"

STEP 7: CONFIGURE PERMISSIONS
-----------------------------
1. Grant camera permissions:
   - Windows: Settings > Privacy > Camera > Allow apps to access camera
   - macOS: System Preferences > Security & Privacy > Camera
   - Linux: Usually automatic, check with: ls /dev/video*

2. Grant microphone permissions:
   - Windows: Settings > Privacy > Microphone > Allow apps to access microphone
   - macOS: System Preferences > Security & Privacy > Microphone
   - Linux: Check audio devices: arecord -l

STEP 8: FIRST RUN
----------------
1. Navigate to your project directory:
   cd smart_glasses

2. Activate virtual environment (if using):
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate

3. Run the application:
   python smart_glasses.py

4. Wait for initialization messages:
   - "Loading YOLOv8 model..." (may download model file first time)
   - "TTS initialized successfully!"
   - "Speech recognition ready!"
   - "Facial recognition initialized!"

5. Look for the camera window to open

STEP 9: INITIAL TESTING
-----------------------
1. Test camera:
   - Say "test camera" or press 'f' key
   - You should see a camera feed window

2. Test voice recognition:
   - Say "help" - system should respond with voice
   - Try "what mode" - should announce current mode

3. Test basic detection:
   - Say "what do you see"
   - Point camera at objects and wait for description

4. Test mode switching:
   - Say "switch to text reading"
   - Say "switch to face recognition"
   - Press 'm' key to cycle through modes

STEP 10: FACE RECOGNITION SETUP
-------------------------------
1. Switch to face recognition mode:
   - Say "switch to face recognition"

2. Learn your first face:
   - Position your face in camera view
   - Say "learn face as [your name]"
   - Wait for confirmation

3. Test recognition:
   - Say "who is this" while your face is visible
   - System should identify you

TROUBLESHOOTING COMMON ISSUES
=============================

ISSUE: "Camera not available"
SOLUTION: 
- Check if other apps are using camera
- Try different camera indices
- Restart the application

ISSUE: "Speech recognition not working"
SOLUTION:
- Check microphone permissions
- Ensure microphone is not muted
- Try speaking louder and clearer
- Check microphone in system settings

ISSUE: "ModuleNotFoundError for cv2/face_recognition"
SOLUTION:
- Ensure virtual environment is activated
- Reinstall the specific package:
  pip uninstall opencv-python
  pip install opencv-python

ISSUE: "Tesseract not found"
SOLUTION:
- Verify Tesseract installation path
- Add to PATH environment variable
- Restart command prompt/terminal

ISSUE: "Face recognition installation fails"
SOLUTION:
- Install Visual Studio Build Tools (Windows)
- Install cmake and build tools
- Try: pip install --upgrade pip setuptools wheel
- Install dlib separately first

ISSUE: "No audio output"
SOLUTION:
- Check system volume
- Try different TTS voice
- Restart the application

USAGE TIPS
==========
1. Speak clearly and at normal pace
2. Wait for system responses before next command
3. Ensure good lighting for camera features
4. Keep objects/faces centered in camera view
5. Use "help" command to see all available commands

VOICE COMMANDS QUICK REFERENCE
==============================
Mode Switching:
- "switch to detection"
- "switch to gps" 
- "switch to text reading"
- "switch to face recognition"

Detection:
- "what do you see"
- "find person"
- "count bottles"

Text Reading:
- "read text"
- "read slowly"

Face Recognition:
- "who is this"
- "learn face as John"
- "list known faces"

System:
- "help"
- "system status"
- "exit"

NEXT STEPS
==========
1. Learn faces of family/friends
2. Test different lighting conditions
3. Try text reading with books/signs
4. Experiment with object detection
5. Customize settings in code if needed

____________________________________________________________________________________________________________________________________________________________________________________________________________________


#Main Setup Steps:

System Requirements Check
Install System Dependencies (Windows/Ubuntu/macOS specific)
Create Project Directory
Install Python Packages (in correct order)
Download Project Files
Test Installation
Configure Permissions
First Run
Initial Testing
Face Recognition Setup
