This repository contains the source code for an integrated smart glasses system that leverages AI and computer vision technologies. The system supports multiple modes, including object detection (using YOLOv8), GPS navigation, text reading (via OCR), and facial recognition, all controlled through voice commands. It is designed to run on a local machine with a webcam and microphone, providing a hands-free experience.


Features
Object Detection: Detect and describe objects in the camera view using the YOLOv8 model.
GPS Navigation: Retrieve current location, save locations, and simulate navigation (future enhancement).
Text Reading: Read text from the camera feed using Tesseract OCR with customizable language and speed.
Facial Recognition: Identify known faces, learn new faces, and manage a face database.
Voice Control: Unified voice command system for mode switching and feature activation.
Real-time Overlay: Display system status and detected objects on the video feed.
Prerequisites
System Dependencies
Tesseract OCR: Required for text reading.
Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki and install (e.g., using tesseract-installer.exe).
Ubuntu: sudo apt install tesseract-ocr
macOS: brew install tesseract
dlib: Required for facial recognition (may need CMake and Visual Studio Build Tools on Windows).
Install via pip install dlib or use a pre-built wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/#dlib.
Hardware
Webcam (for video input).
Microphone (for voice commands).
Installation
Clone the Repository

bash


git clone https://github.com/yourusername/smart-glasses.git
cd smart-glasses
Set Up a Virtual Environment (optional but recommended)
bash




python -m venv smart_glasses_env
smart_glasses_env\Scripts\activate  # On Windows
# Or on macOS/Linux: source smart_glasses_env/bin/activate
Install Python Dependencies Create or update requirements.txt with the following content and run:
bash




pip install -r requirements.txt
requirements.txt:
text

flask==3.0.3
opencv-python==4.10.0.84
ultralytics==8.2.48
pyttsx3==2.94
speechrecognition==3.10.4
pyaudio==0.2.14
requests==2.32.3
pytesseract==0.3.10
pillow==10.4.0
face_recognition==1.3.0
numpy==1.26.4
werkzeug==3.0.3
jinja2==3.1.4
Download Model Files
Place the yolov8n.pt file in the project root directory (pre-trained YOLOv8 model).
Ensure face_database.pkl and face_names.json are in the root for facial recognition (will be created if missing).
Configure Tesseract
Set the TESSDATA_PREFIX environment variable to the Tesseract data directory if needed (e.g., C:\Program Files\Tesseract-OCR\tessdata on Windows).
Usage



Run the Application
bash




python smart_glasses_main.py
The system will initialize the camera, start voice control, and display a video feed with overlays.
Press q to quit, m to toggle modes, f for instant face recognition, or r for instant text reading.
Voice Commands Use the following commands (say "help" for a full list):
Mode Switching: "switch to detection", "switch to gps", "switch to text reading", "switch to face recognition".
Object Detection: "start detection", "stop detection", "what do you see", "find [object]", "count [objects]".
GPS Navigation: "where am i", "navigate to [place]", "save location as [name]", "distance to [place]", "stop navigation".
Text Reading: "read text", "start reading", "stop reading", "read again", "read slowly", "read fast", "change language to [lang]".
Facial Recognition: "who is this", "start face recognition", "stop face recognition", "learn face as [name]", "forget [name]", "list known faces".
System: "help", "system status", "test camera", "exit".
Folder Structure
text



smart_glasses/
├── __pycache__/
├── .dist/
├── smart_glasses_env/
├── templates/
│   ├── index.html
│   └── vosk-model-small-en-us-0.15/
├── app.py
├── cmake-installer.exe
├── face_database.pkl
├── face_names.json
├── nul
├── requirements.txt
├── smart_glasses_location.py
├── smart_glasses_main.py
├── tesseract-installer.exe
├── tesseract-ocr.exe
├── vosk-model-small-en-us-0.15/
└── yolov8n.pt
Note: Consolidate vosk-model-small-en-us-0.15 into a models/ subdirectory for cleaner organization.
Contributing
Fork the repository.
Create a new branch (git checkout -b feature-branch).
Make your changes and commit them (git commit -m "Description of changes").
Push to the branch (git push origin feature-branch).
Open a Pull Request with a clear description of your changes.

Acknowledgments
Ultralytics YOLOv8 for object detection.
Tesseract OCR for text reading.
face_recognition for facial recognition.
xAI for inspiration and tools.
Notes
Customization: Replace yourusername with your GitHub username and [Your Name or GitHub Username] with your name or handle. Add a LICENSE file if not already present.

Future Enhancements: The GPS distance calculation is noted as a future feature.
