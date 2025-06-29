"""
Smart Glasses Main Integration - Complete Version with Facial Recognition
Combines object detection, GPS functionality, text reading, and facial recognition with unified voice control
"""

import cv2
import numpy as np
from ultralytics import YOLO
import pyttsx3
import speech_recognition as sr
import threading
import time
from collections import defaultdict
import requests
import pytesseract
from PIL import Image
import os
import face_recognition
import pickle
import json
from datetime import datetime

class IntegratedSmartGlasses:
    def __init__(self):
        print("🔧 Initializing Integrated Smart Glasses System...")
        
        # Load YOLOv8 model for object detection
        print("📱 Loading YOLOv8 model...")
        try:
            self.model = YOLO("yolov8n.pt")
            print("✅ YOLOv8 model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            self.model = None
        
        # Initialize text-to-speech
        print("🔊 Initializing TTS...")
        try:
            self.tts = pyttsx3.init()
            self.tts.setProperty('rate', 150)
            self.tts.setProperty('volume', 0.9)
            
            # Try to set a female voice if available
            voices = self.tts.getProperty('voices')
            if voices:
                for voice in voices:
                    if any(keyword in voice.name.lower() for keyword in ['female', 'zira', 'hazel']):
                        self.tts.setProperty('voice', voice.id)
                        break
            print("✅ TTS initialized successfully!")
        except Exception as e:
            print(f"❌ TTS initialization error: {e}")
            self.tts = None
        
        # Initialize speech recognition
        print("🎤 Setting up speech recognition...")
        try:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            
            # Calibrate microphone
            print("   Calibrating microphone...")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
                self.recognizer.energy_threshold = 4000
            print("✅ Speech recognition ready!")
        except Exception as e:
            print(f"❌ Microphone setup error: {e}")
            self.microphone = None
        
        # System state
        self.current_mode = "detection"  # "detection", "gps", "text_reading", "face_recognition"
        self.listening = False
        self.speech_thread = None
        
        # Detection settings
        self.detection_mode = "paused"  # "continuous", "paused", "on_command", "selective"
        self.confidence_threshold = 0.25
        self.target_objects = set()
        self.frame_width = 640
        self.frame_height = 480
        
        # GPS functionality
        self.current_location = None
        self.is_navigating = False
        self.saved_locations = {}
        
        # Text reading functionality
        self.text_reading_active = False
        self.last_read_text = ""
        self.reading_language = 'eng'  # Default OCR language
        
        # Facial recognition functionality
        print("👤 Initializing facial recognition...")
        self.face_recognition_active = False
        self.known_faces = {}  # Dictionary to store known face encodings
        self.face_database_file = "face_database.pkl"
        self.face_names_file = "face_names.json"
        self.recognition_threshold = 0.6
        self.last_recognition_time = 0
        self.recognition_interval = 2.0
        self.load_face_database()
        print("✅ Facial recognition initialized!")
        
        # Camera setup
        self.cap = None
        self.camera_initialized = False
        
        print("✅ Integrated Smart Glasses initialized!")
        self.show_all_commands()
    
    def load_face_database(self):
        """Load known faces from database files"""
        try:
            if os.path.exists(self.face_database_file) and os.path.exists(self.face_names_file):
                with open(self.face_database_file, 'rb') as f:
                    face_encodings = pickle.load(f)
                with open(self.face_names_file, 'r') as f:
                    face_names = json.load(f)
                
                # Combine encodings and names
                for name, encoding in zip(face_names, face_encodings):
                    self.known_faces[name] = encoding
                
                print(f"📚 Loaded {len(self.known_faces)} known faces from database")
            else:
                print("📚 No existing face database found. Starting fresh.")
        except Exception as e:
            print(f"❌ Error loading face database: {e}")
            self.known_faces = {}
    
    def save_face_database(self):
        """Save known faces to database files"""
        try:
            face_names = list(self.known_faces.keys())
            face_encodings = list(self.known_faces.values())
            
            with open(self.face_database_file, 'wb') as f:
                pickle.dump(face_encodings, f)
            with open(self.face_names_file, 'w') as f:
                json.dump(face_names, f)
            
            print(f"💾 Saved {len(self.known_faces)} faces to database")
        except Exception as e:
            print(f"❌ Error saving face database: {e}")
    
    def initialize_camera(self):
        """Initialize camera with fallback options"""
        if self.camera_initialized and self.cap and self.cap.isOpened():
            return True
            
        print("📷 Initializing camera...")
        
        # Try different camera indices
        for idx in [0, 1, -1, 2]:
            try:
                print(f"   Trying camera index {idx}...")
                self.cap = cv2.VideoCapture(idx)
                
                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        print(f"✅ Camera initialized with index {idx}")
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
                        self.camera_initialized = True
                        return True
                    else:
                        self.cap.release()
                        
            except Exception as e:
                print(f"   Failed with index {idx}: {e}")
                if self.cap:
                    self.cap.release()
                    
        print("❌ Could not initialize camera")
        self.camera_initialized = False
        return False
    
    def show_all_commands(self):
        """Display all available voice commands"""
        print("\n🎤 INTEGRATED SMART GLASSES COMMANDS:")
        print("=" * 60)
        
        print("🔄 MODE SWITCHING:")
        print("• 'switch to detection' - Object detection mode")
        print("• 'switch to gps' - GPS navigation mode")
        print("• 'switch to text reading' - Text reading mode")
        print("• 'switch to face recognition' - Facial recognition mode")
        print("• 'what mode' - Current mode status")
        
        print("\n🔍 OBJECT DETECTION COMMANDS:")
        print("• 'start detection' - Begin continuous detection")
        print("• 'stop detection' - Pause detection")
        print("• 'what do you see' - Describe current view")
        print("• 'find [object]' - Look for specific object")
        print("• 'count [objects]' - Count specific objects")
        
        print("\n🗺️ GPS NAVIGATION COMMANDS:")
        print("• 'where am i' - Get current location")
        print("• 'navigate to [place]' - Start navigation")
        print("• 'save location as [name]' - Save current location")
        print("• 'distance to [place]' - Get distance")
        print("• 'stop navigation' - End navigation")
        
        print("\n📖 TEXT READING COMMANDS:")
        print("• 'read text' - Read text from camera view")
        print("• 'start reading' - Start continuous text reading")
        print("• 'stop reading' - Stop text reading")
        print("• 'read again' - Repeat last read text")
        print("• 'read slowly' - Read text at slow speed")
        print("• 'read fast' - Read text at fast speed")
        print("• 'change language to [lang]' - Change OCR language")
        
        print("\n👤 FACIAL RECOGNITION COMMANDS:")
        print("• 'who is this' - Identify person in view")
        print("• 'start face recognition' - Begin continuous face recognition")
        print("• 'stop face recognition' - Stop face recognition")
        print("• 'learn face as [name]' - Add new person to database")
        print("• 'forget [name]' - Remove person from database")
        print("• 'list known faces' - Show all known people")
        print("• 'face database status' - Show database information")
        
        print("\n⚙️ SYSTEM COMMANDS:")
        print("• 'help' - Show all commands")
        print("• 'system status' - Show system status")
        print("• 'test camera' - Test camera")
        print("• 'exit' or 'quit' - Close application")
        print("=" * 60)
    
    def start_voice_control(self):
        """Start unified voice control system"""
        if self.listening or not self.microphone:
            return
        
        self.listening = True
        self.speech_thread = threading.Thread(target=self._unified_speech_loop, daemon=True)
        self.speech_thread.start()
        print("🎤 Unified voice control started!")
    
    def stop_voice_control(self):
        """Stop voice control system"""
        self.listening = False
        if self.speech_thread:
            self.speech_thread.join(timeout=1)
        print("🔇 Voice control stopped.")
    
    def _unified_speech_loop(self):
        """Unified speech recognition loop"""
        while self.listening:
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=1.0, phrase_time_limit=3.0)
                
                command = self.recognizer.recognize_google(audio).lower()
                print(f"🎤 Command: '{command}'")
                
                result = self.process_unified_command(command)
                if result == "exit":
                    break
                
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                continue
            except sr.RequestError as e:
                print(f"Speech recognition error: {e}")
                time.sleep(1)
            except Exception as e:
                print(f"Speech loop error: {e}")
                time.sleep(1)
    
    def process_unified_command(self, command):
        """Process unified voice commands with improved handling"""
        try:
            # Mode switching commands
            if "switch to detection" in command:
                self.current_mode = "detection"
                self.text_reading_active = False
                self.face_recognition_active = False
                self.speak("Switched to object detection mode")
                return "mode_switch"
            
            elif "switch to gps" in command:
                self.current_mode = "gps"
                self.text_reading_active = False
                self.face_recognition_active = False
                self.speak("Switched to GPS navigation mode")
                return "mode_switch"
            
            elif "switch to text reading" in command or "switch to reading" in command:
                self.current_mode = "text_reading"
                self.face_recognition_active = False
                self.speak("Switched to text reading mode")
                return "mode_switch"
            
            elif "switch to face recognition" in command or "switch to face" in command:
                self.current_mode = "face_recognition"
                self.text_reading_active = False
                self.speak("Switched to facial recognition mode")
                return "mode_switch"
            
            elif "what mode" in command:
                self.speak(f"Currently in {self.current_mode.replace('_', ' ')} mode")
                return "status"
            
            # System commands
            elif "help" in command:
                self.speak("Showing all available commands in console")
                self.show_all_commands()
                return "help"
            
            elif "system status" in command:
                self.get_system_status()
                return "status"
            
            elif "test camera" in command:
                self.test_camera()
                return "test"
            
            elif any(word in command for word in ["exit", "quit", "stop system"]):
                self.speak("Shutting down system. Goodbye!")
                return "exit"
            
            # Route commands based on current mode
            elif self.current_mode == "detection":
                return self.process_detection_command(command)
            elif self.current_mode == "gps":
                return self.process_gps_command(command)
            elif self.current_mode == "text_reading":
                return self.process_text_reading_command(command)
            elif self.current_mode == "face_recognition":
                return self.process_face_recognition_command(command)
            else:
                self.speak("Command not recognized. Say help for available commands.")
                return None
            
        except Exception as e:
            print(f"❌ Command processing error: {e}")
            self.speak("Error processing command")
            return None
    
    def process_face_recognition_command(self, command):
        """Process facial recognition commands"""
        try:
            if "who is this" in command or "identify person" in command:
                self.speak("Identifying person in view")
                self.identify_person()
                return "identify"
                
            elif "start face recognition" in command or "continuous recognition" in command:
                self.face_recognition_active = True
                self.speak("Starting continuous facial recognition")
                return "start_recognition"
                
            elif "stop face recognition" in command:
                self.face_recognition_active = False
                self.speak("Facial recognition stopped")
                return "stop_recognition"
                
            elif "learn face as" in command:
                name = command.split("learn face as")[-1].strip()
                if name:
                    self.speak(f"Learning new face as {name}")
                    self.learn_new_face(name)
                else:
                    self.speak("Please specify a name to learn the face")
                return "learn_face"
                
            elif "forget" in command and any(word in command for word in ["person", "face"]):
                name = command.replace("forget", "").replace("person", "").replace("face", "").strip()
                if name:
                    self.forget_person(name)
                else:
                    self.speak("Please specify the name of the person to forget")
                return "forget_face"
                
            elif "list known faces" in command or "show known people" in command:
                self.list_known_faces()
                return "list_faces"
                
            elif "face database status" in command or "database info" in command:
                self.face_database_status()
                return "database_status"
                
            else:
                self.speak("Face recognition command not recognized")
                return None
                
        except Exception as e:
            print(f"❌ Face recognition command error: {e}")
            self.speak("Error in face recognition command")
            return None
    
    def identify_person(self):
        """Identify person in current camera view"""
        frame = self.capture_frame()
        
        if frame is None:
            self.speak("Camera not available. Cannot identify person.")
            return
        
        try:
            # Find faces in the frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            
            if not face_locations:
                self.speak("No faces detected in the current view")
                return
            
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            identified_people = []
            
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(
                    list(self.known_faces.values()), 
                    face_encoding, 
                    tolerance=self.recognition_threshold
                )
                
                name = "Unknown"
                face_distances = face_recognition.face_distance(list(self.known_faces.values()), face_encoding)
                
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = list(self.known_faces.keys())[best_match_index]
                        confidence = (1 - face_distances[best_match_index]) * 100
                        identified_people.append(f"{name} with {confidence:.1f}% confidence")
                    else:
                        identified_people.append("Unknown person")
                else:
                    identified_people.append("Unknown person")
            
            if len(identified_people) == 1:
                self.speak(f"I can see {identified_people[0]}")
            else:
                people_text = ", ".join(identified_people)
                self.speak(f"I can see {len(identified_people)} people: {people_text}")
            
            print(f"👤 Identified: {identified_people}")
            
        except Exception as e:
            print(f"❌ Person identification error: {e}")
            self.speak("Error identifying person")
    
    def learn_new_face(self, name):
        """Learn a new face and add to database"""
        frame = self.capture_frame()
        
        if frame is None:
            self.speak("Camera not available. Cannot learn face.")
            return
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            
            if not face_locations:
                self.speak("No faces detected. Please ensure a face is clearly visible.")
                return
            
            if len(face_locations) > 1:
                self.speak("Multiple faces detected. Please ensure only one person is in view.")
                return
            
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            if face_encodings:
                # Check if this person is already known
                for known_name, known_encoding in self.known_faces.items():
                    distance = face_recognition.face_distance([known_encoding], face_encodings[0])[0]
                    if distance < self.recognition_threshold:
                        self.speak(f"This person looks like {known_name}. Face not learned.")
                        return
                
                # Add new face to database
                self.known_faces[name] = face_encodings[0]
                self.save_face_database()
                
                self.speak(f"Successfully learned {name}'s face")
                print(f"✅ Learned new face: {name}")
            else:
                self.speak("Could not encode the face. Please try again.")
                
        except Exception as e:
            print(f"❌ Face learning error: {e}")
            self.speak("Error learning face")
    
    def forget_person(self, name):
        """Remove a person from the face database"""
        try:
            if name in self.known_faces:
                del self.known_faces[name]
                self.save_face_database()
                self.speak(f"Forgotten {name}")
                print(f"🗑️ Removed {name} from database")
            else:
                self.speak(f"I don't know anyone named {name}")
                
        except Exception as e:
            print(f"❌ Error forgetting person: {e}")
            self.speak("Error removing person from database")
    
    def list_known_faces(self):
        """List all known people in the database"""
        if not self.known_faces:
            self.speak("No known faces in database")
            return
        
        names = list(self.known_faces.keys())
        count = len(names)
        
        if count == 1:
            self.speak(f"I know 1 person: {names[0]}")
        else:
            names_text = ", ".join(names)
            self.speak(f"I know {count} people: {names_text}")
        
        print(f"👥 Known faces: {names}")
    
    def face_database_status(self):
        """Show face database status information"""
        count = len(self.known_faces)
        status_text = f"Face database contains {count} known people"
        
        if count > 0:
            status_text += f". Recognition threshold is {self.recognition_threshold:.1f}"
        
        self.speak(status_text)
        print(f"📊 {status_text}")
    
    def process_continuous_face_recognition(self, frame):
        """Process continuous face recognition"""
        current_time = time.time()
        
        if current_time - self.last_recognition_time < self.recognition_interval:
            return frame
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            
            if face_locations:
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    matches = face_recognition.compare_faces(
                        list(self.known_faces.values()), 
                        face_encoding, 
                        tolerance=self.recognition_threshold
                    )
                    
                    name = "Unknown"
                    confidence = 0
                    
                    face_distances = face_recognition.face_distance(list(self.known_faces.values()), face_encoding)
                    
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = list(self.known_faces.keys())[best_match_index]
                            confidence = (1 - face_distances[best_match_index]) * 100
                    
                    # Draw rectangle around face
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    
                    # Draw label
                    label = f"{name}"
                    if name != "Unknown":
                        label += f" ({confidence:.1f}%)"
                    
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                    cv2.putText(frame, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                
                self.last_recognition_time = current_time
            
        except Exception as e:
            print(f"❌ Continuous face recognition error: {e}")
        
        return frame
    
    def process_text_reading_command(self, command):
        """Process text reading commands"""
        try:
            if "read text" in command or "read this" in command:
                self.speak("Reading text from current view")
                self.read_text_from_camera()
                return "read_text"
                
            elif "start reading" in command or "continuous reading" in command:
                self.text_reading_active = True
                self.speak("Starting continuous text reading")
                return "start_reading"
                
            elif "stop reading" in command:
                self.text_reading_active = False
                self.speak("Text reading stopped")
                return "stop_reading"
                
            elif "read again" in command or "repeat" in command:
                if self.last_read_text:
                    self.speak(f"Repeating: {self.last_read_text}")
                else:
                    self.speak("No previous text to repeat")
                return "read_again"
                
            elif "read slowly" in command or "slow reading" in command:
                self.tts.setProperty('rate', 100)
                self.speak("Reading speed set to slow")
                return "speed_change"
                
            elif "read fast" in command or "fast reading" in command:
                self.tts.setProperty('rate', 200)
                self.speak("Reading speed set to fast")
                return "speed_change"
                
            elif "normal speed" in command:
                self.tts.setProperty('rate', 150)
                self.speak("Reading speed set to normal")
                return "speed_change"
                
            elif "change language" in command:
                # Extract language from command
                if "english" in command:
                    self.reading_language = 'eng'
                    self.speak("OCR language set to English")
                elif "spanish" in command:
                    self.reading_language = 'spa'
                    self.speak("OCR language set to Spanish")
                elif "french" in command:
                    self.reading_language = 'fra'
                    self.speak("OCR language set to French")
                elif "german" in command:
                    self.reading_language = 'deu'
                    self.speak("OCR language set to German")
                else:
                    self.speak("Language not recognized. Using English.")
                    self.reading_language = 'eng'
                return "language_change"
                
            else:
                self.speak("Text reading command not recognized")
                return None
                
        except Exception as e:
            print(f"❌ Text reading command error: {e}")
            self.speak("Error in text reading command")
            return None
    
    def read_text_from_camera(self):
        """Read text from current camera view using OCR"""
        frame = self.capture_frame()
        
        if frame is None:
            self.speak("Camera not available. Cannot read text.")
            return
        
        try:
            # Convert frame to PIL Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Preprocess image for better OCR
            processed_image = self.preprocess_for_ocr(pil_image)
            
            # Extract text using Tesseract OCR
            extracted_text = pytesseract.image_to_string(
                processed_image, 
                lang=self.reading_language,
                config='--psm 6'  # Assume a single uniform block of text
            ).strip()
            
            if extracted_text:
                self.last_read_text = extracted_text
                # Clean up text for better speech
                cleaned_text = self.clean_text_for_speech(extracted_text)
                
                if len(cleaned_text) > 500:  # If text is too long
                    self.speak("Found long text. Reading first part.")
                    cleaned_text = cleaned_text[:500] + "... text continues"
                
                print(f"📖 Extracted text: {extracted_text}")
                self.speak(cleaned_text)
                
            else:
                self.speak("No readable text found in the current view")
                print("📖 No text detected")
                
        except Exception as e:
            print(f"❌ Text reading error: {e}")
            self.speak("Error reading text. Please ensure text is clear and well-lit.")
    
    def preprocess_for_ocr(self, image):
        """Preprocess image for better OCR results"""
        # Convert PIL to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
        
        # Convert back to PIL
        return Image.fromarray(processed)
    
    def clean_text_for_speech(self, text):
        """Clean extracted text for better speech synthesis"""
        # Remove excessive whitespace
        cleaned = ' '.join(text.split())
        
        # Replace common OCR mistakes
        replacements = {
            '|': 'I',
            '0': 'O',
            '5': 'S',
            '@': 'a',
            '3': 'e',
            '1': 'l',
        }
        
        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)
        
        # Remove special characters that don't read well
        import re
        cleaned = re.sub(r'[^\w\s\.,!?;:]', ' ', cleaned)
        
        return cleaned
    
    def process_detection_command(self, command):
        """Process object detection commands"""
        try:
            if "start detection" in command:
                self.detection_mode = "continuous"
                self.speak("Starting continuous object detection")
                return "detection_start"
                
            elif "stop detection" in command:
                self.detection_mode = "paused"
                self.speak("Object detection paused")
                return "detection_stop"
                
            elif "what do you see" in command or "describe" in command:
                self.speak("Analyzing current view")
                self.describe_current_view()
                return "describe"
                
            elif "find" in command:
                parts = command.split("find")
                if len(parts) > 1:
                    target = parts[1].strip()
                    self.target_objects.add(target)
                    self.detection_mode = "selective"
                    self.speak(f"Looking for {target}")
                    self.find_specific_object(target)
                else:
                    self.speak("Please specify what to find")
                return "find"
                    
            elif "count" in command:
                parts = command.split("count")
                if len(parts) > 1:
                    target = parts[1].strip()
                    self.speak(f"Counting {target}")
                    self.count_specific_objects(target)
                else:
                    self.speak("Please specify what to count")
                return "count"
                    
            else:
                self.speak("Detection command not recognized")
                return None
                
        except Exception as e:
            print(f"❌ Detection command error: {e}")
            self.speak("Error in detection command")
            return None
    
    def process_gps_command(self, command):
        """Process GPS navigation commands"""
        try:
            if "where am i" in command:
                self.get_current_location()
                return "location"
                
            elif "navigate to" in command:
                destination = command.split("navigate to")[-1].strip()
                self.start_navigation(destination)
                return "navigate"
                
            elif "save location as" in command:
                location_name = command.split("save location as")[-1].strip()
                self.save_current_location(location_name)
                return "save_location"
                
            elif "distance to" in command:
                destination = command.split("distance to")[-1].strip()
                self.get_distance_to(destination)
                return "distance"
                
            elif "stop navigation" in command:
                self.stop_navigation()
                return "stop_navigation"
                
            elif "my coordinates" in command:
                self.get_coordinates()
                return "coordinates"
                
            else:
                self.speak("GPS command not recognized")
                return None
                
        except Exception as e:
            print(f"❌ GPS command error: {e}")
            self.speak("Error in GPS command")
            return None
    
    def capture_frame(self):
        """Capture a frame from camera"""
        if not self.camera_initialized:
            if not self.initialize_camera():
                return None
                
        try:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                return frame
            else:
                print("❌ Failed to capture frame")
                return None
        except Exception as e:
            print(f"❌ Frame capture error: {e}")
            return None
    
    def describe_current_view(self):
        """Describe what's currently visible"""
        frame = self.capture_frame()
        
        if frame is None:
            self.speak("Camera not available. Cannot analyze view.")
            return
        
        if self.model is None:
            self.speak("Detection model not available")
            return
        
        try:
            results = self.model(frame, conf=self.confidence_threshold)
            detected_objects = self.process_detections(results[0])
            
            if not detected_objects:
                self.speak("I don't see any recognizable objects")
                return
            
            # Group objects by class
            object_counts = defaultdict(int)
            for obj in detected_objects:
                object_counts[obj['name']] += 1
            
            description = "I can see "
            items = []
            for obj_class, count in object_counts.items():
                if count == 1:
                    items.append(f"a {obj_class}")
                else:
                    items.append(f"{count} {obj_class}s")
            
            if len(items) == 1:
                description += items[0]
            elif len(items) == 2:
                description += f"{items[0]} and {items[1]}"
            else:
                description += ", ".join(items[:-1]) + f", and {items[-1]}"
            
            self.speak(description)
            print(f"🔍 Detection results: {description}")
            
        except Exception as e:
            print(f"❌ Description error: {e}")
            self.speak("Error analyzing view")
    
    def find_specific_object(self, target_object):
        """Find a specific object"""
        frame = self.capture_frame()
        
        if frame is None:
            self.speak("Camera not available")
            return
        
        if self.model is None:
            self.speak("Detection model not available")
            return
        
        try:
            results = self.model(frame, conf=self.confidence_threshold)
            detected_objects = self.process_detections(results[0])
            
            found_objects = []
            for obj in detected_objects:
                if target_object.lower() in obj['name'].lower():
                    found_objects.append(obj)
            
            if found_objects:
                count = len(found_objects)
                if count == 1:
                    position = found_objects[0]['position']
                    self.speak(f"I found a {target_object} at {position}")
                else:
                    self.speak(f"I found {count} {target_object}s")
                print(f"✅ Found {count} {target_object}(s)")
            else:
                self.speak(f"I don't see any {target_object}")
                print(f"❌ No {target_object} found")
                
        except Exception as e:
            print(f"❌ Find object error: {e}")
            self.speak("Error finding object")
    
    def count_specific_objects(self, target_object):
        """Count specific objects"""
        frame = self.capture_frame()
        
        if frame is None:
            self.speak("Camera not available")
            return
        
        if self.model is None:
            self.speak("Detection model not available")
            return
        
        try:
            results = self.model(frame, conf=self.confidence_threshold)
            detected_objects = self.process_detections(results[0])
            
            count = 0
            for obj in detected_objects:
                if target_object.lower() in obj['name'].lower():
                    count += 1
            
            if count == 0:
                self.speak(f"I don't see any {target_object}")
            elif count == 1:
                self.speak(f"I see one {target_object}")
            else:
                self.speak(f"I see {count} {target_object}s")
            
            print(f"📊 Count result: {count} {target_object}(s)")
            
        except Exception as e:
            print(f"❌ Count error: {e}")
            self.speak("Error counting objects")
    
    def test_camera(self):
        """Test camera functionality"""
        self.speak("Testing camera")
        frame = self.capture_frame()
        
        if frame is not None:
            self.speak("Camera is working correctly")
            cv2.imshow("Camera Test", frame)
            cv2.waitKey(2000)  # Show for 2 seconds
            cv2.destroyAllWindows()
        else:
            self.speak("Camera is not working. Please check camera permissions.")
    
    def get_system_status(self):
        """Get and speak system status"""
        camera_status = "working" if self.camera_initialized else "not available"
        model_status = "loaded" if self.model else "not loaded"
        reading_status = "active" if self.text_reading_active else "inactive"
        face_status = "active" if self.face_recognition_active else "inactive"
        face_count = len(self.known_faces)
        
        status_parts = [
            f"Current mode: {self.current_mode.replace('_', ' ')}",
            f"Detection: {self.detection_mode}",
            f"Camera: {camera_status}",
            f"Model: {model_status}",
            f"Text reading: {reading_status}",
            f"Face recognition: {face_status}",
            f"Known faces: {face_count}",
            f"Navigation: {'Active' if self.is_navigating else 'Inactive'}"
        ]
        
        status_text = "System status: " + ", ".join(status_parts)
        self.speak(status_text)
        print(f"ℹ️ {status_text}")
    
    # Object Detection Methods
    def process_detections(self, results):
        """Process YOLO detection results"""
        detected_objects = []
        
        if results.boxes is not None:
            for box in results.boxes:
                confidence = float(box.conf[0])
                
                if confidence > self.confidence_threshold:
                    class_id = int(box.cls[0])
                    class_name = results.names[class_id]
                    
                    box_coords = box.xyxy[0].tolist()
                    position = self.get_object_position(box_coords)
                    
                    detected_objects.append({
                        'name': class_name,
                        'confidence': confidence,
                        'position': position,
                        'coords': box_coords
                    })
        
        return detected_objects
    
    def get_object_position(self, box_coords):
        """Get relative position of object"""
        x1, y1, x2, y2 = box_coords
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        if center_x < self.frame_width / 3:
            h_pos = "left"
        elif center_x > 2 * self.frame_width / 3:
            h_pos = "right"
        else:
            h_pos = "center"
        
        if center_y < self.frame_height / 3:
            v_pos = "top"
        elif center_y > 2 * self.frame_height / 3:
            v_pos = "bottom"
        else:
            v_pos = "middle"
        
        return f"{v_pos} {h_pos}"
    
    def draw_boxes(self, image, results):
        """Draw detection boxes on image"""
        annotated_image = image.copy()
        
        if results.boxes is not None:
            for box in results.boxes:
                confidence = float(box.conf[0])
                
                if confidence > self.confidence_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls[0])
                    class_name = results.names[class_id]
                    
                    # Color based on confidence
                    if confidence > 0.7:
                        color = (0, 255, 0)  # Green
                    elif confidence > 0.5:
                        color = (0, 255, 255)  # Yellow
                    else:
                        color = (0, 165, 255)  # Orange
                    
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                    
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(annotated_image, label, (x1, y1-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return annotated_image
    
    # GPS Methods
    def get_current_location(self):
        """Get current location"""
        try:
            self.speak("Getting your current location")
            response = requests.get('http://ip-api.com/json/', timeout=5)
            data = response.json()
            
            if data['status'] == 'success':
                self.current_location = {
                    'city': data['city'],
                    'region': data['regionName'],
                    'country': data['country'],
                    'lat': data['lat'],
                    'lon': data['lon']
                }
                
                location_text = f"You are in {data['city']}, {data['regionName']}, {data['country']}"
                self.speak(location_text)
                print(f"📍 Location: {location_text}")
            else:
                self.speak("Unable to determine your location")
                
        except Exception as e:
            print(f"❌ Location error: {e}")
            self.speak("Error getting location. Please check your internet connection.")
    
    def start_navigation(self, destination):
        """Start navigation"""
        self.speak(f"Starting navigation to {destination}")
        self.is_navigating = True
        print(f"🧭 Navigation started to: {destination}")
        # In a real implementation, this would start turn-by-turn navigation
    
    def stop_navigation(self):
        """Stop navigation"""
        self.is_navigating = False
        self.speak("Navigation stopped")
        print("🛑 Navigation stopped")
    
    def save_current_location(self, name):
        """Save current location"""
        if self.current_location:
            self.saved_locations[name] = self.current_location.copy()
            self.speak(f"Location saved as {name}")
            print(f"💾 Location saved as: {name}")
        else:
            self.speak("Current location not available. Please get location first.")
    
    def get_distance_to(self, destination):
        """Get distance to destination"""
        self.speak(f"Calculating distance to {destination}")
        print(f"📏 Distance calculation to: {destination}")
        # In a real implementation, this would calculate actual distance
        self.speak("Distance calculation feature coming soon")
    
    def get_coordinates(self):
        """Get GPS coordinates"""
        if self.current_location:
            lat = self.current_location['lat']
            lon = self.current_location['lon']
            self.speak(f"Your coordinates are {lat:.4f} latitude, {lon:.4f} longitude")
            print(f"🌐 Coordinates: {lat:.4f}, {lon:.4f}")
        else:
            self.speak("Location not available. Please get your location first.")
    
    def speak(self, message):
        """Text-to-speech output with error handling"""
        try:
            print(f"🔊 {message}")
            if self.tts:
                self.tts.say(message)
                self.tts.runAndWait()
        except Exception as e:
            print(f"❌ Speech error: {e}")
    
    def add_system_overlay(self, frame):
        """Add system information overlay to frame"""
        # Mode indicator
        mode_colors = {
            "detection": (0, 255, 0),
            "gps": (255, 0, 0),
            "text_reading": (255, 0, 255),
            "face_recognition": (255, 255, 0)
        }
        mode_color = mode_colors.get(self.current_mode, (255, 255, 255))
        
        cv2.rectangle(frame, (10, 10), (300, 40), (0, 0, 0), -1)
        cv2.putText(frame, f"Mode: {self.current_mode.replace('_', ' ').upper()}", (15, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
        
        # Mode-specific status
        if self.current_mode == "detection":
            status_text = f"Detection: {self.detection_mode} | Conf: {self.confidence_threshold:.1f}"
            cv2.rectangle(frame, (10, 45), (400, 70), (0, 0, 0), -1)
            cv2.putText(frame, status_text, (15, 62), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        elif self.current_mode == "gps":
            gps_status = "Navigation: ON" if self.is_navigating else "Navigation: OFF"
            cv2.rectangle(frame, (10, 45), (300, 70), (0, 0, 0), -1)
            cv2.putText(frame, gps_status, (15, 62), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        elif self.current_mode == "text_reading":
            reading_status = "Reading: ON" if self.text_reading_active else "Reading: OFF"
            cv2.rectangle(frame, (10, 45), (300, 70), (0, 0, 0), -1)
            cv2.putText(frame, reading_status, (15, 62), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        
        elif self.current_mode == "face_recognition":
            face_status = f"Recognition: {'ON' if self.face_recognition_active else 'OFF'} | Known: {len(self.known_faces)}"
            cv2.rectangle(frame, (10, 45), (400, 70), (0, 0, 0), -1)
            cv2.putText(frame, face_status, (15, 62), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Controls
        cv2.putText(frame, "Press 'm' to switch mode, 'f' for face recognition, 'q' to quit", 
                   (15, frame.shape[0] - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Voice control indicator
        voice_status = "🎤 LISTENING" if self.listening else "🎤 OFF"
        cv2.putText(frame, voice_status, (frame.shape[1] - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    def run_integrated_system(self):
        """Run the integrated smart glasses system"""
        print("\n🎥 Starting Integrated Smart Glasses System...")
        print("Press 'q' to quit, 'm' to toggle mode, 'f' for face recognition, 'r' for text reading")
        
        try:
            # Initialize camera and start voice control
            self.initialize_camera()
            self.start_voice_control()
            
            if not self.camera_initialized:
                print("⚠️ Camera not available - running in audio-only mode")
                self.speak("Camera not available. Running in voice-only mode. Say help for commands.")
                
                # Keep system running for voice commands only
                while self.listening:
                    time.sleep(1)
                return
            
            last_detection_time = 0
            last_reading_time = 0
            detection_interval = 2.0
            reading_interval = 3.0
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to read from camera")
                    break
                
                current_time = time.time()
                annotated_frame = frame.copy()
                
                # Object detection processing (only in detection mode)
                if self.current_mode == "detection" and self.model:
                    should_detect = False
                    
                    if self.detection_mode == "continuous":
                        if current_time - last_detection_time > detection_interval:
                            should_detect = True
                            last_detection_time = current_time
                    
                    if should_detect:
                        try:
                            results = self.model(frame, conf=self.confidence_threshold)
                            detected_objects = self.process_detections(results[0])
                            annotated_frame = self.draw_boxes(frame, results[0])
                            
                            if detected_objects:
                                self.speak_detections(detected_objects)
                        except Exception as e:
                            print(f"❌ Detection error: {e}")
                
                # Text reading processing (only in text_reading mode)
                elif self.current_mode == "text_reading" and self.text_reading_active:
                    if current_time - last_reading_time > reading_interval:
                        try:
                            self.read_text_from_camera()
                            last_reading_time = current_time
                        except Exception as e:
                            print(f"❌ Text reading error: {e}")
                
                # Face recognition processing (only in face_recognition mode)
                elif self.current_mode == "face_recognition":
                    if self.face_recognition_active:
                        annotated_frame = self.process_continuous_face_recognition(annotated_frame)
                
                # Add system overlay
                self.add_system_overlay(annotated_frame)
                
                # Display frame
                window_title = f"Smart Glasses - {self.current_mode.replace('_', ' ').title()} Mode"
                cv2.imshow(window_title, annotated_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('m'):  # Toggle mode
                    modes = ["detection", "gps", "text_reading", "face_recognition"]
                    current_idx = modes.index(self.current_mode)
                    next_idx = (current_idx + 1) % len(modes)
                    self.current_mode = modes[next_idx]
                    self.text_reading_active = False  # Reset reading state
                    self.face_recognition_active = False  # Reset face recognition state
                    self.speak(f"Switched to {self.current_mode.replace('_', ' ')} mode")
                elif key == ord('r'):  # Instant text reading
                    if self.current_mode != "text_reading":
                        self.current_mode = "text_reading"
                        self.speak("Switched to text reading mode")
                    self.read_text_from_camera()
                elif key == ord('f'):  # Instant face recognition
                    if self.current_mode != "face_recognition":
                        self.current_mode = "face_recognition"
                        self.speak("Switched to face recognition mode")
                    self.identify_person()
        
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        finally:
            self.shutdown()
    
    def speak_detections(self, detected_objects):
        """Speak detected objects"""
        if not detected_objects:
            return
        
        # Group by object type
        object_counts = defaultdict(list)
        for obj in detected_objects:
            object_counts[obj['name']].append(obj['position'])
        
        speech_parts = []
        for obj_name, positions in object_counts.items():
            if len(positions) == 1:
                speech_parts.append(f"{obj_name} on the {positions[0]}")
            else:
                speech_parts.append(f"{len(positions)} {obj_name}s")
        
        if speech_parts:
            message = "Detected: " + ", ".join(speech_parts[:3])  # Limit to 3 items
            self.speak(message)
    
    def shutdown(self):
        """Shutdown the entire system"""
        print("🔄 Shutting down Smart Glasses system...")
        
        self.stop_voice_control()
        
        # Save face database before shutdown
        if self.known_faces:
            self.save_face_database()
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        print("✅ System shutdown complete")

# Main execution
if __name__ == "__main__":
    try:
        print("🚀 Welcome to Smart Glasses with Facial Recognition!")
        print("Make sure you have all required packages installed:")
        print("- Tesseract OCR for text reading")
        print("- dlib and face_recognition for facial recognition")
        print("\nInstallation commands:")
        print("pip install ultralytics opencv-python pyttsx3 speechrecognition pyaudio requests pytesseract pillow face_recognition")
        print("\nOn Windows: Download Tesseract from https://github.com/UB-Mannheim/tesseract/wiki")
        print("On Ubuntu: sudo apt install tesseract-ocr")
        print("On macOS: brew install tesseract")
        
        smart_glasses = IntegratedSmartGlasses()
        smart_glasses.run_integrated_system()
        
    except KeyboardInterrupt:
        print("\n⚠️ Shutting down Smart Glasses...")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\n📦 Make sure you have all required packages installed:")
        print("pip install ultralytics opencv-python pyttsx3 speechrecognition pyaudio requests pytesseract pillow face_recognition")
        print("\n🔧 Also install system dependencies:")
        print("- Tesseract OCR for text reading")
        print("- dlib for facial recognition (may require cmake and visual studio build tools on Windows)")
    finally:
        print("👋 Thank you for using Smart Glasses!")
