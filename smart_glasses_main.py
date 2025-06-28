"""
Smart Glasses Main Integration - FIXED VERSION
Combines object detection and GPS functionality with unified voice control
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
import torchvision

class IntegratedSmartGlasses:
    def __init__(self):  # Fixed: was _init_
        print(" Initializing Integrated Smart Glasses System...")
        
        # Load YOLOv8 model for object detection
        print("📱 Loading YOLOv8 model...")
        try:
            self.model = YOLO("yolov8n.pt")
            print(" YOLOv8 model loaded successfully!")
        except Exception as e:
            print(f" Error loading YOLO model: {e}")
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
            print(" TTS initialized successfully!")
        except Exception as e:
            print(f" TTS initialization error: {e}")
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
            print(" Speech recognition ready!")
        except Exception as e:
            print(f" Microphone setup error: {e}")
            self.microphone = None
        
        # System state
        self.current_mode = "detection"  # "detection" or "gps"
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
        
        # Camera setup
        self.cap = None
        self.camera_initialized = False
        
        print(" Integrated Smart Glasses initialized!")
        self.show_all_commands()
    
    def initialize_camera(self):
        """Initialize camera with fallback options"""
        if self.camera_initialized and self.cap and self.cap.isOpened():
            return True
            
        print(" Initializing camera...")
        
        # Try different camera indices
        for idx in [0, 1, -1, 2]:
            try:
                print(f"   Trying camera index {idx}...")
                self.cap = cv2.VideoCapture(idx)
                
                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        print(f" Camera initialized with index {idx}")
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
                    
        print(" Could not initialize camera")
        self.camera_initialized = False
        return False
    
    def show_all_commands(self):
        """Display all available voice commands"""
        print("\n INTEGRATED SMART GLASSES COMMANDS:")
        print("=" * 60)
        
        print(" MODE SWITCHING:")
        print("• 'switch to detection' - Object detection mode")
        print("• 'switch to gps' - GPS navigation mode") 
        print("• 'what mode' - Current mode status")
        
        print("\n OBJECT DETECTION COMMANDS:")
        print("• 'start detection' - Begin continuous detection")
        print("• 'stop detection' - Pause detection")
        print("• 'what do you see' - Describe current view")
        print("• 'find [object]' - Look for specific object")
        print("• 'count [objects]' - Count specific objects")
        
        print("\n GPS NAVIGATION COMMANDS:")
        print("• 'where am i' - Get current location")
        print("• 'navigate to [place]' - Start navigation")
        print("• 'save location as [name]' - Save current location")
        print("• 'distance to [place]' - Get distance")
        print("• 'stop navigation' - End navigation")
        
        print("\n SYSTEM COMMANDS:")
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
        print(" Voice control stopped.")
    
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
                self.speak("Switched to object detection mode")
                return "mode_switch"
            
            elif "switch to gps" in command:
                self.current_mode = "gps"
                self.speak("Switched to GPS navigation mode")
                return "mode_switch"
            
            elif "what mode" in command:
                self.speak(f"Currently in {self.current_mode} mode")
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
            else:
                self.speak("Command not recognized. Say help for available commands.")
                return None
            
        except Exception as e:
            print(f"❌ Command processing error: {e}")
            self.speak("Error processing command")
            return None
    
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
                print(" Failed to capture frame")
                return None
        except Exception as e:
            print(f" Frame capture error: {e}")
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
            print(f" Description error: {e}")
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
                print(f" Found {count} {target_object}(s)")
            else:
                self.speak(f"I don't see any {target_object}")
                print(f" No {target_object} found")
                
        except Exception as e:
            print(f" Find object error: {e}")
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
            print(f" Count error: {e}")
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
        
        status_parts = [
            f"Current mode: {self.current_mode}",
            f"Detection: {self.detection_mode}",
            f"Camera: {camera_status}",
            f"Model: {model_status}",
            f"Navigation: {'Active' if self.is_navigating else 'Inactive'}"
        ]
        
        status_text = "System status: " + ", ".join(status_parts)
        self.speak(status_text)
        print(f" {status_text}")
    
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
                print(f" Location: {location_text}")
            else:
                self.speak("Unable to determine your location")
                
        except Exception as e:
            print(f" Location error: {e}")
            self.speak("Error getting location. Please check your internet connection.")
    
    def start_navigation(self, destination):
        """Start navigation"""
        self.speak(f"Starting navigation to {destination}")
        self.is_navigating = True
        print(f" Navigation started to: {destination}")
        # In a real implementation, this would start turn-by-turn navigation
    
    def stop_navigation(self):
        """Stop navigation"""
        self.is_navigating = False
        self.speak("Navigation stopped")
        print(" Navigation stopped")
    
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
        print(f" Distance calculation to: {destination}")
        # In a real implementation, this would calculate actual distance
        self.speak("Distance calculation feature coming soon")
    
    def get_coordinates(self):
        """Get GPS coordinates"""
        if self.current_location:
            lat = self.current_location['lat']
            lon = self.current_location['lon']
            self.speak(f"Your coordinates are {lat:.4f} latitude, {lon:.4f} longitude")
            print(f" Coordinates: {lat:.4f}, {lon:.4f}")
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
            print(f" Speech error: {e}")
    
    def add_system_overlay(self, frame):
        """Add system information overlay to frame"""
        # Mode indicator
        mode_color = (0, 255, 0) if self.current_mode == "detection" else (255, 0, 0)
        cv2.rectangle(frame, (10, 10), (200, 40), (0, 0, 0), -1)
        cv2.putText(frame, f"Mode: {self.current_mode.upper()}", (15, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
        
        # Detection status (if in detection mode)
        if self.current_mode == "detection":
            status_text = f"Detection: {self.detection_mode} | Conf: {self.confidence_threshold:.1f}"
            cv2.rectangle(frame, (10, 45), (400, 70), (0, 0, 0), -1)
            cv2.putText(frame, status_text, (15, 62), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # GPS status (if in GPS mode)
        elif self.current_mode == "gps":
            gps_status = "Navigation: ON" if self.is_navigating else "Navigation: OFF"
            cv2.rectangle(frame, (10, 45), (300, 70), (0, 0, 0), -1)
            cv2.putText(frame, gps_status, (15, 62), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Controls
        cv2.putText(frame, "Press 'm' to switch mode, 'q' to quit", 
                   (15, frame.shape[0] - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Voice control indicator
        voice_status = "🎤 LISTENING" if self.listening else "🎤 OFF"
        cv2.putText(frame, voice_status, (frame.shape[1] - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    def run_integrated_system(self):
        """Run the integrated smart glasses system"""
        print("\n🎥 Starting Integrated Smart Glasses System...")
        print("Press 'q' to quit, 'm' to toggle mode")
        
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
            detection_interval = 2.0
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print(" Failed to read from camera")
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
                            print(f" Detection error: {e}")
                
                # Add system overlay
                self.add_system_overlay(annotated_frame)
                
                # Display frame
                window_title = f"Smart Glasses - {self.current_mode.title()} Mode"
                cv2.imshow(window_title, annotated_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('m'):  # Toggle mode
                    self.current_mode = "gps" if self.current_mode == "detection" else "detection"
                    self.speak(f"Switched to {self.current_mode} mode")
        
        except KeyboardInterrupt:
            print("\n Interrupted by user")
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
        print(" Shutting down Smart Glasses system...")
        
        self.stop_voice_control()
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        print(" System shutdown complete")

# Main execution
if __name__ == "__main__":  # Fixed: was _name_ == "_main_"
    try:
        smart_glasses = IntegratedSmartGlasses()
        smart_glasses.run_integrated_system()
        
    except KeyboardInterrupt:
        print("\n Shutting down Smart Glasses...")
    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()
        print("\n Make sure you have all required packages installed:")
        print("pip install ultralytics opencv-python pyttsx3 speechrecognition pyaudio requests")
    finally:
        print(" Thank you for using Smart Glasses!")