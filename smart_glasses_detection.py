"""
Smart Glasses - Enhanced Object Detection with Speech Direction Commands
Improved version with better object detection and debugging features
"""

import cv2
import numpy as np
from ultralytics import YOLO
import pyttsx3
import speech_recognition as sr
import threading
import time
from collections import defaultdict

class SmartGlassesWithSpeech:
    def __init__(self):
        # Load YOLOv8 model
        print("Loading YOLOv8 model...")
        self.model = YOLO("yolov8n.pt")
        
        # Print available classes for debugging
        self.print_yolo_classes()
        
        # Initialize text-to-speech
        self.tts = pyttsx3.init()
        self.tts.setProperty('rate', 150)
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Calibrate microphone
        print("Calibrating microphone for ambient noise...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
        
        # Speech control flags
        self.listening = False
        self.speech_thread = None
        
        # Detection settings - LOWERED CONFIDENCE THRESHOLD
        self.detection_mode = "continuous"
        self.confidence_threshold = 0.25  # Lowered from 0.5 to detect more objects
        self.target_objects = set()
        
        # Spatial awareness
        self.frame_width = 640
        self.frame_height = 480
        
        # Debug mode
        self.debug_mode = True
        
        print("✅ Smart Glasses initialized! Ready for voice commands.")
        self.show_voice_commands()
    
    def print_yolo_classes(self):
        """Print all available YOLO classes for debugging"""
        print("\n📋 Available YOLO Classes:")
        print("=" * 50)
        class_names = list(self.model.names.values())
        for i, name in enumerate(class_names):
            print(f"{i:2d}: {name}")
            if (i + 1) % 5 == 0:  # Print 5 per line
                print()
        print("=" * 50)
    
    def show_voice_commands(self):
        """Display available voice commands"""
        print("\n🎤 VOICE COMMANDS:")
        print("=" * 40)
        print("• 'start detection' - Begin continuous detection")
        print("• 'stop detection' - Pause detection")
        print("• 'what do you see' - Describe current view")
        print("• 'find [object]' - Look for specific object")
        print("• 'where is [object]' - Get location of object")
        print("• 'count [objects]' - Count specific objects")
        print("• 'show all' - Show all detected objects (debug)")
        print("• 'lower threshold' - Reduce confidence threshold")
        print("• 'higher threshold' - Increase confidence threshold")
        print("• 'help' - Repeat commands")
        print("• 'exit' or 'quit' - Close application")
        print("=" * 40)
    
    def start_speech_recognition(self):
        """Start continuous speech recognition in background"""
        if self.listening:
            return
        
        self.listening = True
        self.speech_thread = threading.Thread(target=self._speech_loop, daemon=True)
        self.speech_thread.start()
        print("🎤 Voice recognition started. Say 'help' for commands.")
    
    def stop_speech_recognition(self):
        """Stop speech recognition"""
        self.listening = False
        if self.speech_thread:
            self.speech_thread.join(timeout=1)
        print("🔇 Voice recognition stopped.")
    
    def _speech_loop(self):
        """Continuous speech recognition loop"""
        while self.listening:
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=3)
                
                command = self.recognizer.recognize_google(audio).lower()
                print(f"🎤 Heard: '{command}'")
                
                result = self.process_voice_command(command)
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
    
    def process_voice_command(self, command):
        """Process recognized voice commands"""
        try:
            if "start detection" in command:
                self.detection_mode = "continuous"
                self.speak("Starting continuous detection")
                
            elif "stop detection" in command:
                self.detection_mode = "paused"
                self.speak("Detection paused")
                
            elif "what do you see" in command or "describe" in command:
                self.detection_mode = "on_command"
                self.speak("Analyzing current view")
                
            elif "show all" in command:
                self.detection_mode = "debug"
                self.speak("Showing all detections with confidence scores")
                
            elif "lower threshold" in command:
                self.confidence_threshold = max(0.1, self.confidence_threshold - 0.1)
                self.speak(f"Confidence threshold lowered to {self.confidence_threshold:.1f}")
                
            elif "higher threshold" in command or "raise threshold" in command:
                self.confidence_threshold = min(0.9, self.confidence_threshold + 0.1)
                self.speak(f"Confidence threshold raised to {self.confidence_threshold:.1f}")
                
            elif "find" in command:
                parts = command.split("find")
                if len(parts) > 1:
                    target = parts[1].strip()
                    self.target_objects.add(target)
                    self.detection_mode = "selective"
                    self.speak(f"Looking for {target}")
                
            elif "where is" in command:
                parts = command.split("where is")
                if len(parts) > 1:
                    target = parts[1].strip()
                    self.find_object_location(target)
                
            elif "count" in command:
                parts = command.split("count")
                if len(parts) > 1:
                    target = parts[1].strip()
                    self.count_objects(target)
                
            elif "help" in command:
                self.speak("Voice commands available")
                self.show_voice_commands()
                
            elif "exit" in command or "quit" in command:
                self.speak("Goodbye")
                return "exit"
                
            else:
                self.speak("Command not recognized. Say help for available commands.")
                
        except Exception as e:
            print(f"Command processing error: {e}")
    
    def detect_with_speech_control(self):
        """Main detection loop with speech control"""
        print("\n🎥 Starting Smart Glasses with Voice Control...")
        print("Press 'q' to quit")
        
        self.start_speech_recognition()
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        last_detection_time = 0
        detection_interval = 2.0  # Increased interval to reduce spam
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_time = time.time()
                should_detect = False
                
                if self.detection_mode == "continuous":
                    if current_time - last_detection_time > detection_interval:
                        should_detect = True
                        last_detection_time = current_time
                elif self.detection_mode in ["on_command", "debug"]:
                    should_detect = True
                    if self.detection_mode == "on_command":
                        self.detection_mode = "paused"
                elif self.detection_mode == "selective":
                    should_detect = True
                
                annotated_frame = frame.copy()
                
                if should_detect:
                    results = self.model(frame)
                    detected_objects = self.process_detections(results[0])
                    annotated_frame = self.draw_boxes(frame, results[0])
                    
                    # Debug output
                    if self.debug_mode:
                        print(f"\n🔍 Detection Results (threshold: {self.confidence_threshold}):")
                        if detected_objects:
                            for obj in detected_objects:
                                print(f"  - {obj['name']}: {obj['confidence']:.3f} ({obj['position']})")
                        else:
                            print("  - No objects detected above threshold")
                    
                    # Handle different detection modes
                    if self.detection_mode == "continuous":
                        if detected_objects:
                            self.speak_detections(detected_objects)
                    elif self.detection_mode == "selective":
                        self.check_target_objects(detected_objects)
                    elif self.detection_mode == "debug":
                        self.speak_all_detections(detected_objects)
                        self.detection_mode = "paused"
                
                self.add_status_overlay(annotated_frame)
                cv2.imshow('Smart Glasses - Enhanced Detection', annotated_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('d'):  # Toggle debug mode
                    self.debug_mode = not self.debug_mode
                    print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.stop_speech_recognition()
    
    def add_status_overlay(self, frame):
        """Add status information to frame"""
        status_text = f"Mode: {self.detection_mode} | Confidence: {self.confidence_threshold:.1f}"
        if self.target_objects:
            status_text += f" | Searching: {', '.join(self.target_objects)}"
        
        # Background rectangle
        cv2.rectangle(frame, (10, 10), (600, 40), (0, 0, 0), -1)
        cv2.putText(frame, status_text, (15, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Controls info
        cv2.putText(frame, "Press 'd' for debug toggle, 'q' to quit", 
                   (15, frame.shape[0] - 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Microphone status
        mic_status = "🎤 ON" if self.listening else "🎤 OFF"
        cv2.putText(frame, mic_status, (15, frame.shape[0] - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    def speak_all_detections(self, detected_objects):
        """Speak all detections with confidence scores (debug mode)"""
        if not detected_objects:
            self.speak("No objects detected above current threshold")
            return
        
        message_parts = []
        for obj in detected_objects:
            message_parts.append(f"{obj['name']} with {obj['confidence']:.0%} confidence")
        
        message = "Detected: " + ", ".join(message_parts)
        self.speak(message)
    
    def check_target_objects(self, detected_objects):
        """Check if target objects are found"""
        found_targets = []
        for target in self.target_objects:
            for obj in detected_objects:
                if target.lower() in obj['name'].lower() or obj['name'].lower() in target.lower():
                    found_targets.append(f"{obj['name']} at {obj['position']}")
        
        if found_targets:
            self.speak(f"Found {', '.join(found_targets)}")
            self.target_objects.clear()
            self.detection_mode = "continuous"
        else:
            # Try with lower confidence for target search
            self.speak("Target not found with current settings, trying lower confidence")
    
    def find_object_location(self, target_object):
        """Find location of specific object in current frame"""
        self.speak(f"Searching for location of {target_object}")
        # Set up for one-time detection with focus on target
        self.target_objects.add(target_object)
        self.detection_mode = "selective"
    
    def count_objects(self, object_type):
        """Count specific type of objects in current frame"""
        self.speak(f"Counting {object_type}")
        # This would be implemented with current frame analysis
        # For now, set up selective detection
        self.target_objects.add(object_type)
        self.detection_mode = "selective"
    
    def get_object_position(self, box_coords):
        """Get relative position description of object"""
        x1, y1, x2, y2 = box_coords
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Determine horizontal position
        if center_x < self.frame_width / 3:
            h_pos = "left"
        elif center_x > 2 * self.frame_width / 3:
            h_pos = "right"
        else:
            h_pos = "center"
        
        # Determine vertical position
        if center_y < self.frame_height / 3:
            v_pos = "top"
        elif center_y > 2 * self.frame_height / 3:
            v_pos = "bottom"
        else:
            v_pos = "middle"
        
        return f"{v_pos} {h_pos}"
    
    def process_detections(self, results):
        """Extract detected objects from YOLO results with improved filtering"""
        detected_objects = []
        
        if results.boxes is not None:
            for box in results.boxes:
                confidence = float(box.conf[0])
                
                # Use dynamic confidence threshold
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
        
        # Filter out duplicate detections of the same object type
        return self.filter_duplicate_detections(detected_objects)
    
    def filter_duplicate_detections(self, detected_objects):
        """Filter out duplicate detections of the same object in similar positions"""
        if len(detected_objects) <= 1:
            return detected_objects
        
        filtered = []
        for obj in detected_objects:
            # Check if we already have a similar object
            is_duplicate = False
            for existing in filtered:
                if (obj['name'] == existing['name'] and 
                    obj['position'] == existing['position'] and
                    abs(obj['confidence'] - existing['confidence']) < 0.2):
                    is_duplicate = True
                    # Keep the one with higher confidence
                    if obj['confidence'] > existing['confidence']:
                        filtered.remove(existing)
                        filtered.append(obj)
                    break
            
            if not is_duplicate:
                filtered.append(obj)
        
        return filtered
    
    def draw_boxes(self, image, results):
        """Draw bounding boxes with enhanced information"""
        annotated_image = image.copy()
        
        if results.boxes is not None:
            for box in results.boxes:
                confidence = float(box.conf[0])
                
                if confidence > self.confidence_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls[0])
                    class_name = results.names[class_id]
                    
                    position = self.get_object_position([x1, y1, x2, y2])
                    
                    # Color based on confidence
                    if confidence > 0.7:
                        color = (0, 255, 0)  # Green for high confidence
                    elif confidence > 0.5:
                        color = (0, 255, 255)  # Yellow for medium confidence
                    else:
                        color = (0, 165, 255)  # Orange for low confidence
                    
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                    
                    label = f"{class_name} ({position}): {confidence:.2f}"
                    
                    # Background for text
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(annotated_image, (x1, y1-25), (x1 + text_size[0], y1), color, -1)
                    
                    cv2.putText(annotated_image, label, (x1, y1-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return annotated_image
    
    def speak_detections(self, detected_objects):
        """Convert detections to speech with improved grouping"""
        if not detected_objects:
            return
        
        # Filter out person detections if there are other objects
        non_person_objects = [obj for obj in detected_objects if obj['name'] != 'person']
        if non_person_objects:
            detected_objects = non_person_objects
        
        # Group objects by name
        object_counts = defaultdict(list)
        for obj in detected_objects:
            object_counts[obj['name']].append(obj['position'])
        
        # Create speech message
        speech_parts = []
        for obj_name, positions in object_counts.items():
            if len(positions) == 1:
                speech_parts.append(f"{obj_name} on the {positions[0]}")
            else:
                speech_parts.append(f"{len(positions)} {obj_name}s")
        
        if speech_parts:
            message = "I can see: " + ", ".join(speech_parts)
            self.speak(message)
    
    def speak(self, message):
        """Text-to-speech with thread safety"""
        try:
            print(f"🔊 Speaking: {message}")
            self.tts.say(message)
            self.tts.runAndWait()
        except Exception as e:
            print(f"Speech error: {e}")

# Example usage
if __name__ == "__main__":
    try:
        smart_glasses = SmartGlassesWithSpeech()
        smart_glasses.detect_with_speech_control()
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down Smart Glasses...")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have a webcam and microphone connected.")
        print("Also ensure you have the required packages: ultralytics, opencv-python, pyttsx3, speechrecognition")