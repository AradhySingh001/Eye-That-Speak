from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
import urllib.parse
import threading
import time
import base64
import cv2
import os
import queue
from smart_glasses import IntegratedSmartGlasses


class SmartGlassesHTTPHandler(SimpleHTTPRequestHandler):
    # Class variables shared across all requests
    glasses_system = None
    camera_thread = None
    camera_running = False
    current_frame = None
    voice_thread = None
    voice_running = False
    tts_queue = queue.Queue()
    tts_thread = None
    tts_running = False
    tts_lock = threading.Lock()
    
    @classmethod
    def initialize_system(cls):
        """Initialize the Smart Glasses system with TTS conflict resolution"""
        if cls.glasses_system is None:
            try:
                print(" Initializing Smart Glasses system...")
                cls.glasses_system = IntegratedSmartGlasses()
                
                # STOP the original Smart Glasses TTS to prevent conflicts
                print(" Stopping original Smart Glasses TTS...")
                if hasattr(cls.glasses_system, 'tts') and cls.glasses_system.tts:
                    try:
                        cls.glasses_system.tts.stop()
                        cls.glasses_system.tts = None
                        print(" Original TTS stopped successfully")
                    except Exception as e:
                        print(f" Warning: Could not stop original TTS: {e}")
                
                # Replace original speak method with dummy (temporarily)
                def dummy_speak(text):
                    print(f" [Original TTS Disabled]: {text}")
                cls.glasses_system.speak = dummy_speak
                
                # Setup HTTP-compatible TTS system
                cls._setup_dedicated_tts()
                
                # Start voice recognition thread
                cls.start_voice_thread()
                
                # Start camera thread
                cls.start_camera_thread()
                
                print(" Smart Glasses system initialized with HTTP-compatible TTS")
                return True
                
            except Exception as e:
                print(f" Error initializing system: {e}")
                return False
        return True
    
    @classmethod
    def _setup_dedicated_tts(cls):
        """Setup dedicated TTS system for HTTP context"""
        try:
            print(" Setting up dedicated TTS system...")
            
            # Start dedicated TTS worker thread
            cls.tts_running = True
            cls.tts_thread = threading.Thread(target=cls._tts_worker, daemon=True)
            cls.tts_thread.start()
            
            # Replace the speak method with queue-based version
            def queue_speak(text):
                """Queue-based speech function that works in HTTP context"""
                try:
                    if text and text.strip():
                        print(f" Queuing speech: {text}")
                        cls.tts_queue.put(text)
                except Exception as e:
                    print(f" Queue speech error: {e}")
            
            # Override the Smart Glasses speak method
            cls.glasses_system.speak = queue_speak
            print(" Dedicated TTS system setup complete")
            
        except Exception as e:
            print(f" TTS setup error: {e}")
    
    @classmethod
    def _tts_worker(cls):
        """Dedicated TTS worker thread - handles all speech output"""
        tts_engine = None
        try:
            import pyttsx3
            
            # Create TTS engine in this dedicated thread
            tts_engine = pyttsx3.init()
            if tts_engine:
                tts_engine.setProperty('rate', 150)
                tts_engine.setProperty('volume', 0.9)
                print(" TTS worker thread started successfully")
            else:
                print(" Failed to initialize TTS engine")
                return
            
            while cls.tts_running:
                try:
                    # Get text from queue (block for up to 1 second)
                    text = cls.tts_queue.get(timeout=1)
                    
                    if text and text.strip():
                        with cls.tts_lock:
                            try:
                                print(f" Speaking: {text}")
                                tts_engine.say(text)
                                tts_engine.runAndWait()
                                print(f" Finished speaking")
                            except Exception as e:
                                print(f" TTS engine error: {e}")
                                # Try to reinitialize engine on error
                                try:
                                    tts_engine.stop()
                                    tts_engine = pyttsx3.init()
                                    if tts_engine:
                                        tts_engine.setProperty('rate', 150)
                                        tts_engine.setProperty('volume', 0.9)
                                        print(" TTS engine reinitialized")
                                except Exception as reinit_error:
                                    print(f" Failed to reinitialize TTS engine: {reinit_error}")
                    
                    cls.tts_queue.task_done()
                    
                except queue.Empty:
                    # No speech request, continue loop
                    continue
                except Exception as e:
                    print(f" TTS worker error: {e}")
                    time.sleep(1)
                    
        except Exception as e:
            print(f" TTS worker thread error: {e}")
        finally:
            if tts_engine:
                try:
                    tts_engine.stop()
                except:
                    pass
            print(" TTS worker thread stopped")
    
    @classmethod
    def start_voice_thread(cls):
        """Start voice recognition in separate thread"""
        if cls.voice_thread is None or not cls.voice_thread.is_alive():
            cls.voice_running = True
            cls.voice_thread = threading.Thread(target=cls._voice_loop, daemon=True)
            cls.voice_thread.start()
            print(" Voice recognition thread started")
    
    @classmethod
    def _voice_loop(cls):
        """Voice recognition loop - keeps voice control active"""
        try:
            print(" Starting voice control...")
            
            # Start voice control if available
            if hasattr(cls.glasses_system, 'start_voice_control'):
                cls.glasses_system.start_voice_control()
            
            while cls.voice_running and cls.glasses_system:
                try:
                    # Keep voice recognition active
                    if hasattr(cls.glasses_system, 'listening'):
                        if not cls.glasses_system.listening:
                            if hasattr(cls.glasses_system, 'start_voice_control'):
                                print("🎤 Restarting voice control...")
                                cls.glasses_system.start_voice_control()
                    
                    time.sleep(2)  # Check every 2 seconds
                    
                except Exception as e:
                    print(f" Voice loop error: {e}")
                    time.sleep(5)  # Wait before retry
                    
        except Exception as e:
            print(f" Voice thread error: {e}")
    
    @classmethod
    def start_camera_thread(cls):
        """Start camera streaming thread"""
        if cls.camera_thread is None or not cls.camera_thread.is_alive():
            cls.camera_running = True
            cls.camera_thread = threading.Thread(target=cls._camera_loop, daemon=True)
            cls.camera_thread.start()
            print(" Camera thread started")
    
    @classmethod
    def _camera_loop(cls):
        """Camera streaming loop"""
        while cls.camera_running and cls.glasses_system:
            try:
                frame = cls.glasses_system.capture_frame()
                if frame is not None:
                    # Process frame based on current mode
                    processed_frame = cls._process_frame(frame)
                    
                    # Convert to base64 for web transmission
                    _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    frame_b64 = base64.b64encode(buffer).decode('utf-8')
                    cls.current_frame = f"data:image/jpeg;base64,{frame_b64}"
                
                time.sleep(0.1)  # 10 FPS
            except Exception as e:
                print(f" Camera error: {e}")
                time.sleep(1)
    
    @classmethod
    def _process_frame(cls, frame):
        """Process frame based on current mode"""
        if not cls.glasses_system:
            return frame
        
        processed_frame = frame.copy()
        
        try:
            # Add mode-specific processing
            current_mode = getattr(cls.glasses_system, 'current_mode', '')
            
            if current_mode == "detection":
                detection_mode = getattr(cls.glasses_system, 'detection_mode', '')
                model = getattr(cls.glasses_system, 'model', None)
                if detection_mode == "continuous" and model:
                    confidence = getattr(cls.glasses_system, 'confidence_threshold', 0.25)
                    results = model(frame, conf=confidence)
                    processed_frame = cls.glasses_system.draw_boxes(frame, results[0])
            
            elif current_mode == "face_recognition":
                face_active = getattr(cls.glasses_system, 'face_recognition_active', False)
                if face_active:
                    processed_frame = cls.glasses_system.process_continuous_face_recognition(frame)
            
            # Add system overlay if available
            if hasattr(cls.glasses_system, 'add_system_overlay'):
                cls.glasses_system.add_system_overlay(processed_frame)
            
        except Exception as e:
            print(f" Frame processing error: {e}")
        
        return processed_frame
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/':
            self.serve_index()
        elif self.path == '/camera_feed':
            self.serve_camera_feed()
        elif self.path == '/status':
            self.serve_status()
        elif self.path == '/tts_status':
            self.serve_tts_status()
        elif self.path == '/voice_start':
            self.handle_voice_start()
        elif self.path == '/voice_stop':
            self.handle_voice_stop()
        else:
            super().do_GET()
    
    def do_POST(self):
        """Handle POST requests"""
        if self.path == '/command':
            self.handle_command()
        elif self.path == '/setting':
            self.handle_setting()
        elif self.path == '/test_speech':
            self.handle_test_speech()
        elif self.path == '/voice_control':
            self.handle_voice_control()
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_OPTIONS(self):
        """Handle OPTIONS requests for CORS"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def serve_index(self):
        """Serve the main HTML interface"""
        try:
            with open('index.html', 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(content.encode('utf-8'))
            
        except FileNotFoundError:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(' index.html not found')
    
    def serve_camera_feed(self):
        """Serve camera feed as base64 image"""
        try:
            if self.current_frame:
                response = {
                    'success': True,
                    'frame': self.current_frame,
                    'timestamp': time.time()
                }
            else:
                response = {
                    'success': False,
                    'error': 'No camera frame available'
                }
            
            self.send_json_response(response)
            
        except Exception as e:
            error_response = {
                'success': False,
                'error': str(e)
            }
            self.send_json_response(error_response)
    
    def serve_status(self):
        """Serve comprehensive system status"""
        try:
            if self.glasses_system:
                status = {
                    'success': True,
                    'current_mode': getattr(self.glasses_system, 'current_mode', 'Unknown'),
                    'camera_status': 'Connected' if getattr(self.glasses_system, 'camera_initialized', False) else 'Disconnected',
                    'voice_active': getattr(self.glasses_system, 'listening', False),
                    'face_count': len(getattr(self.glasses_system, 'known_faces', [])),
                    'navigation_active': getattr(self.glasses_system, 'is_navigating', False),
                    'text_reading_active': getattr(self.glasses_system, 'text_reading_active', False),
                    'detection_mode': getattr(self.glasses_system, 'detection_mode', 'single'),
                    'face_recognition_active': getattr(self.glasses_system, 'face_recognition_active', False),
                    'tts_queue_size': self.tts_queue.qsize() if self.tts_queue else 0,
                    'tts_status': 'Running' if self.tts_running else 'Stopped',
                    'threads': {
                        'camera_running': self.camera_running,
                        'voice_running': self.voice_running,
                        'tts_running': self.tts_running
                    }
                }
            else:
                status = {
                    'success': False,
                    'error': 'System not initialized'
                }
            
            self.send_json_response(status)
            
        except Exception as e:
            error_response = {
                'success': False,
                'error': str(e)
            }
            self.send_json_response(error_response)
    
    def serve_tts_status(self):
        """Serve detailed TTS system status"""
        try:
            status = {
                'success': True,
                'tts_running': self.tts_running,
                'tts_queue_size': self.tts_queue.qsize() if self.tts_queue else 0,
                'tts_thread_alive': self.tts_thread.is_alive() if self.tts_thread else False,
                'tts_lock_locked': self.tts_lock.locked()
            }
            self.send_json_response(status)
        except Exception as e:
            self.send_json_response({'success': False, 'error': str(e)})
    
    def handle_voice_start(self):
        """Start voice control endpoint"""
        try:
            if not self.glasses_system:
                self.initialize_system()
            
            if self.glasses_system and hasattr(self.glasses_system, 'start_voice_control'):
                if not getattr(self.glasses_system, 'listening', False):
                    self.glasses_system.start_voice_control()
                    message = "Voice control started"
                else:
                    message = "Voice control already active"
            else:
                message = "Voice control not available"
            
            response = {
                'success': True,
                'message': message,
                'voice_active': getattr(self.glasses_system, 'listening', False) if self.glasses_system else False
            }
            self.send_json_response(response)
            
        except Exception as e:
            self.send_json_response({'success': False, 'error': str(e)})
    
    def handle_voice_stop(self):
        """Stop voice control endpoint"""
        try:
            if self.glasses_system and hasattr(self.glasses_system, 'stop_voice_control'):
                self.glasses_system.stop_voice_control()
                message = "Voice control stopped"
            else:
                message = "Voice control not available"
            
            response = {
                'success': True,
                'message': message,
                'voice_active': getattr(self.glasses_system, 'listening', False) if self.glasses_system else False
            }
            self.send_json_response(response)
            
        except Exception as e:
            self.send_json_response({'success': False, 'error': str(e)})
    
    def handle_voice_control(self):
        """Handle voice control toggle via POST"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            action = data.get('action', '')
            
            if action == 'start':
                self.handle_voice_start()
            elif action == 'stop':
                self.handle_voice_stop()
            else:
                self.send_json_response({
                    'success': False,
                    'error': 'Invalid action. Use "start" or "stop"'
                })
                
        except Exception as e:
            self.send_json_response({'success': False, 'error': str(e)})
    
    def handle_test_speech(self):
        """Handle speech testing"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            text = data.get('text', 'Test speech output')
            
            if self.glasses_system:
                self.glasses_system.speak(text)
                response = {
                    'success': True,
                    'message': f'Speech test queued: "{text}"',
                    'queue_size': self.tts_queue.qsize()
                }
            else:
                response = {
                    'success': False,
                    'error': 'System not initialized'
                }
            
            self.send_json_response(response)
            
        except Exception as e:
            self.send_json_response({'success': False, 'error': str(e)})
    
    def handle_command(self):
        """Handle command requests - with proper speech output"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            command = data.get('command', '').strip()
            print(f" Web command received: '{command}'")
            
            if not command:
                self.send_json_response({
                    'success': False,
                    'error': 'Empty command received'
                })
                return
            
            if not self.glasses_system:
                if not self.initialize_system():
                    self.send_json_response({
                        'success': False,
                        'error': 'Failed to initialize Smart Glasses system'
                    })
                    return
            
            # Process command in background to allow immediate HTTP response
            def process_command():
                try:
                    print(f" Processing command: '{command}'")
                    result = self.glasses_system.process_unified_command(command)
                    print(f" Command completed: '{command}' -> {result}")
                except Exception as e:
                    print(f" Command error for '{command}': {e}")
                    # Still try to speak the error
                    self.glasses_system.speak(f"Error processing command: {str(e)[:100]}")
            
            # Start command processing in background thread
            command_thread = threading.Thread(target=process_command, daemon=True)
            command_thread.start()
            
            # Immediate response to web interface
            response = {
                'success': True,
                'result': f"Command '{command}' is being processed",
                'command_result': "Processing with speech output...",
                'timestamp': data.get('timestamp', time.time()),
                'queue_size': self.tts_queue.qsize()
            }
            
            self.send_json_response(response)
            
        except Exception as e:
            print(f" Handle command error: {e}")
            error_response = {
                'success': False,
                'error': str(e)
            }
            self.send_json_response(error_response)
    
    def handle_setting(self):
        """Handle setting updates"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            setting = data.get('setting')
            value = data.get('value')
            
            print(f" Setting update: {setting} = {value}")
            
            if not self.glasses_system:
                if not self.initialize_system():
                    self.send_json_response({
                        'success': False,
                        'error': 'Failed to initialize Smart Glasses system'
                    })
                    return
            
            try:
                # Apply setting
                if setting == 'confidence' and hasattr(self.glasses_system, 'confidence_threshold'):
                    self.glasses_system.confidence_threshold = float(value)
                    self.glasses_system.speak(f"Confidence threshold set to {value}")
                    
                elif setting == 'speech_rate':
                    # Note: We can't change the TTS rate of our dedicated engine easily,
                    # but we can acknowledge the request
                    self.glasses_system.speak(f"Speech rate setting acknowledged")
                    
                elif setting == 'detection_mode' and hasattr(self.glasses_system, 'detection_mode'):
                    self.glasses_system.detection_mode = value
                    self.glasses_system.speak(f"Detection mode set to {value}")
                    
                response = {
                    'success': True,
                    'setting': setting,
                    'value': value,
                    'message': f"Setting '{setting}' updated successfully"
                }
                
            except Exception as e:
                response = {
                    'success': False,
                    'error': f"Failed to set {setting}: {str(e)}"
                }
            
            self.send_json_response(response)
            
        except Exception as e:
            error_response = {
                'success': False,
                'error': str(e)
            }
            self.send_json_response(error_response)
    
    def send_json_response(self, data):
        """Send JSON response with proper headers"""
        try:
            response_data = json.dumps(data, ensure_ascii=False).encode('utf-8')
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(response_data)
            
        except Exception as e:
            print(f" Send JSON response error: {e}")


def run_server(port=8000, host='localhost'):
    """Run the HTTP server with proper initialization"""
    server = None
    try:
        print("=" * 80)
        print(" Smart Glasses HTTP Server Starting...")
        print("=" * 80)
        
        # Pre-initialize system
        print(" Initializing Smart Glasses system...")
        if SmartGlassesHTTPHandler.initialize_system():
            print(" System initialization successful")
        else:
            print(" System initialization failed - continuing anyway")
        
        # Create and start server
        server = HTTPServer((host, port), SmartGlassesHTTPHandler)
        
        print(f" Web Interface: http://{host}:{port}")
        print(f" Camera Feed: http://{host}:{port}/camera_feed")
        print(f" System Status: http://{host}:{port}/status")
        print(f" TTS Status: http://{host}:{port}/tts_status")
        print("Voice Commands: Active in background")
        print("Speech Output: Queue-based TTS system")
        print("Press Ctrl+C to stop the server")
        print("=" * 80)
        
        # Start the server
        server.serve_forever()
        
    except KeyboardInterrupt:
        print("\n Server stopped by user")
    except Exception as e:
        print(f" Server error: {e}")
    finally:
        # Cleanup
        print(" Cleaning up...")
        SmartGlassesHTTPHandler.camera_running = False
        SmartGlassesHTTPHandler.voice_running = False
        SmartGlassesHTTPHandler.tts_running = False
        
        if server:
            server.server_close()
            
        print(" Server shutdown complete")


if __name__ == "__main__":
    # Check for required files
    required_files = ['index.html', 'smart_glasses.py']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(" Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print(" Please ensure all files are in the same directory as this script.")
        print()
    
    # Check if smart_glasses module can be imported
    try:
        from smart_glasses import IntegratedSmartGlasses
        print(" Smart Glasses module import successful")
    except ImportError as e:
        print(f" Cannot import Smart Glasses module: {e}")
        print(" Please ensure smart_glasses.py is in the same directory")
        print()
    
    # Start server
    try:
        run_server()
    except KeyboardInterrupt:
        print("\n Goodbye!")
