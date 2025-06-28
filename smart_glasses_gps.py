"""
Smart Glasses GPS Module - Voice-Controlled Navigation
Integrates with the main smart glasses system for location-based services
"""

import pyttsx3
import speech_recognition as sr
import requests
import json
import math
import time
import threading
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import sqlite3
from datetime import datetime

class SmartGlassesGPS:
    def __init__(self):
        # Initialize text-to-speech
        self.tts = pyttsx3.init()
        self.tts.setProperty('rate', 150)
        self.tts.setProperty('volume', 0.9)
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Initialize geocoder
        self.geolocator = Nominatim(user_agent="smart_glasses_gps")
        
        # GPS and navigation state
        self.current_location = None
        self.destination = None
        self.route_directions = []
        self.current_direction_index = 0
        self.is_navigating = False
        
        # Voice control
        self.listening = False
        self.speech_thread = None
        
        # Database for saved locations
        self.init_database()
        
        # Navigation settings
        self.navigation_update_interval = 30  # seconds
        self.last_navigation_update = 0
        
        print("🗺️ Smart Glasses GPS Module initialized!")
        self.show_gps_commands()
    
    def init_database(self):
        """Initialize SQLite database for saved locations"""
        try:
            self.conn = sqlite3.connect('smart_glasses_locations.db', check_same_thread=False)
            cursor = self.conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS saved_locations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    address TEXT NOT NULL,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            self.conn.commit()
            print("📁 Location database initialized")
        except Exception as e:
            print(f"Database initialization error: {e}")
    
    def show_gps_commands(self):
        """Display available GPS voice commands"""
        print("\n GPS VOICE COMMANDS:")
        print("=" * 50)
        print(" LOCATION COMMANDS:")
        print("• 'where am i' - Get current location")
        print("• 'my coordinates' - Get GPS coordinates")
        print("• 'save location as [name]' - Save current location")
        
        print("\n NAVIGATION COMMANDS:")
        print("• 'navigate to [address]' - Start navigation")
        print("• 'go to [saved location]' - Navigate to saved place")
        print("• 'how to get to [place]' - Get directions")
        print("• 'next direction' - Repeat current direction")
        print("• 'skip direction' - Move to next direction")
        print("• 'stop navigation' - End navigation")
        
        print("\n INFORMATION COMMANDS:")
        print("• 'distance to [place]' - Get distance")
        print("• 'travel time to [place]' - Get estimated time")
        print("• 'nearby [type]' - Find nearby places (gas, food, etc.)")
        print("• 'weather here' - Get local weather")
        
        print("\n SAVED LOCATIONS:")
        print("• 'list saved locations' - Show all saved places")
        print("• 'delete location [name]' - Remove saved location")
        print("• 'rename location [old] to [new]' - Rename location")
        
        print("\n SYSTEM COMMANDS:")
        print("• 'gps help' - Repeat commands")
        print("• 'gps status' - Show GPS status")
        print("• 'exit gps' - Close GPS module")
        print("=" * 50)
    
    def start_gps_voice_control(self):
        """Start GPS voice recognition"""
        if self.listening:
            return
        
        self.listening = True
        self.speech_thread = threading.Thread(target=self._gps_speech_loop, daemon=True)
        self.speech_thread.start()
        
        # Calibrate microphone
        print(" Calibrating microphone for GPS commands...")
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            print(" GPS voice control started. Say 'gps help' for commands.")
        except Exception as e:
            print(f"Microphone calibration error: {e}")
    
    def stop_gps_voice_control(self):
        """Stop GPS voice recognition"""
        self.listening = False
        self.is_navigating = False
        if self.speech_thread:
            self.speech_thread.join(timeout=1)
        print(" GPS voice control stopped.")
    
    def _gps_speech_loop(self):
        """Continuous GPS speech recognition loop"""
        while self.listening:
            try:
                with self.microphone as source:
                    # Shorter timeout for GPS commands
                    audio = self.recognizer.listen(source, timeout=2, phrase_time_limit=5)
                
                command = self.recognizer.recognize_google(audio).lower()
                print(f" GPS Command: '{command}'")
                
                result = self.process_gps_command(command)
                if result == "exit":
                    break
                
            except sr.WaitTimeoutError:
                # Check for navigation updates during silence
                if self.is_navigating:
                    self.check_navigation_updates()
                continue
            except sr.UnknownValueError:
                continue
            except sr.RequestError as e:
                print(f"GPS speech recognition error: {e}")
                time.sleep(1)
            except Exception as e:
                print(f"GPS speech loop error: {e}")
                time.sleep(1)
    
    def process_gps_command(self, command):
        """Process GPS voice commands"""
        try:
            # Location commands
            if "where am i" in command:
                self.get_current_location_speech()
            
            elif "my coordinates" in command:
                self.get_coordinates_speech()
            
            elif "save location as" in command:
                location_name = command.split("save location as")[-1].strip()
                self.save_current_location(location_name)
            
            # Navigation commands
            elif "navigate to" in command:
                destination = command.split("navigate to")[-1].strip()
                self.start_navigation(destination)
            
            elif "go to" in command:
                saved_location = command.split("go to")[-1].strip()
                self.navigate_to_saved_location(saved_location)
            
            elif "how to get to" in command:
                destination = command.split("how to get to")[-1].strip()
                self.get_directions_only(destination)
            
            elif "next direction" in command:
                self.repeat_current_direction()
            
            elif "skip direction" in command:
                self.skip_to_next_direction()
            
            elif "stop navigation" in command:
                self.stop_navigation()
            
            # Information commands
            elif "distance to" in command:
                destination = command.split("distance to")[-1].strip()
                self.get_distance_to(destination)
            
            elif "travel time to" in command:
                destination = command.split("travel time to")[-1].strip()
                self.get_travel_time_to(destination)
            
            elif "nearby" in command:
                place_type = command.split("nearby")[-1].strip()
                self.find_nearby_places(place_type)
            
            elif "weather here" in command:
                self.get_local_weather()
            
            # Saved location management
            elif "list saved locations" in command:
                self.list_saved_locations()
            
            elif "delete location" in command:
                location_name = command.split("delete location")[-1].strip()
                self.delete_saved_location(location_name)
            
            elif "rename location" in command and " to " in command:
                parts = command.split("rename location")[-1].strip().split(" to ")
                if len(parts) == 2:
                    old_name, new_name = parts[0].strip(), parts[1].strip()
                    self.rename_saved_location(old_name, new_name)
            
            # System commands
            elif "gps help" in command:
                self.speak("GPS commands available")
                self.show_gps_commands()
            
            elif "gps status" in command:
                self.get_gps_status()
            
            elif "exit gps" in command:
                self.speak("Closing GPS module")
                return "exit"
            
            else:
                self.speak("GPS command not recognized. Say 'gps help' for available commands.")
        
        except Exception as e:
            print(f"GPS command processing error: {e}")
            self.speak("Error processing GPS command")
    
    def get_current_location_speech(self):
        """Get and speak current location"""
        try:
            # In a real implementation, you'd use actual GPS
            # For demo, we'll use IP-based location
            response = requests.get('http://ip-api.com/json/', timeout=5)
            data = response.json()
            
            if data['status'] == 'success':
                self.current_location = {
                    'latitude': data['lat'],
                    'longitude': data['lon'],
                    'city': data['city'],
                    'region': data['regionName'],
                    'country': data['country']
                }
                
                location_text = f"You are currently in {data['city']}, {data['regionName']}, {data['country']}"
                self.speak(location_text)
            else:
                self.speak("Unable to determine current location")
                
        except Exception as e:
            print(f"Location error: {e}")
            self.speak("Error getting current location")
    
    def get_coordinates_speech(self):
        """Get and speak GPS coordinates"""
        if self.current_location:
            lat = self.current_location['latitude']
            lon = self.current_location['longitude']
            coord_text = f"Your coordinates are {lat:.4f} latitude, {lon:.4f} longitude"
            self.speak(coord_text)
        else:
            self.speak("Current location not available. Say 'where am I' first.")
    
    def save_current_location(self, location_name):
        """Save current location with a name"""
        if not self.current_location:
            self.speak("Current location not available. Get your location first.")
            return
        
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO saved_locations 
                (name, address, latitude, longitude) 
                VALUES (?, ?, ?, ?)
            ''', (
                location_name,
                f"{self.current_location['city']}, {self.current_location['region']}",
                self.current_location['latitude'],
                self.current_location['longitude']
            ))
            self.conn.commit()
            self.speak(f"Location saved as {location_name}")
        except Exception as e:
            print(f"Save location error: {e}")
            self.speak("Error saving location")
    
    def start_navigation(self, destination):
        """Start navigation to destination"""
        try:
            self.speak(f"Finding route to {destination}")
            
            # Geocode destination
            location = self.geolocator.geocode(destination)
            if not location:
                self.speak(f"Could not find {destination}")
                return
            
            self.destination = {
                'name': destination,
                'latitude': location.latitude,
                'longitude': location.longitude,
                'address': location.address
            }
            
            # Get route (simplified - in reality you'd use a routing API)
            self.get_route_directions()
            
            if self.route_directions:
                self.is_navigating = True
                self.current_direction_index = 0
                self.speak(f"Navigation started to {destination}")
                self.speak_current_direction()
            else:
                self.speak("Could not get directions")
                
        except Exception as e:
            print(f"Navigation error: {e}")
            self.speak("Error starting navigation")
    
    def navigate_to_saved_location(self, saved_location):
        """Navigate to a saved location"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('SELECT * FROM saved_locations WHERE name LIKE ?', (f'%{saved_location}%',))
            location = cursor.fetchone()
            
            if location:
                _, name, address, lat, lon, _ = location
                self.start_navigation(address)
            else:
                self.speak(f"Saved location '{saved_location}' not found")
                
        except Exception as e:
            print(f"Saved location navigation error: {e}")
            self.speak("Error navigating to saved location")
    
    def get_route_directions(self):
        """Get route directions (simplified implementation)"""
        # In a real implementation, you'd use Google Maps API, MapBox, etc.
        # This is a simplified version for demonstration
        
        if not self.current_location or not self.destination:
            return
        
        # Calculate distance and bearing
        current_pos = (self.current_location['latitude'], self.current_location['longitude'])
        dest_pos = (self.destination['latitude'], self.destination['longitude'])
        
        distance = geodesic(current_pos, dest_pos).kilometers
        
        # Simplified directions
        self.route_directions = [
            f"Head towards {self.destination['name']}",
            f"Continue for {distance:.1f} kilometers",
            f"Arrive at {self.destination['name']}"
        ]
    
    def speak_current_direction(self):
        """Speak the current navigation direction"""
        if self.is_navigating and self.current_direction_index < len(self.route_directions):
            direction = self.route_directions[self.current_direction_index]
            self.speak(direction)
    
    def repeat_current_direction(self):
        """Repeat current direction"""
        if self.is_navigating:
            self.speak_current_direction()
        else:
            self.speak("No active navigation")
    
    def skip_to_next_direction(self):
        """Move to next direction"""
        if self.is_navigating and self.current_direction_index < len(self.route_directions) - 1:
            self.current_direction_index += 1
            self.speak_current_direction()
        elif self.is_navigating:
            self.speak("You have arrived at your destination")
            self.stop_navigation()
        else:
            self.speak("No active navigation")
    
    def stop_navigation(self):
        """Stop current navigation"""
        self.is_navigating = False
        self.destination = None
        self.route_directions = []
        self.current_direction_index = 0
        self.speak("Navigation stopped")
    
    def get_distance_to(self, destination):
        """Get distance to destination"""
        try:
            if not self.current_location:
                self.speak("Current location not available")
                return
            
            location = self.geolocator.geocode(destination)
            if not location:
                self.speak(f"Could not find {destination}")
                return
            
            current_pos = (self.current_location['latitude'], self.current_location['longitude'])
            dest_pos = (location.latitude, location.longitude)
            
            distance = geodesic(current_pos, dest_pos).kilometers
            
            if distance < 1:
                distance_text = f"Distance to {destination} is {distance*1000:.0f} meters"
            else:
                distance_text = f"Distance to {destination} is {distance:.1f} kilometers"
            
            self.speak(distance_text)
            
        except Exception as e:
            print(f"Distance calculation error: {e}")
            self.speak("Error calculating distance")
    
    def get_travel_time_to(self, destination):
        """Get estimated travel time"""
        try:
            # Simplified calculation (assuming average speed)
            if not self.current_location:
                self.speak("Current location not available")
                return
            
            location = self.geolocator.geocode(destination)
            if not location:
                self.speak(f"Could not find {destination}")
                return
            
            current_pos = (self.current_location['latitude'], self.current_location['longitude'])
            dest_pos = (location.latitude, location.longitude)
            
            distance_km = geodesic(current_pos, dest_pos).kilometers
            
            # Assume average speed of 50 km/h for driving
            travel_time_hours = distance_km / 50
            
            if travel_time_hours < 1:
                time_text = f"Estimated travel time to {destination} is {travel_time_hours*60:.0f} minutes"
            else:
                hours = int(travel_time_hours)
                minutes = int((travel_time_hours - hours) * 60)
                time_text = f"Estimated travel time to {destination} is {hours} hours and {minutes} minutes"
            
            self.speak(time_text)
            
        except Exception as e:
            print(f"Travel time calculation error: {e}")
            self.speak("Error calculating travel time")
    
    def find_nearby_places(self, place_type):
        """Find nearby places of specified type"""
        try:
            if not self.current_location:
                self.speak("Current location not available")
                return
            
            # This would integrate with Places API in real implementation
            self.speak(f"Searching for nearby {place_type}")
            # Simplified response
            self.speak(f"Found several {place_type} locations nearby. Would you like directions to the closest one?")
            
        except Exception as e:
            print(f"Nearby search error: {e}")
            self.speak("Error searching for nearby places")
    
    def get_local_weather(self):
        """Get local weather information"""
        try:
            if not self.current_location:
                self.speak("Current location not available")
                return
            
            # This would integrate with weather API in real implementation
            city = self.current_location.get('city', 'your location')
            self.speak(f"Getting weather information for {city}")
            # Simplified response
            self.speak("Weather information would be available with API integration")
            
        except Exception as e:
            print(f"Weather error: {e}")
            self.speak("Error getting weather information")
    
    def list_saved_locations(self):
        """List all saved locations"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('SELECT name, address FROM saved_locations ORDER BY name')
            locations = cursor.fetchall()
            
            if locations:
                self.speak(f"You have {len(locations)} saved locations:")
                for name, address in locations:
                    self.speak(f"{name}: {address}")
            else:
                self.speak("No saved locations found")
                
        except Exception as e:
            print(f"List locations error: {e}")
            self.speak("Error listing saved locations")
    
    def delete_saved_location(self, location_name):
        """Delete a saved location"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('DELETE FROM saved_locations WHERE name LIKE ?', (f'%{location_name}%',))
            
            if cursor.rowcount > 0:
                self.conn.commit()
                self.speak(f"Deleted location {location_name}")
            else:
                self.speak(f"Location {location_name} not found")
                
        except Exception as e:
            print(f"Delete location error: {e}")
            self.speak("Error deleting location")
    
    def rename_saved_location(self, old_name, new_name):
        """Rename a saved location"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('UPDATE saved_locations SET name = ? WHERE name LIKE ?', 
                         (new_name, f'%{old_name}%'))
            
            if cursor.rowcount > 0:
                self.conn.commit()
                self.speak(f"Renamed {old_name} to {new_name}")
            else:
                self.speak(f"Location {old_name} not found")
                
        except Exception as e:
            print(f"Rename location error: {e}")
            self.speak("Error renaming location")
    
    def get_gps_status(self):
        """Get GPS system status"""
        status_parts = []
        
        if self.current_location:
            status_parts.append(f"Location: {self.current_location['city']}")
        else:
            status_parts.append("Location: Unknown")
        
        if self.is_navigating:
            status_parts.append(f"Navigating to: {self.destination['name']}")
        else:
            status_parts.append("Navigation: Inactive")
        
        # Count saved locations
        try:
            cursor = self.conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM saved_locations')
            count = cursor.fetchone()[0]
            status_parts.append(f"Saved locations: {count}")
        except:
            status_parts.append("Saved locations: Unknown")
        
        status_text = "GPS Status: " + ", ".join(status_parts)
        self.speak(status_text)
    
    def check_navigation_updates(self):
        """Check if navigation updates are needed"""
        current_time = time.time()
        if (self.is_navigating and 
            current_time - self.last_navigation_update > self.navigation_update_interval):
            
            self.last_navigation_update = current_time
            self.speak("Continue following current direction")
    
    def speak(self, message):
        """Text-to-speech with thread safety"""
        try:
            print(f"🔊 GPS: {message}")
            self.tts.say(message)
            self.tts.runAndWait()
        except Exception as e:
            print(f"GPS speech error: {e}")
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_gps_voice_control()
        if hasattr(self, 'conn'):
            self.conn.close()
        print("🗺️ GPS module cleaned up")

# Integration function to add GPS to main smart glasses
def integrate_gps_with_smart_glasses(smart_glasses_instance):
    """
    Function to integrate GPS functionality with the main SmartGlassesWithSpeech class
    Call this function to add GPS capabilities to your existing smart glasses
    """
    gps_module = SmartGlassesGPS()
    
    # Add GPS reference to main instance
    smart_glasses_instance.gps = gps_module
    
    # Extend voice command processing
    original_process_command = smart_glasses_instance.process_voice_command
    
    def enhanced_process_command(command):
        # Check if it's a GPS command
        gps_keywords = ['navigate', 'where am i', 'location', 'directions', 'distance', 'gps']
        if any(keyword in command.lower() for keyword in gps_keywords):
            return gps_module.process_gps_command(command)
        else:
            return original_process_command(command)
    
    # Replace the method
    smart_glasses_instance.process_voice_command = enhanced_process_command
    
    # Start GPS voice control
    gps_module.start_gps_voice_control()
    
    print("✅ GPS functionality integrated with Smart Glasses!")
    return gps_module

# Standalone usage
if __name__ == "__main__":
    try:
        gps_system = SmartGlassesGPS()
        gps_system.start_gps_voice_control()
        
        print("\nGPS module running independently...")
        print("Press Ctrl+C to exit")
        
        # Keep running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n Shutting down GPS module...")
        if 'gps_system' in locals():
            gps_system.cleanup()
    except Exception as e:
        print(f"GPS Error: {e}")
        print("Make sure you have the required packages: geopy, requests, sqlite3")