import pyttsx3
import pyaudio
import wave
import os
import time
import numpy as np

class AudioSystem:
    def __init__(self, robot=None):
        self.robot = robot
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        # Only initialize PyAudio if we are in Mock mode or robot not provided
        if not self.robot or not self.robot.is_connected():
            self.p = pyaudio.PyAudio()
        else:
            self.p = None
            
        # TTS Engine (still used for generation, but playback might differ)
        self.engine = pyttsx3.init()
        # Set properties (optional)
        self.engine.setProperty('rate', 150)

    def speak(self, text):
        """Synthesizes speech from text."""
        print(f"Robot says: {text}")
        if self.robot and self.robot.is_connected():
            # Issue: pyttsx3 generates audio to speakers or file.
            # We need to save to file then play via SDK.
            filename = os.path.abspath("speech_output.wav")
            # delete if exists to be sure?
            if os.path.exists(filename):
                try:
                    os.remove(filename)
                except:
                    pass
            
            self.engine.save_to_file(text, filename)
            self.engine.runAndWait() # This waits for the file to be written
            
            if os.path.exists(filename):
                print(f"Audio saved to {filename}")
            else:
                print(f"Error: Audio file not saved at {filename}")

            # Play via SDK
            # Assuming robot wrapper has access to mini
            if hasattr(self.robot.mini, 'play_sound'):
                 self.robot.mini.play_sound(filename)
            elif hasattr(self.robot.mini, 'media_manager'):
                 self.robot.mini.media_manager.play_sound(filename)
        else:
            # Local playback (Mock)
            self.engine.say(text)
            self.engine.runAndWait()

    def listen(self, duration=4):
        """
        Records audio for a fixed duration using Reachy SDK or PyAudio.
        """
        print("Listening...")
        filename = "user_input.wav"
        
        if self.robot and self.robot.is_connected():
            # Use SDK
            # Expected behavior:
            # 1. start_recording()
            # 2. get_audio_sample() returns chunks (float32, ~256 samples)
            # 3. stop_recording()
            
            audio_chunks = []
            
            if hasattr(self.robot.mini, 'media_manager'):
                 try:
                     self.robot.mini.media_manager.start_recording()
                     
                     start_time = time.time()
                     while (time.time() - start_time) < duration:
                         chunk = self.robot.mini.media_manager.get_audio_sample()
                         if chunk is not None:
                             audio_chunks.append(chunk)
                         else:
                             # Wait a tiny bit to avoid busy loop if no data
                             # But not too long to avoid overflow/missed chunks
                             time.sleep(0.002) 
                             
                     self.robot.mini.media_manager.stop_recording()
                     
                     if audio_chunks:
                         # Concatenate and convert to int16
                         full_data = np.concatenate(audio_chunks)
                         # Check conversion need
                         if full_data.dtype == np.float32 or full_data.dtype == np.float64:
                              # Clamp and convert
                              # Assuming range -1.0 to 1.0
                              full_data = np.clip(full_data, -1.0, 1.0)
                              full_data = (full_data * 32767).astype(np.int16)
                         
                         # Save
                         with wave.open(filename, 'wb') as wf:
                             wf.setnchannels(1)
                             wf.setsampwidth(2) # 16-bit
                             wf.setframerate(self.RATE)
                             wf.writeframes(full_data.tobytes())
                         
                         print(f"Finished recording. Captured {len(audio_chunks)} chunks.")
                     else:
                         print("Warning: No audio data captured via SDK.")
                         
                 except Exception as e:
                     print(f"SDK Listen Error: {e}")
            else:
                 print("Error: Robot connected but no media_manager found.")
                 
        else:
            # PyAudio Mock/Local
            if self.p:
                stream = self.p.open(format=self.FORMAT,
                                     channels=self.CHANNELS,
                                     rate=self.RATE,
                                     input=True,
                                     frames_per_buffer=self.CHUNK)
                
                frames = []
                # Simple time based loop
                start_time = time.time()
                while (time.time() - start_time) < duration:
                    try:
                        data = stream.read(self.CHUNK, exception_on_overflow=False)
                        frames.append(data)
                    except Exception as e:
                        print(f"DEBUG: Audio read error: {e}")
                        break
                
                stream.stop_stream()
                stream.close()
                
                with wave.open(filename, 'wb') as wf:
                    wf.setnchannels(self.CHANNELS)
                    wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
                    wf.setframerate(self.RATE)
                    wf.writeframes(b''.join(frames))

        print("Finished recording.")
        return filename

    def cleanup(self):
        if self.p:
            self.p.terminate()

if __name__ == "__main__":
    audio = AudioSystem()
    audio.speak("System initialized.")
    # audio.listen(3)
