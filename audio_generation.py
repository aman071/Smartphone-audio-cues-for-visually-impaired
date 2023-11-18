from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
from screeninfo import get_monitors
import numpy as np
import cv2
import json
import tempfile
import os
import time

def get_screen_resolution(use_default):
    if use_default:
        return 640, 480  # Default resolution

    monitors = get_monitors()
    if monitors:
        return monitors[0].width, monitors[0].height
    else:
        print("Unable to determine screen resolution.")
        return None

def text_to_speech(text, lang='en'):
    tts = gTTS(text=text, lang=lang, slow=False)
    with tempfile.NamedTemporaryFile(delete=False) as temp_audio:
        tts.save(temp_audio.name)
        audio_segment = AudioSegment.from_file(temp_audio.name)
        os.remove(temp_audio.name)
    return audio_segment

input_file = 'results.json'
with open(input_file, 'r') as f:
    results_data = json.load(f)

use_default_resolution = 1  # 0 to use device's resolution
screen_width, screen_height = get_screen_resolution(use_default_resolution)

# Total duration of audio (in milliseconds)
total_duration = 10000  # 10sec

# Silent audio segment
audio = AudioSegment.silent(duration=total_duration)

for frame_idx, frame_results in enumerate(results_data):
    frame_duration = total_duration / len(results_data)
    frame_audio = AudioSegment.silent(duration=frame_duration)

    for obj_result in frame_results:
        bounding_box = obj_result['bounding_box']
        class_name = obj_result['class_name']

        # Adjusting amplitude based on distance between y-coordinates
        distance_y = bounding_box['y'] + bounding_box['height'] - bounding_box['y']
        amplitude = np.clip(1 - distance_y / screen_height, 0.1, 1)  # Amplitude range = [0.1, 1]

        # Midpoint of bounding box for panning audio
        midpoint_x = bounding_box['x'] + bounding_box['width'] / 2
        midpoint_y = bounding_box['y'] + bounding_box['height'] / 2

        # Distance between midpoints and pan accordingly
        distance_to_center = midpoint_x - screen_width / 2
        pan = np.clip(distance_to_center / (screen_width / 2), -1, 1)  # Pan range = [-1, 1]

        voice_audio = text_to_speech(class_name)
        voice_audio = voice_audio - (1 - amplitude) * 50
        voice_audio = voice_audio.pan(pan)
        frame_audio = frame_audio.overlay(voice_audio)

        # print(f"Frame {frame_idx + 1}, Object: {class_name}, Pan: {pan}, Amplitude: {amplitude:.2f}")

    audio = audio.overlay(frame_audio, position=int(frame_idx * frame_duration))
    if (frame_idx + 1) % 3 == 0:    # Speak out the class name every 3 seconds
        text_to_speech("Attention, " + class_name).play()

# Export the stereo audio to a file
audio.export('stereo_audio_with_speech_amplitude.wav', format='wav')