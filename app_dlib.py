# -*- coding: utf-8 -*-

import io
import time
import dlib
import numpy as np
import pygame
import pygame.camera
from PIL import Image, ImageFont, ImageDraw

from flask import Flask, render_template, Response, request, jsonify, g
import base64
import threading

app = Flask(__name__)

pygame.init()
pygame.camera.init()
pygame.mixer.init()

font = ImageFont.truetype("arial/arial.ttf", 30)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')

(L_START, L_END) = (42, 48)
(R_START, R_END) = (36, 42)
(MOUTH_START, MOUTH_END) = (48, 68)
(NOSE_POINT) = 30

alarm_playing = False

def reset_alarm(seconds=10):
    global alarm_playing
    time.sleep(seconds)  # delay before resetting the alarm
    alarm_playing = False

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

global camera_on
camera_on = False  # Changed to True to start capturing as soon as the Flask app starts

@app.route('/toggle_camera', methods=['POST'])
def toggle_camera():
    global camera_on
    camera_on = not camera_on
    # print("Current camera_on status:", camera_on)  # Debug line
    return jsonify(status=camera_on)

def gen_frames():
    
    global camera_on  # Use the global variable
    
    global alarm_playing
    
    # Initialize frame to some default image (Here, a 640x480 black image)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    frame_bytes = None
    
    cam = pygame.camera.Camera(pygame.camera.list_cameras()[0])
    cam.start()

    while True:
        if camera_on:  # Only process frames if camera_on is True
            cam.start()
            # Capture frame from camera
            img = cam.get_image()

            # Convert to NumPy array and BGR format
            frame = pygame.surfarray.array3d(img)
            frame = frame.transpose([1, 0, 2])
            # frame = frame[:, :, [2, 1, 0]]

            # Flip the frame horizontally
            frame = frame[:, ::-1, :]  # <-- This line flips the frame

            # Convert to grayscale for face detection
            gray = np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140])

            rects = detector(gray.astype('uint8'), 0)

            for rect in rects:

                x = rect.left()
                y = rect.top()
                w = rect.width()
                h = rect.height()

                # Determine face landmarks
                shape = predictor(gray.astype('uint8'), rect)
                shape = np.array([(shape.part(i).x, shape.part(i).y) for i in range(0, 68)])

                # Compute eye aspect ratios
                leftEAR = eye_aspect_ratio(shape[L_START:L_END])
                rightEAR = eye_aspect_ratio(shape[R_START:R_END])
                ear = (leftEAR + rightEAR) / 2.0

                # Create a PIL image for drawing text
                frame_pil = Image.fromarray(frame.astype('uint8'))
                draw = ImageDraw.Draw(frame_pil)

                if ear < 0.15:
                    draw.text((20, 30), " • Sleepy (EAR: {:.2f})".format(ear), font=font, fill=(255, 0, 0, 0))

                    if not alarm_playing:
                        # pygame.mixer.music.load('alarm.mp3') ### 10 segundos
                        pygame.mixer.music.load('alarma.mp3') ### 3 segundos
                        pygame.mixer.music.play()
                        alarm_playing = True
                        threading.Timer(5, reset_alarm).start()
                else:
                    draw.text((20, 30), " • Awake (EAR: {:.2f})".format(ear), font=font, fill=(0, 255, 0, 0))

                frame = np.array(frame_pil)
        
        else:
            cam.stop()
            
        # Convert frame to JPEG format for streaming
        if frame is not None:
            pil_img = Image.fromarray(frame.astype('uint8'))
            buffer = io.BytesIO()
            pil_img.save(buffer, format="JPEG")
            frame_bytes = buffer.getvalue()

        if frame_bytes is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/processed_feed')
def processed_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    pygame.camera.init()  # Initialize the camera before running the Flask app
    app.debug = True
    app.run(port=5002)