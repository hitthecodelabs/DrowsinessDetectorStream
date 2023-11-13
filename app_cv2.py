# -*- coding: utf-8 -*-

import vlc
import time
import cv2, dlib
import numpy as np

import pygame
from playsound import playsound

from PIL import ImageFont, ImageDraw, Image
from flask import Flask, render_template, Response

import threading

app = Flask(__name__)

pygame.init()
pygame.mixer.init()

font = ImageFont.truetype("arial/arial.ttf", 30)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')

(L_START, L_END) = (42, 48)
(R_START, R_END) = (36, 42)

# Add additional constants
(MOUTH_START, MOUTH_END) = (48, 68)
(NOSE_POINT) = 30

alarm_playing = False

def reset_alarm():
    global alarm_playing
    time.sleep(10) # retraso antes de resetear la alarma
    alarm_playing = False

def detect_eyes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    return frame

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def gen_frames():  
    
    global alarm_playing
    
    # Abrir la cámara
    cap = cv2.VideoCapture(0)
    
    # Bucle infinito (hasta que se rompa)
    while True:
        # Leer un fotograma de la cámara
        success, frame = cap.read()
        frame = cv2.flip(frame, 1)
        
        # Si el fotograma no se lee correctamente, entonces rompemos el bucle
        if not success:
            break
        else:
            # Convertir la imagen a escala de grises
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detectar los rostros en la imagen en escala de grises
            rects = detector(gray, 0)
            
            # Para cada rostro detectado
            for rect in rects:
                
                x = rect.left()
                y = rect.top()
                w = rect.width()
                h = rect.height()

                # Draw rectangle on the frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
                
                # Determinar la forma del rostro usando un predictor
                shape = predictor(gray, rect)
                
                # Convertir la forma del rostro en una matriz numpy
                shape = np.array([(shape.part(i).x, shape.part(i).y) for i in range(0, 68)])
                
                # Extraer las coordenadas de los ojos izquierdo y derecho
                leftEye = shape[L_START:L_END]
                rightEye = shape[R_START:R_END]
                mouth = shape[MOUTH_START:MOUTH_END]
                
                # Calcular la relación de aspecto del ojo para ambos ojos
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                
                # Get nose point
                nose = shape[NOSE_POINT]
                
                # Draw points on eyes, mouth and nose with white color
                cv2.circle(frame, tuple(leftEye[0]), 2, (255,255,255), -1)
                cv2.circle(frame, tuple(leftEye[3]), 2, (255,255,255), -1)
                cv2.circle(frame, tuple(rightEye[0]), 2, (255,255,255), -1)
                cv2.circle(frame, tuple(rightEye[3]), 2, (255,255,255), -1)
                cv2.circle(frame, tuple(mouth[0]), 2, (255,255,255), -1)
                cv2.circle(frame, tuple(mouth[6]), 2, (255,255,255), -1)
                cv2.circle(frame, tuple(nose), 2, (255,255,255), -1)
                
                # Calcular la relación de aspecto promedio del ojo
                ear = (leftEAR + rightEAR) / 2.0
                
                # Convertir la imagen a PIL Image
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(frame_pil)

                # Si la relación de aspecto del ojo es menor que 0.3, significa que los ojos están cerrados
                # por lo tanto, la persona está adormilada. De lo contrario, la persona está despierta.
                if ear < 0.20: ### 0.25 by default
                    draw.text((20, 30), " • Con sueño", font=font, fill=(255,0,0,0))
                    
                    if not alarm_playing:
                    
                        pygame.mixer.music.load('alarma.mp3')
                        pygame.mixer.music.play()

                        alarm_playing = True
                        threading.Timer(10, reset_alarm).start()  # resetear la alarma después de 10 segundos
                    
                else:
                    draw.text((20, 30), " • Despierto", font=font, fill=(0,255,0,0))
                
                # Convertir la imagen de PIL Image de nuevo a OpenCV
                frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
            
            # Codificar la imagen para su transmisión
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            # Generar un fotograma para la transmisión
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=5001)
    # app.run(host='0.0.0.0', port=5001, debug=True)
    app.debug = True
    app.run()
