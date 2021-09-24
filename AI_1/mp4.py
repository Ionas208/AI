import cv2
import numpy as np
import os

video = cv2.VideoCapture('wide.mp4')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

try:
    if not os.path.exists('faces'):
        os.makedirs('faces')
except OSError:
    print('Permission denied')

face_counter = 0
while True:
    ret, frame = video.read()
    if ret:
        

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
        for (x, y, w, h) in faces:
            frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face = frame[y:y+h, x:x+w]
            face_name = './faces/face' + str(face_counter) + ".jpg"
            face_counter += 1
            print(f'creating {face_name}')
            cv2.imwrite(face_name, face)

        
    else:
        break