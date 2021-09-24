import cv2

cam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
back_sub = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cam.read()  

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    
    #for (x, y, w, h) in faces:
        #frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('', back_sub.apply(frame))

    key = cv2.waitKey(1)

    if key == 27:
        break

cv2.destroyAllWindows()
