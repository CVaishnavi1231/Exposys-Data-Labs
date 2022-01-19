#  face detection in video and image

import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap1 = cv2.VideoCapture(0)

# RGB format ( red green blue )  Gray scale


while (True):
    ret, img1 = cap1.read()

    # convert image
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    # to detect the face
    faces = face_cascade.detectMultiScale(gray1, 1.1, 4)

    # draw the rectangle
    # detect multi scale (x,y,w,h)

    for (x, y, w, h) in faces:
        cv2.rectangle(img1, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow('img', img1)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    cap1.release()