
# Importing packages

import cv2 # open cv package
from playsound import playsound

fire_cascade=cv2.CascadeClassifier("fire_detection.xml")

cap=cv2.VideoCapture(0)

while(True):
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    fire=fire_cascade.detectMultiScale(frame,1.2,5)
    # four values
    # x , y , w , h
    for (x,y,w,h) in fire:
        cv2.rectangle(frame,(x-20,y-20),(x+w+20,y+h+20),(255,0,0),2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=frame[y:y+h,x:x+w]
        print("fire detected")
        playsound("audio.mp3")
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break

