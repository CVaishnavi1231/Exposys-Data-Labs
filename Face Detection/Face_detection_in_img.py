import cv2

face_cascade= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

img=cv2.imread("F:/Certificates and photos/facebookphotos/photo1.jpg")

# RGB format ( red green blue )  Gray scale 
 
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
# to detect the face
faces= face_cascade.detectMultiScale(gray,1.1,4)
    
# draw the rectangle
# detect multi scale (x,y,w,h)
    
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
cv2.imshow('img',img)
cv2.waitKey()
