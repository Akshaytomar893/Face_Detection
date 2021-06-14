#importing required libraries
import cv2
import os

#Initializing the haarcascade classifier ob openCV module
cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascPath)

#capturing Video input through Cam
video_capture=cv2.VideoCapture(0)

#main loop
while True:

    #reading input viddeo frame by frame
    ret , frames=video_capture.read()

    #converting the frame into grayscale 
    gray=cv2.cvtColor(frames , cv2.COLOR_BGR2GRAY)

    #detecting face
    faces=face_cascade.detectMultiScale(gray , scaleFactor=1.1 , minNeighbors=5 , minSize=(30,30) , flags=cv2.CASCADE_SCALE_IMAGE)

    #drawing rectangle around the face
    for (x ,y ,w ,h) in faces:
        cv2.rectangle(frames, (x, y), (x+w, y+h), (0, 255, 0), 3)

    #Showing the image
    cv2.imshow('Video' , frames)

    #breaks the loop if 'esc' key is hit
    if cv2.waitKey(10) & 0xFF==27:
        break
#stoping the video capturing and destroying all the windows
video_capture.release()
cv2.destroyAllWindows()