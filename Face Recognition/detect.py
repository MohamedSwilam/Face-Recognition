import cv2
import cv2.face
import os
import numpy as np


subjects = ["","User1","User2","User3","User4","User5","User6","User7","User8","User9","User10","User11","User12","User13","User14","User15","User16","User17","User18","User19","User20","User21","User22","User23","User24","User25","User26","Swilam","Mostafa Hammad","Mohamed Ahmed"]

def detect_video_faces():
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read("recognizer/train.yml")
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')#load xml file
    face_scaleFactor = 1.3
    cap = cv2.VideoCapture(0)
    while 1:
        ret, img = cap.read() #read images
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = face_cascade.detectMultiScale(gray, face_scaleFactor, 5) #detect features in face

        for (x, y, w, h) in face:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
            myFace = gray[y:y + h, x:x + w]

            label, confidence = face_recognizer.predict(myFace) #compare my face with trainer

            label_text = subjects[label]
            label_text = label_text + " " + str(int(confidence)) + "%"

            #if confidence > 75:
             #   label_text = subjects[label]
              #  label_text = label_text + " " + str(int(confidence)) + "%"
           # else:
            #    label_text = str(int(confidence)) + "%"

            cv2.putText(img, label_text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 0), 2)

        cv2.imshow('img', img) #show frame

        k = cv2.waitKey(100) & 0xff
        if k == 27:
            break
    cap.release()


print("Predicting images...")

detect_video_faces()

print("Prediction complete")