import cv2
import cv2.face
import os
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')#read haarcascade file


def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors = 5);#detect features from file
    #print(faces)
    if (len(faces) == 0):
        return None, None

    (x, y, w, h) = faces[0] #Dimesions of face
    #print(gray[y:y + w, x:x + h])
    return gray[y:y + w, x:x + h], faces[0]

def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path) #Get all Folders in the directory

    faces = []
    labels = []

    for dir_name in dirs:
        if not dir_name.startswith("User"):
            continue;

        label = int(dir_name.replace("User", "")) #remove user from string and convert to int

        subject_dir_path = data_folder_path + "/" + dir_name # = training-data/User1

        subject_images_names = os.listdir(subject_dir_path) #Get all images in Folder User1

        for image_name in subject_images_names:

            image_path = subject_dir_path + "/" + image_name# = training-data/User1/image_0001.jpg

            image = cv2.imread(image_path) #read image

            cv2.imshow("Training on image...", cv2.resize(image, (400, 500))) #show image
            cv2.waitKey(100)

            face, rect = detect_face(image)

            if face is not None:
                faces.append(face)
                labels.append(label)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces, labels


ctr = 1
dirs = os.listdir("training-data") #Get all Folders in the directory
for dir_name in dirs:
    ctr += 1

name = input("please enter the name of the person")

newDir = "./training-data/User"+str(ctr)
try:
    if not os.path.exists(newDir):
        os.makedirs(newDir)
        newFolderName = "User" + str(ctr)

        cam = cv2.VideoCapture(0)
        counter = 0
        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.4, 5)
            for (x, y, w, h) in faces:
                counter += 1
                imgPath = "training-data/" + newFolderName + "/" + str(counter) + ".jpg"
                cv2.imwrite(imgPath, gray[y:y + h, x:x + w])
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.waitKey(100)

                cv2.imshow("Person face", img)
            cv2.waitKey(1)
            if counter > 50:
                break
        cam.release()
        cv2.destroyAllWindows()

        print("Preparing data...")
        faces, labels = prepare_training_data("training-data")
        print("Data prepared")

        print("Total faces: ", len(faces))
        print("Total labels: ", len(labels))

        face_recognizer = cv2.face.LBPHFaceRecognizer_create()  # Object to train

        face_recognizer.train(faces, np.array(labels))
        face_recognizer.save('recognizer/train.yml')
except OSError:
    print('Error: Creating directory. ' + newDir)








