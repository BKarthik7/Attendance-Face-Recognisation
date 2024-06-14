import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch
from sklearn.neighbors import KNeighborsClassifier

def speak(text):
    speaker = Dispatch("SAPI.SpVoice")
    speaker.Speak(text)

# Define file paths
haarcascade_path = 'C:/Users/bangi/Downloads/faceAttendance/haarcascade_frontalface_default .xml'
background_path = 'C:/Users/bangi/Downloads/faceAttendance/bg.png'
names_path = 'C:/Users/bangi/Downloads/faceAttendance/data/names.pkl'
faces_data_path = 'C:/Users/bangi/Downloads/faceAttendance/data/faces_data.pkl'

# Initialize video capture and face detection
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier(haarcascade_path)

if facedetect.empty():
    raise FileNotFoundError(f"Cannot load Haar Cascade XML file at {haarcascade_path}")

# Load background image
imgBackground = cv2.imread(background_path)
if imgBackground is None:
    raise FileNotFoundError(f"Cannot load background image at {background_path}")

# Load labels and face data
with open(names_path, 'rb') as w:
    LABELS = pickle.load(w)
with open(faces_data_path, 'rb') as f:
    FACES = pickle.load(f)

# Ensure training data dimensions
image_size = (50, 50)
if FACES.shape[1] != image_size[0] * image_size[1] * 3:
    raise ValueError(f"Training images should be resized to {image_size} with 3 channels (RGB)")

# Train KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Column names for CSV
COL_NAMES = ['NAME', 'TIME']

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, image_size).flatten().reshape(1, -1)
        
        if resized_img.shape[1] != FACES.shape[1]:
            raise ValueError(f"Input image dimensions do not match training data dimensions")
        
        output = knn.predict(resized_img)
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        date = datetime.now().strftime("%d-%m-%Y")
        attendance_file = f"C:/Users/bangi/Downloads/faceAttendance/Attendance/Attendance_{date}.csv"
        
        # Draw rectangles and label on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, str(output[0]), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        
        attendance = [str(output[0]), str(timestamp)]
    
    # Overlay the frame on the background image
    imgBackground[162:162 + 480, 55:55 + 640] = frame
    cv2.imshow("Frame", imgBackground)
    
    k = cv2.waitKey(1)
    
    if k == ord('o'):
        speak("Attendance Taken..")
        time.sleep(5)
        
        with open(attendance_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not os.path.isfile(attendance_file) or os.path.getsize(attendance_file) == 0:
                writer.writerow(COL_NAMES)
            writer.writerow(attendance)
    
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
