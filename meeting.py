from imutils import paths
import face_recognition
import pickle
import cv2
import os
import cv2
import numpy as np
import pickle
from PIL import Image
import threading
import time
from datetime import datetime
import csv



class meeting_attendance:
    
    def __init__(self):
        
        self.imagePaths = list(paths.list_images(os.path.join(os.getcwd(),'dataset'))) #define path to data
        self.knownEncodings = []
        self.knownNames = []
        self.data = pickle.loads(open("face3.pickle", "rb").read()) #loading pickle file
        self.csv_folder = os.path.join(os.getcwd(),"CSV_files") # path to csv file
        
        
    def face_d(self): #detecting face and crop

        self.cwd = os.getcwd() # get current working directory path
        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") # load cascade file

            
        self.cap = cv2.VideoCapture('9.mp4') # video capture from the file
        start_time = time.time() # record the starting time
        
        name = "Unknown"
        names = []
        
        while True:
            end_time = time.time()
            ret, img = self.cap.read() # read the frames
            
			# end the loop when the video ends
            if ret == False:
                break
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert the frame into grayscale
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5) # detect faces

            duration = time.strftime("%S", time.gmtime(end_time - start_time)) # calculate duration
            
			# capture the faces and mark the attendance
			# check faces in every 2 munites
            if duration == '02':

                for (x,y,w,h) in faces:
                    #cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
                    name_list = self.Recognition(img) # predict names
                    for i in range(len(name_list)):
                        if name_list[i] not in names and name_list[i] != "Unknown":
                            names.append(name_list[i])
                start_time = end_time
            cv2.imshow("face",img) # display frames
                

            k = cv2.waitKey(30) & 0xff
			# break the loop when the user hits esc button
            if k==27:
                ret, img = self.cap.read()
                break

        self.cap.release()
        self.store_csv(names) # call the store_csv function

        
    # recognition function
    def Recognition(self,image):
        cv2.destroyAllWindows() # destroy all windows
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert bgr to rgb
		
		# locate the faces and recognize them
        boxes = face_recognition.face_locations(rgb,model="cnn")
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []
        name = "Unknown"
        for encoding in encodings:
            matches = face_recognition.compare_faces(self.data["encodings"],encoding)
            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                for i in matchedIdxs:
                    name = self.data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get) # get the predicted name
            names.append(name) # append the name to names list
        for ((top, right, bottom, left), name) in zip(boxes, names):
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2) # put a text
        cv2.imshow("Recognized",image) # display frames
        return names # return names list

        
    def store_csv(self, data_list):
        current = os.getcwd()  # get current working directory path
        path = os.path.join(current,"Meeting_CSV")
        now = datetime.now()
        meeting_date = now.strftime("%x")
        meeting_time = now.strftime("%X")
        
        datetime_value = "{}_{}".format(meeting_date,meeting_time)
        csv_file_name = "Meetings.csv"
		# create csv if not exist
        if not os.path.exists(os.path.join(path,csv_file_name)):
            with open(os.path.join(path,csv_file_name), 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Date and Time", "Names"])

        data_list = [datetime_value,data_list]
		
		# store data into the csv file
        with open(os.path.join(path,csv_file_name), 'a+', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(data_list)

g = meeting_attendance()
g.face_d()
cv2.destroyAllWindows()