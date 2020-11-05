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


class face_recog:
    
    def __init__(self):
        
        
        self.imagePaths = list(paths.list_images(os.path.join(os.getcwd(),'dataset'))) #define path to data
        self.knownEncodings = []
        self.knownNames = []
        self.csv_folder = os.path.join(os.getcwd(),"CSV_files") # path to csv file
        
        
    def Encode_dataset(self):
        print("Encoding Started")
		
		# read all the images one by on in the dataset
        for (i, imagePath) in enumerate(self.imagePaths):
            print("Encoding {} out of {}".format(i + 1,len(self.imagePaths)))
            
            name = imagePath.split(os.path.sep)[-2]
            image = cv2.imread(imagePath) # read the image
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert it to RGB mode
			
			# detect the coordinates of the bounding boxes for each face and compute the facial embedding for the face
            boxes = face_recognition.face_locations(rgb,model="cnn")
            encodings = face_recognition.face_encodings(rgb, boxes)
            
            for encoding in encodings:
			
			# add each encoding + name to our set of known names and encodings
                self.knownEncodings.append(encoding)
                self.knownNames.append(name)
                
        data = {"encodings": self.knownEncodings, "names": self.knownNames}
        f = open('test.pickle', "wb")
        f.write(pickle.dumps(data)) # store them in a pickle file
        f.close()
        
        
    def face_d(self,stream):

        self.cwd = os.getcwd() # get current working directory path
        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") # load cascade file

            
        self.cap = cv2.VideoCapture(stream)# video capture from the given source
        
        name = "Unknown"
        # end the loop when the video ends
        while True:
            ret, img = self.cap.read()  # read the frames
            if ret == False:
                break
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert the frame into grayscale
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5) # detect faces
            
            
            for (x,y,w,h) in faces:
                
                cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2) # draw bounding boxes arounf faces
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]

                name = self.Recognition(roi_color) # classify the name

                if name != "Unknown":
                    cv2.putText(img, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2) # if a face is identified put text on bounding box
                    cv2.destroyAllWindows() # destroy All Windows
                    cv2.imshow('img',img) # display frames
                    self.save_data(name) # save data
            k = cv2.waitKey(30) & 0xff
            if k==27 or name != "Unknown":
                ret, img = self.cap.read()
                break

        self.cap.release()
        return name
        
        
    def Recognition(self,image):
        
        self.data = pickle.loads(open("test.pickle", "rb").read()) #load pickle file
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert bgr to rgb
		
		# locate the faces and recognize them
        boxes = face_recognition.face_locations(rgb,model="cnn")
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []
        name = "Unknown"
        
		# attempt to match each face in the input image to our known encodings
        for encoding in encodings:

        	matches = face_recognition.compare_faces(self.data["encodings"],encoding)
        	
        
			# if found a known face executes rest
        	if True in matches:
				# find the indexes of all matched faces then initialize a dictionary to count the total number of times each face was matched
        		matchedIdxs = [i for (i, b) in enumerate(matches) if b]
        		counts = {}
        
        		for i in matchedIdxs:
        			name = self.data["names"][i]
        			counts[name] = counts.get(name, 0) + 1
        		name = max(counts, key=counts.get)

        	names.append(name) # update the names list
        return name
    
    def recognise_from_an_image(self,img):
        
        
       img = cv2.imread(img)
       gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
       
       self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
       faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
       
       for (x,y,w,h) in faces:
           cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
           roi_gray = gray[y:y+h, x:x+w]
           roi_color = img[y:y+h, x:x+w]
           name = self.Recognition(roi_color)
                  
       return name
    
	
	# save the data into a CSV file
    def save_data(self,name):
        emplist = []
        now = datetime.now()          
        csv_file_name = "{}.csv".format(now.strftime("%x").replace("/","-"))
		
		# create a csv file if not exist
        if not os.path.exists(os.path.join(self.csv_folder,csv_file_name)):
            with open(os.path.join(self.csv_folder,csv_file_name), 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Name", "Time"])
						
        if name not in emplist:
		
			# find the detected time
            now = datetime.now()          
            emp_time = now.strftime("%X")
            data_list = [name,emp_time]
            emplist.append(name)
			
			# store the data into CSV file
            with open(os.path.join(self.csv_folder,csv_file_name), 'a+', newline='') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow(data_list)
x =face_recog()
g = x.face_d(0) #video stream
print(g)
#print(x.recognise_from_an_image("testH.jpg"))
#x.Encode_dataset()
cv2.destroyAllWindows()