#facedatacollection

import cv2
import numpy as np

#initialize camera
cap= cv2.VideoCapture(0)

#Face Detection
face_cascade= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
skip=0
face_data=[]
dataset_path='./data/'

file_name= input("Enter name of Person: ")

while True:
    ret, frame=cap.read()

    if ret==False:
        continue


    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces=face_cascade.detectMultiScale(frame,1.3,5)
    faces= sorted(faces, key=lambda f:f[2]*f[3])

    for face in faces[-1:]:
        x,y,w,h=face
        cv2.rectangle(frame,(x,y),(x+w,y+h), (0,255,255),2)

        #Extract the reqired face or region of interest
        offset=10
        face_section= frame[y-offset: y+h+offset, x-offset:x+w+offset]
        face_section= cv2.resize(face_section,(100,100))

        skip+=1
        if skip%10==0:
            face_data.append(face_section)
            print(len(face_data))


        cv2.imshow("Frame", frame)
        cv2.imshow("Face Section", face_section)



    #Store every 10th face
    
    if(skip%10==0):
        pass





    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

#Convert our face list array into a numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))
print(face_data.shape)

#save data into file system
np.save(dataset_path+file_name+".npy", face_data)
print("Data Successfully save at "+ dataset_path+file_name+'.npy')

cap.release()
cv2.destroyAllWindows()




