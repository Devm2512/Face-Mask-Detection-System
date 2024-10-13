import cv2
import numpy as np
from keras.models import load_model
#To read and image and show it.
img=cv2.imread("Face3.jpg")
facemodel = cv2.CascadeClassifier('face.xml')
maskmodel = load_model('mask.h5',compile=False)
faces = facemodel.detectMultiScale(img)

#to read an image and show it
vid = cv2.VideoCapture("Video2.mp4")
while(vid.isOpened()):
    flag,frame = vid.read()
    if(flag):
        faces = facemodel.detectMultiScale(frame)
        for(x,y,l,h) in faces:
            #cropping the face
            face_img = frame[y:y+h,x:x+l]
    #       Resize the crop face
            face_img = cv2.resize(face_img,(224,224),interpolation=cv2.INTER_AREA)
            #COnverting shape of image according to desire dimension of model
            face_img = np.asarray(face_img,dtype = np.float32).reshape(1,224,224,3)
            #normalize the image
            face_img = (face_img/127.5)-1
            p = maskmodel.predict(face_img)[0][0]
            if p >0.9:
                cv2.rectangle(frame,(x,y),(x+l,y+h),(0,0,255),4)
            else:
                cv2.rectangle(frame,(x,y),(x+l,y+h),(0,255,0),4)
        cv2.namedWindow("dev window",cv2.WINDOW_NORMAL)
        cv2.imshow("dev window",frame)
        k = cv2.waitKey(1)
        if (k==ord('x')):
            break
    else:
        break
cv2.destroyAllWindows()
