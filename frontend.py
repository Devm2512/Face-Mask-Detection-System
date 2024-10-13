import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
thicc = 2
count=0
score=0
rpred=[99]
lpred=[99]
st.title("Drowsiness Detection System")
st.sidebar.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSxUUxxMeR9gDXjU4Ron1sxH0Obzjbo9wj5Mw&usqp=CAU")
choice = st.sidebar.selectbox("Menu",("Home","URL","Camera"))
if (choice =="Home"):
    st.image("https://cdn.hackernoon.com/images/oO6rUouOWRYlzw88QM9pb0KyMIJ3-bxfy3m27.png")
elif (choice == "URL"):
    url = st.text_input("Enter your URL with mp4 extension")
    btn = st.button("Start Detection")
    window = st.empty()
    if btn:
        i = 1
        btn2 = st.button("Stop Detection")
        if btn2:
            st.experimental_rerun()
        facemodel = cv2.CascadeClassifier("face.xml")
        leye = cv2.CascadeClassifier("haarcascade_lefteye_2splits.xml")
        reye = cv2.CascadeClassifier("haarcascade_righteye_2splits.xml")
        maskmodel = load_model("keras_model.h5")
        vid = cv2.VideoCapture(url)
        while(vid.isOpened()):
            flag,frame = vid.read()
            if flag:
                faces = facemodel.detectMultiScale(frame)
                left_eye = leye.detectMultiScale(frame)
                right_eye = reye.detectMultiScale(frame)
                for (x,y,l,w) in faces:
                    face_img = frame[y:y+w,x:x+l]
                    face_img = cv2.resize(face_img,(224,224),interpolation=cv2.INTER_AREA)
                    face_img = np.asarray(face_img,dtype = np.float32).reshape(1,224,224,3)
                    face_img = (face_img/127.5)-1
                    p = maskmodel.predict(face_img)[0][0]
                for (x,y,l,w) in right_eye:
                    r_eye=frame[y:y+w,x:x+l]
                    r_eye = cv2.resize(r_eye,(24,24),interpolation=cv2.INTER_AREA)
                    r_eye = np.asarray(r_eye,dtype = np.float32).reshape(24,24,-1)
                    r_eye= (r_eye/127.5)-1
                    r_eye = np.expand_dims(r_eye,axis=0)
                    rpred = maskmodel.predict(r_eye)[0][0]
                    if(rpred > 0.9):
                        path = "nomask/"+ str(i)+".jpg"
                        cv2.imwrite(path,frame[y:y+w,x:x+l])
                        i = i+1
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,0,255),4)
                    else:
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,255,0),4)
                window.image(frame,channels = "BGR")
                for (x,y,l,w) in left_eye:
                    l_eye=frame[y:y+w,x:x+l]
                    l_eye = cv2.resize(l_eye,(24,24),interpolation=cv2.INTER_AREA)
                    l_eye = np.asarray(l_eye,dtype = np.float32).reshape(24,24,-1)
                    l_eye= (l_eye/127.5)-1
                    l_eye = np.expand_dims(l_eye,axis=0)
                    lpred = maskmodel.predict(l_eye)[0][0]
                    if(lpred > 0.9):
                        path = "nomask/"+ str(i)+".jpg"
                        cv2.imwrite(path,frame[y:y+w,x:x+l])
                        i = i+1
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,0,255),2)
                    else:
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,255,0),2)
                window.image(frame,channels = "BGR") 
elif (choice =="Camera"):
    cam = st.selectbox("Choose Camera",("None","Primary","Secondary"))
    btn = st.button("Start Detection")
    window = st.empty()
    if btn:
        i = 1
        btn2 = st.button("Stop Detection")
        if btn2:
            st.experimental_rerun()
        facemodel = cv2.CascadeClassifier("face.xml")
        leye = cv2.CascadeClassifier("haarcascade_lefteye_2splits.xml")
        reye = cv2.CascadeClassifier("haarcascade_righteye_2splits.xml")
        maskmodel = load_model("keras_model.h5")
        if cam =="Primary":
            cam = 0
        else:
            cam = 1
        vid = cv2.VideoCapture(cam)
        while(vid.isOpened()):
            flag,frame = vid.read()
            if flag:
                faces = facemodel.detectMultiScale(frame)
                left_eye = leye.detectMultiScale(frame)
                right_eye = reye.detectMultiScale(frame)
                for (x,y,l,w) in faces:
                    face_img = frame[y:y+w,x:x+l]
                    face_img = cv2.resize(face_img,(224,224),interpolation=cv2.INTER_AREA)
                    face_img = np.asarray(face_img,dtype = np.float32).reshape(1,224,224,3)
                    face_img = (face_img/127.5)-1
                    p = maskmodel.predict(face_img)[0][0]
                for (x,y,l,w) in right_eye:
                    r_eye=frame[y:y+w,x:x+l]
                    r_eye = cv2.resize(r_eye,(24,24),interpolation=cv2.INTER_AREA)
                    r_eye = np.asarray(r_eye,dtype = np.float32).reshape(24,24,-1)
                    r_eye= (r_eye/127.5)-1
                    r_eye = np.expand_dims(r_eye,axis=0)
                    rpred = maskmodel.predict(r_eye)[0][0]
                    if(rpred > 0.9):
                        path = "nomask/"+ str(i)+".jpg"
                        cv2.imwrite(path,frame[y:y+w,x:x+l])
                        i = i+1
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,0,255),4)
                    else:
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,255,0),4)
                window.image(frame,channels = "BGR")
                for (x,y,l,w) in left_eye:
                    l_eye=frame[y:y+w,x:x+l]
                    l_eye = cv2.resize(l_eye,(24,24),interpolation=cv2.INTER_AREA)
                    l_eye = np.asarray(l_eye,dtype = np.float32).reshape(24,24,-1)
                    l_eye= (l_eye/127.5)-1
                    l_eye = np.expand_dims(l_eye,axis=0)
                    lpred = maskmodel.predict(l_eye)[0][0]
                    if(lpred > 0.9):
                        path = "nomask/"+ str(i)+".jpg"
                        cv2.imwrite(path,frame[y:y+w,x:x+l])
                        i = i+1
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,0,255),2)
                    else:
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,255,0),2)
                window.image(frame,channels = "BGR")
    
    
