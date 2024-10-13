import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
st.title("FACE MASK DETECTION SYSTEM")
st.sidebar.image("https://cdn.hackernoon.com/images/oO6rUouOWRYlzw88QM9pb0KyMIJ3-82bk3j2w.png")
choice=st.sidebar.selectbox("Menu",("HOME","URL","CAMERA"))
if(choice=="HOME"):
    st.image("data:image/png;base64,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")
elif(choice=="URL"):
    url=st.text_input("Enter your URL")
    btn=st.button("Start Detection")
    window=st.empty()
    if btn:
        i=1
        btn2=st.button("Stop Detection")
        if btn2:
            st.experimental_rerun()
        facemodel=cv2.CascadeClassifier("face.xml")
        maskmodel=load_model("mask.h5")
        vid=cv2.VideoCapture(url)
        while(vid.isOpened()):
            flag,frame=vid.read()
            if flag:
                faces=facemodel.detectMultiScale(frame)
                for (x,y,l,w) in faces:
                    face_img=frame[y:y+w,x:x+l]
                    face_img=cv2.resize(face_img,(224,224),interpolation=cv2.INTER_AREA)
                    face_img=np.asarray(face_img,dtype=np.float32).reshape(1,224,224,3)
                    face_img=(face_img/127.5)-1
                    p=maskmodel.predict(face_img)[0][0]
                    if(p>0.9):
                        path="nomask/"+str(i)+".jpg"
                        cv2.imwrite(path,frame[y:y+w,x:x+l])
                        i=i+1
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,0,255),4)
                    else:
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(255,0,0),4)
                window.image(frame,channels='BGR')       
elif(choice=="CAMERA"):
    cam=st.selectbox("Choose Camera",("None","Primary","Secondary"))
    btn=st.button("Start Detection")
    window=st.empty()
    if btn:
        i=1
        btn2=st.button("Stop Detection")
        if btn2:
            st.experimental_rerun()
        facemodel=cv2.CascadeClassifier("face.xml")
        maskmodel=load_model("mask.h5")
        if cam=="Primary":
            cam=0
        else:
            cam=1
        vid=cv2.VideoCapture(cam)
        while(vid.isOpened()):
            flag,frame=vid.read()
            if flag:
                faces=facemodel.detectMultiScale(frame)
                for (x,y,l,w) in faces:
                    face_img=frame[y:y+w,x:x+l]
                    face_img=cv2.resize(face_img,(224,224),interpolation=cv2.INTER_AREA)
                    face_img=np.asarray(face_img,dtype=np.float32).reshape(1,224,224,3)
                    face_img=(face_img/127.5)-1
                    p=maskmodel.predict(face_img)[0][0]
                    if(p>0.9):
                        path="nomask/"+str(i)+".jpg"
                        cv2.imwrite(path,frame[y:y+w,x:x+l])
                        i=i+1
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,0,255),4)
                    else:
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,255,0),4)
                window.image(frame,channels='BGR')       
