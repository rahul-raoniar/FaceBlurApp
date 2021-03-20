pip install 

import streamlit as st
import numpy as np
from PIL import Image
import cv2


@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    return img


face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
def detect_and_blur_face(img):
    face_img = img.copy()
    face_points = face_cascade.detectMultiScale(face_img, scaleFactor=1.5, minNeighbors=6)
    
    for (x,y,w,h) in face_points:
        roi = face_img[y:y+h, x:x+w]
        face_img[y:y+h, x:x+w] = 255
    return face_img 



def main():
    st.title("Face Remover App")

    menu = ["Home", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        image_file = st.file_uploader("Upload an images",
                                      type = ["png", "jpg", "jpeg"]) 
        if image_file is not None:
            if st.button("Process"):
                st.subheader("Original Image")
                st.image(load_image(image_file))

                img = load_image(image_file)
                img_array = np.asarray(img)

                result = detect_and_blur_face(img_array)
                st.subheader("Processed Image")
                st.image(result)


    else:
        st.subheader("About")



if __name__ == "__main__":
    main()