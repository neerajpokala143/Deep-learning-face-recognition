import os
import streamlit as st
import pandas as pd
import numpy as np
import tempfile
from PIL import Image
import asyncio
import io
import glob
import sys
import time
import uuid
import requests
from urllib.parse import urlparse
from io import BytesIO
# To install this module, run:
# python -m pip install Pillow
from io import BytesIO
from PIL import ImageDraw
import json

        
#if app_mode=='Face Recognization':
  #st.image(os.path.join('./images','face recognition.jpg'),use_column_width=True )
st.title("Face Recognition with deep learning Powered by Azure")
st.header('Face Recognition:')
st.markdown("Using Azure I build to detect, identify and analyse faces in images.")
st.text("Detect the objects in images")
image_file =  st.file_uploader("Upload Images (less than 1mb)", type=["png","jpg","jpeg"])
if image_file is not None:
  img = Image.open(image_file)
  #st.image(image_file,width=250,caption='Uploaded image')
  byte_io = BytesIO()
  img.save(byte_io, 'PNG')#PNG
  image = byte_io.getvalue()


  button_translate=st.button('Click me to show recognized photo',help='To give the image')

  if button_translate and image_file :

    def draw_face(img):

        subscription_key = 'ea8c44f876804e43ab35a26a09d59da5'  
        BASE_URL = "https://recognition-ai.cognitiveservices.azure.com/" + '/face/v1.0/detect'  
        headers = {
        # Request headers
        'Content-Type': 'application/octet-stream', 
        'recognitionModel': 'recognition_01' ,
        'Ocp-Apim-Subscription-Key': subscription_key,
        'detectionModel': 'detection_01',
        }
        response = requests.post(BASE_URL, headers=headers, data=image)
        faces = response.json()
        print(faces)
        def getRectangle(faceDictionary):
            rect = faceDictionary['faceRectangle']
            left = rect['left']
            top = rect['top']
            bottom = left + rect['height']
            right = top + rect['width']
            return ((left, top), (bottom, right))


        output_image = Image.open(BytesIO(image))
        
    #   For each face returned use the face rectangle and draw a red box.
        draw = ImageDraw.Draw(output_image)
        for face in faces:
            draw.rectangle(getRectangle(face), outline='red',width=2)
        return output_image
   #image_data = open(image_file, "rb").read()

    image = draw_face(img)

    st.image(image, caption='Output image')
    
    
    
