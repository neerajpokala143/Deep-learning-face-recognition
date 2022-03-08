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


st.sidebar.title("Crop prediction")

Mode = st.sidebar.selectbox('Choose mode',
                            ['About Project', 'About Me', 'Crop Predictor',
                                'Dataset', 'Range of Crops']
                            )

if Mode == 'About Project':

    st.markdown(
        """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <title>Document</title>
            <style>
                .class1 {
                    font-family: 'Times New Roman', Times, serif;
                    font-size: 50px;
                    font-weight: bold;
                    color:lightseagreen;
                }
                .class2 {
                    padding-left: 300px;
                    font-style: italic;
                    color:crimson;
                }
                .classmain {
                    display: flex;
                    flex-flow: column;
                    text-align: center;
                }
                ul {
                    list-style: disc;
                }
                #id1 {
                    font-family: 'Times New Roman', Times, serif;
                    padding-left: 20px;
                    font-size: 30px;
                    color: cornflowerblue;
                }
                #id2 {
                    font-family: 'Times New Roman', Times, serif;
                    padding-left: 20px;
                    font-size: 30px;
                    color: cornflowerblue;
                }
                #id3 {
                    font-family: 'Times New Roman', Times, serif;
                    padding-left: 20px;
                    font-size: 30px;
                    color: cornflowerblue;
                }
                p{
                    padding-left: 20px;
                }
            </style>
            <link rel="stylesheet" href="crop.css">
        </head>
        <body>
            <div class="classmain">
                <div class="class1"><u>Crop Recommendation</u></div>
                <div class="class2">Maximize agricultural yield by recommending appropriate crops</div>
            </div>
            <p id="id1"><strong><u>Prerequisites:</u></strong></p>
            <p>
            <ul>
                <li>
                    In general, agriculture is the backbone of India and also plays an important role in Indian economy
                    by providing a certain percentage of domestic product to ensure the food security. But now-a-days, food
                    production and prediction are getting depleted due to unnatural climatic changes, which will adversely
                    affect the economy of farmers by getting a poor yield and also help the farmers to remain less familiar
                    in forecasting the future crops.
                </li>
                <li>
                    This project work helps the beginner farmer in such a way to guide them for sowing the reasonable
                    crops according to the physical parameters of the soil. This achieved by deploying machine learning, one
                    of the advanced technologies in crop prediction. Logistic regression, a supervised learning algorithm
                    puts forth in the way to achieve it. The crops are determined here, with the appropriate physical
                    parameters which effects the crops and helps the crops to achieve a successful growth.
                </li>
                <li>
                    In addition as the software, a Web application being developed. The users are encouraged to enter
                    parameters will be taken automatically in this application in order to start the prediction process.
                </li>
            </ul>
            </p>
        
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
    
    
    
