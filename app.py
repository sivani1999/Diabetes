import numpy as np                            
import pandas as pd
import cv2             
from tensorflow import keras
from PIL import Image                           
import os                                                
from keras.models import Sequential, load_model                                   
import warnings

from tensorflow.keras.preprocessing.image import img_to_array

import tensorflow as tf
from flask import Flask, request, render_template

DEFAULT_IMAGE_SIZE = target_size = (64, 64)

def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None:
            image = cv2.resize(image, DEFAULT_IMAGE_SIZE)
            return img_to_array(image)
        else:
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None

model=tf.keras.models.load_model('./model/model.h5')
classes = { '0':'Mild','1':'Moderate','2':'No_DR','3':'Proliferate_DR','4':'Severe'}
app=Flask(__name__)
#UPLOAD_FOLDER='/UPLOAD'
#app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER

@app.route("/")
def func():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    # print(request.files)
    if 'image_file' not in request.files:
        return "no image"
    else:
        image=request.files['image_file']
        path=os.path.join(image.filename)
        image.save(path)

        test_image = convert_image_to_array(path)
        test_image = np.expand_dims(test_image, axis = 0)
        test_image=np.array(test_image)
        pred=int(model.predict_classes(test_image)[0])
        
        res=[]
        if pred==0:
            res.append('Mild')
        elif pred==1:
            res.append('Moderate')
        elif pred==2:
            res.append('No_DR')
        elif pred==3:
            res.append('Proliferate_DR')
        elif pred==4:
            res.append('Severe')
        return render_template('result.html',name=res)
        
if __name__=='__main__':
    app.run()
        
