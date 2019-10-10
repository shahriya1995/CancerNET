# main.py

from flask import Blueprint, render_template, redirect, url_for, request, flash
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

import tensorflow as tf
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import requests
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

import pickle
# load the model from disk
#from pyimagesearch import config
from imutils import paths
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import stripe





main = Blueprint('main', __name__)


# import the necessary packages
import os

# initialize the path to the *original* input directory of images
ORIG_INPUT_DATASET = "datasets/orig"

# initialize the base path to the *new* directory that will contain
# our images after computing the training and testing split


b = os.path.dirname(__file__)

new_pred = os.path.sep.join([b, "datasets/idc/new"])
# define the amount of data that will be used training
TRAIN_SPLIT = 0.8

# the amount of validation data will be a percentage of the
# *training* data
VAL_SPLIT = 0.1

# MODEL_PATH ='models/your_model.h5'
# from keras.applications.resnet50 import ResNet50


# model = ResNet50(weights='imagenet')
# print('Model loaded. Check http://127.0.0.1:5000/')

## create stripe pubkey and secret key:

pub_key = 'pk_test_xi2VacvK6q9M2157PIarZVhq009ZCFJgb0'
secret_key ='sk_test_tRp2wh6HjpKumodDlQyw6KYJ00eG052xPY'

stripe.api_key = secret_key

@main.route('/')
def index():
    return render_template('index.html',pub_key=pub_key)


@main.route('/thanks')
def thanks():
    return render_template('profile.html',name=current_user.name)

@main.route('/pay',methods=['POST'])
@login_required
def pay():
    
    customer = stripe.Customer.create(email=request.form['stripeEmail'], source=request.form['stripeToken'])

    charge = stripe.Charge.create(
        customer=customer.id,
        amount=100,
        currency='usd',
        description='The Product'
    )

    return redirect(url_for('main.profile'))


def model_predict():
    basepath1 = os.path.dirname(__file__)
    model_path = os.path.join(basepath1, 'finalized_model.sav')

    loaded_model = pickle.load(open(model_path, 'rb'))
    print(loaded_model)
    BS = 32
    print(new_pred)
    totalTest = len(list(paths.list_images(new_pred)))
    valAug = ImageDataGenerator(rescale=1 / 255.0)

    testGen = valAug.flow_from_directory(  
        new_pred,
        class_mode="categorical",
        target_size=(48, 48),
        color_mode="rgb",
        shuffle=False,
        batch_size=BS)

    testGen.reset()
    predIdxs = loaded_model.predict_generator(testGen,steps=(totalTest // BS) + 1)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
    predIdxs = np.argmax(predIdxs, axis=1)
    print(predIdxs)

    # img = image.load_img(img_path, target_size=(224, 224))

    # # Preprocessing the image
    # x = image.img_to_array(img)
    # # x = np.true_divide(x, 255)
    # x = np.expand_dims(x, axis=0)

    # # Be careful how your trained model deals with the input
    # # otherwise, it won't make correct prediction!
    # x = preprocess_input(x)

    # preds = model.predict(x)
    # console.log("Hello")
    return predIdxs

@main.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'datasets/idc/new/0', secure_filename(f.filename))
        f.save(file_path)
        print("Hello1")
        # Make prediction
        result = model_predict()

        # # Process your result for human
        # # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        # result = str(pred_class[0][0][1])               # Convert to string
        return str(result)
    return render_template('profile.html', name=current_user.name)





