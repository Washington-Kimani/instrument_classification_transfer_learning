import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
import os

st.header('Image Classification Model')

# Load the model
model = load_model(str(os.path.join('models', 'instrument_classification.h5')))
# model = load_model('/home/codename/projects/learning/ds n ml/models/trained_successfully/instrument_classifier.h5')

# Image categories
# data_cat = ['drums', 'guitar', 'piano', 'trumpet', 'violin']
data_cat = ['Didgeridoo',
 'Tambourine',
 'Xylophone',
 'acordian',
 'alphorn',
 'bagpipes',
 'banjo',
 'bongo drum',
 'casaba',
 'castanets',
 'clarinet',
 'clavichord',
 'concertina',
 'drums',
 'dulcimer',
 'flute',
 'guiro',
 'guitar',
 'harmonica',
 'harp',
 'marakas',
 'ocarina',
 'piano',
 'saxaphone',
 'sitar',
 'steel drum',
 'trombone',
 'trumpet',
 'tuba',
 'violin']

# Image dimensions
img_height = 224
img_width = 224

# File uploader
uploaded_file = st.file_uploader("Choose an image...")

if uploaded_file is not None:
    # Save the uploaded image to a temporary file
    with open(os.path.join("temp", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load the image
    image_path = os.path.join("temp", uploaded_file.name)
    image_load = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))

    # Preprocess the image
    img_arr = tf.keras.utils.img_to_array(image_load)
    img_bat = tf.expand_dims(img_arr, 0)

    # Make prediction
    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict)

    # Display the image and prediction
    st.image(image_load, width=200)
    st.write('The image is a: ' + data_cat[np.argmax(score)])
    accuracy = np.max(score)*100
    rounded = round(accuracy, 2)
    st.write('With accuracy of ' + str(rounded))