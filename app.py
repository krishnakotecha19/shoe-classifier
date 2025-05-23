import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from keras.layers import TFSMLayer
from keras.models import Sequential
import sys
import streamlit as st
st.write("Python version:", sys.version)


@st.cache_resource
def load_model():
    model_layer = TFSMLayer("model.savedmodel", call_endpoint="serving_default")
    model = Sequential([model_layer])
    return model

model = load_model()

class_names = ["proper", "torn"]

st.title("Shoe Condition Classifier")
st.write("Upload an image of a shoe, and the model will tell you if it's torn or proper.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    if isinstance(prediction, np.ndarray):
        proper_shoes_prob = prediction[0][0]
        torn_shoes_prob = prediction[0][1]
    elif isinstance(prediction, dict):
        key = list(prediction.keys())[0]
        proper_shoes_prob = prediction[key][0][0]
        torn_shoes_prob = prediction[key][0][1]
    else:
        st.write("Unexpected prediction format")
        st.stop()

    predicted_index = np.argmax([proper_shoes_prob, torn_shoes_prob])
    confidence = np.max([proper_shoes_prob, torn_shoes_prob])

    st.write(f"Proper Shoes probability: {proper_shoes_prob * 100:.2f}%")
    st.write(f"Torn Shoes probability: {torn_shoes_prob * 100:.2f}%")

    st.success(f"Prediction: {class_names[predicted_index]}")
    st.info(f"Confidence: {confidence * 100:.2f}%")
