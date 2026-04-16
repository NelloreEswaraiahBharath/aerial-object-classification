# imports
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# load model
model = load_model("models/best_model.h5")

# ui
st.title("Aerial Object Classifier (Bird vs Drone)")

file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if file is not None:
    img = Image.open(file)
    st.image(img, caption="Uploaded Image")

    # preprocessing
    img = img.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    # prediction
    pred = model.predict(img)[0][0]

    if pred > 0.5:
        st.success("🚁 Drone")
    else:
        st.success("🐦 Bird")

    st.write(f"Confidence: {pred:.2f}")