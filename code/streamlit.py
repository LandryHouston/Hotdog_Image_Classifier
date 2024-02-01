import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np
from keras.preprocessing import image


st.markdown(
    f"<h1 align='center'>Hotdog or not Hotdog</h1>",
    unsafe_allow_html=True,
)


file = st.file_uploader("Upload your image:", type=["jpeg", "jpg", "png"])


if file is not None:
    images = Image.open(file).convert("RGB")
    st.image(images, use_column_width=True)

    model = load_model("model.h5")

    temp_file_path = "temp_image.jpg"
    images.save(temp_file_path)

    img = image.load_img(path=temp_file_path, target_size=(300, 300))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)

    class_labels = ["hotdog", "nothotdog"]
    predicted_class = class_labels[np.argmax(predictions)]

    st.markdown(f"<h3 align='center'>{predicted_class}</h3>", unsafe_allow_html=True)
