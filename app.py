"""Streamlit web app"""

import cv2
import numpy as np
import streamlit as st
import torch
import types  # Import the types module
from fer_pytorch.fer import FER

def my_hash_func(func):
    return hash((func.__code__, func.__name__))

@st.cache_data(hash_funcs={types.FunctionType: my_hash_func})  # Use st.cache_data instead of st.cache
def cached_fer():
    fer = FER()
    fer.get_pretrained_model("resnet34")
    return fer

def main():
    fer = cached_fer()

    st.title("Recognize emotions")

    # Add a unique key to st.file_uploader to prevent DuplicateWidgetID error
    uploaded_file = st.file_uploader("Choose an image...", type="jpg", key="unique_file_uploader")

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        st.image(image, caption="Before", use_column_width=True, channels="BGR")
        st.write("")
        st.write("Detecting faces...")

        output = fer.predict_image(image, show_top=True)
        if not output:
            st.write("No faces detected")
        else:
            print(output)
            st.image(image, caption="After", use_column_width=True, channels="BGR")

if __name__ == "__main__":
    main()
