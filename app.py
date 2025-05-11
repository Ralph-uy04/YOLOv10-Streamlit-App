import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO

# Load YOLOv10 model
model = YOLO("yolov10n.pt")

st.title("YOLOv10 Object Detection")
st.write("Upload an image for object detection.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform detection
    results = model(image)

    # Display detection results
    st.image(results[0].plot(), caption="Detected Objects", use_column_width=True)
