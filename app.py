import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO
import asyncio
import os

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

model = YOLO("yolov10n.pt")

st.title("🛠 YOLOv10 Object Detection App 🎈")
st.write("Upload an image and press 'Detect Objects' to start detection.")

detect_button = st.button("Detect Objects 🚀")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.write(f"✅ Uploaded File: {uploaded_file.name}")

    allowed_extensions = [".png", ".jpg", ".jpeg"]
    if not any(uploaded_file.name.endswith(ext) for ext in allowed_extensions):
        st.error("❌ Invalid file format! Please upload a .png, .jpg, or .jpeg file.")
    else:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if detect_button:  
            try:
                results = model(image)

                detections = results[0].boxes.data.cpu().numpy()  
                labels = results[0].names
                confidence_threshold = 0.5 

                detected_image = results[0].plot()

                st.image(detected_image, caption="Detected Objects", use_column_width=True)

                st.write("### 🔍 Detected Objects:")
                for det in detections:
                    x1, y1, x2, y2, conf, cls_idx = det
                    if conf >= confidence_threshold:
                        st.write(f"🔹 **{labels[int(cls_idx)]}** (Confidence: {conf:.2f})")

            except Exception as e:
                st.error(f"❌ Error during detection: {str(e)}")

st.write("💡 Need help? Check out [docs.streamlit.io](https://docs.streamlit.io/) for more features.")

if __name__ == "__main__":
    os.system("streamlit run " + __file__)
