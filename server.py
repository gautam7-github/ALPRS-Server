import numpy as np
import streamlit as st
import torch
from easyocr import Reader
from paddleocr import PaddleOCR
from PIL import Image

# adding a file uploader
hideMenuStyle = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
# st.markdown(hideMenuStyle, unsafe_allow_html=True)


def load_image(image_file):
    img = Image.open(image_file)
    return img


def buildApp():
    st.subheader("Online ALPR System")
    image_file = st.file_uploader(
        "Upload Vehicle Image", type=["png", "jpg", "jpeg"], accept_multiple_files=False
    )
    if image_file is not None:
        image = load_image(image_file)
        # To See details
        file_details = {
            "filename": image_file.name,
            "filetype": image_file.type,
            "filesize": image_file.size,
        }
        st.write(file_details)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                '<h2 style="text-align: center;">Original Image</h2>',
                unsafe_allow_html=True,
            )
        with col2:
            st.image(image, width=416)

        # model inference and plate localisation
        with st.spinner("MODEL Inference Running"):
            converted_img = np.array(image)
            det = model(converted_img)
            xmin, ymin, xmax, ymax = det.pandas().xyxy[0].loc[0].values[:4]
            roi = converted_img[int(ymin) : int(ymax), int(xmin) : int(xmax)]
            results = reader.readtext(roi)
            # resultsPaddle = ocr.ocr(roi, cls=True)
            # print(resultsPaddle)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                '<h2 style="text-align: center;">Detected License Plate</h2>',
                unsafe_allow_html=True,
            )

        with col2:
            st.image(det.render()[0], width=416)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                '<h2 style="text-align: center;">Plate Text</h2>',
                unsafe_allow_html=True,
            )

        with col2:
            for (bbox, text, prob) in results:
                st.markdown(
                    f'<p style="text-align: center;">{text}</p>', unsafe_allow_html=True
                )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                '<h2 style="text-align: center;">ROI License Plate</h2>',
                unsafe_allow_html=True,
            )

        with col2:
            st.image(roi, width=416, channels="RGB")


if __name__ == "__main__":
    with st.spinner("Initalising Resources"):
        model = torch.hub.load(
            "ultralytics/yolov5", "custom", path="model-3.pt", force_reload=False
        )
        ocr = PaddleOCR(use_angle_cls=True, lang="en")
        reader = Reader(lang_list=["en"])
    buildApp()
