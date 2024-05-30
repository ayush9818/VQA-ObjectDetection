import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from PIL import Image
from utils import *
from model.clip_vqa_predictor import VQAWithCLIPModule
from model.blip_vqa_predictor import VQAWithBLIPModule
from model.vilt_vqa_predictor import VQAWithVILTModule
from model.ans_detect_predictor import ADModule
import torch
import os
from ultralytics import YOLO
import shutil

# set device to "cuda" to call the GPU
device = "cuda" if torch.cuda.is_available() else 'cpu'
# set page layout
st.set_page_config(
    page_title="Visual Question Answering",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title('Visual Question Answering')

def main():
    st.sidebar.subheader("Input")
    models_list = ["CLIP", "VILT"]
    selected_model = st.sidebar.selectbox("Select the Model", models_list)
   
    make_prediction(selected_model)

def make_prediction(selected_model):
    text_input = st.sidebar.text_input('Type a question', key="text_input")
    selected_img = st.sidebar.file_uploader("Upload an image", type=["png","jpg","jpeg"])

    enable_object_detection = st.sidebar.checkbox('Object Detection', value=True)

    if not text_input.strip():
        st.warning('Please type a question.')

    if selected_img is None:
        st.warning('Please upload an image.')
    else:
        st.header('Uploaded Image', divider='rainbow')
        temp_directory = 'temp/'
        # Convert uploaded image data into a PIL image object
        pil_image = Image.open(selected_img)     
        # Resize the image while maintaining the original aspect ratio
        resized_img = resize_image(pil_image, target_height=500)
        # Display the resized image
        st.image(resized_img, caption="Resized Image", use_column_width='auto')
        
        if isinstance(resized_img, list):  # Check if the function returned a list
            resized_img = resized_img[0]  # Assuming you want the first image

        # Save the resized image to use in prediction
        resized_path = os.path.join(temp_directory, 'resized_image.jpg')
        resized_img.save(resized_path)
 
    ad_predictor = ADModule(device=device)

    selected_model = selected_model.lower()

    if selected_model == "vilt":
        vqa_predictor = VQAWithVILTModule(device=device)
    else:
        vqa_predictor = VQAWithCLIPModule(device=device) # Default to CLIP if model not selected

    if selected_img is not None and (text_input is not None and text_input.strip()):
        if enable_object_detection:
            st.header('Image Annotation and Labeling', divider='rainbow')
            model = YOLO('model_store/yolov8_best.pt')
            # Path where the predicted results will be saved
            labeled_path = os.path.join(temp_directory, 'yolov8_predictions.jpg')
            # Perform prediction and save the prediction results temporarily
            results = model.predict(resized_path, imgsz=640, conf=0.5, save=True)

            if isinstance(results, list):  # Check if the function returned a list
                results = results[0]  # Assuming you want the first image

            # Now, save the image to your specified path
            results.save(labeled_path)

            # Convert uploaded image data into a PIL image object
            pil_image = Image.open(labeled_path)  
            st.image(pil_image, caption="Labled Image", use_column_width='auto')

        # Check if the file exists and delete it if it does
        if os.path.exists(resized_path):
            os.remove(resized_path)
        if os.path.exists(labeled_path):
            os.remove(labeled_path)
        with st.spinner("Thinking..."):
            if selected_model == "clip":
                ad_ans = ad_predictor.predict(image_path = selected_img, question=text_preprocess(text_input))
                print(f'Answerability Detection Score: {ad_ans:.2f}')
                if ad_ans < 0.65:
                    vqa_ans = f'This question is unanswerable.'
                else:
                    vqa_ans = vqa_predictor.predict(image_path = selected_img, question=text_preprocess(text_input))
            
                st.info("**Your Question**: " + text_input)
                st.success('**Predicted Answer**: ' + vqa_ans.capitalize() + '. ' + f'The answerability score is {ad_ans:.2f}.')
            else:
                vqa_ans = vqa_predictor.predict(image_path = selected_img, question=text_preprocess(text_input))
                st.info("**Your Question**: " + text_input)
                st.success('**Predicted Answer**: ' + vqa_ans.capitalize())

if __name__ == "__main__":
    main()
    