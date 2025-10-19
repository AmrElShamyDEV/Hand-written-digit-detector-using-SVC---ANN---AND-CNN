import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps, ImageFilter
from streamlit_drawable_canvas import st_canvas
import numpy as np
import joblib

st.set_page_config(page_title="Digit Recognizer", page_icon="🔢", layout="centered")


@st.cache_resource
def load_AI_Models():
    ann = load_model("ANN_MODEL.h5")
    cnn = load_model("CNN_MODEL.h5")
    svc = joblib.load("SVC_CLASSIFIER_MODEL.pkl")
    return ann, cnn, svc

ANN_MODEL, CNN_MODEL, SVC_MODEL = load_AI_Models()

st.title("Digit Recognizer using ANN, CNN, and SVM")
st.write("Draw a digit (0–9) in the box below 👇")


canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=15,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

model_choice = st.selectbox("Choose model:", ["ANN Model", "CNN Model", "SVC Model"])

def center_image(img):
    img_array = np.array(img)
    coords = np.column_stack(np.where(img_array > 10))
    if coords.size == 0:
        return img
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    cropped = img_array[y_min:y_max, x_min:x_max]
    new_img = Image.fromarray(cropped).resize((20, 20))
    centered_img = Image.new("L", (28, 28), color=0)
    centered_img.paste(new_img, (4, 4))
    return centered_img

if st.button("Predict"):
    if canvas_result.image_data is not None:
        img = Image.fromarray(canvas_result.image_data.astype("uint8"), "RGBA").convert("RGB").convert("L")
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        img = center_image(img)
        img = img.resize((28, 28))
        img_array = np.array(img)
        img_array = (img_array > 40) * 255  
        img_array = img_array / 255.0

        st.image(img, caption="Processed Input (What Model Sees)", width=150)

        
        if model_choice == "CNN Model":
            input_data = img_array.reshape(1, 28, 28, 1)
            probs = CNN_MODEL.predict(input_data)
            pred_digit = np.argmax(probs)
            confidence = np.max(probs) * 100

        elif model_choice == "ANN Model":
            input_data = img_array.reshape(1, 784)
            probs = ANN_MODEL.predict(input_data)
            pred_digit = np.argmax(probs)
            confidence = np.max(probs) * 100

        else:  
            input_data = img_array.reshape(1, 784)
            pred_digit = SVC_MODEL.predict(input_data)[0]
            probs = None
            confidence = None

       
        if probs is not None:
            st.success(f"Predicted Digit: {pred_digit}  (Confidence: {confidence:.2f}%)")
            st.bar_chart(probs[0])
        else:
            st.success(f"Predicted Digit: {pred_digit} (SVM does not provide confidence scores)")

    else:
        st.warning("Please draw a digit in the canvas.")

if st.button("Clear Canvas"):
    st.experimental_rerun()
