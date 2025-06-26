import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import numpy as np
import cv2

# App settings
st.set_page_config(page_title="Digit Recognizer", layout="centered")
st.title("✍️ Handwritten Digit Recognizer")
st.markdown("Draw a digit below (0–9) and click **Predict** to see the result.")

# Load model
model = tf.keras.models.load_model("mnist_cnn_model.h5")

# Canvas for drawing
canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=20,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Prediction button
if st.button("Predict"):
    if canvas_result.image_data is not None:
        img = canvas_result.image_data

        # Convert to uint8 and BGR format for OpenCV
        img = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_RGBA2GRAY)

        # Resize to 28x28 and invert (to match MNIST white digits)
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        img = cv2.bitwise_not(img)

        # Optional: Threshold to clean up image
        _, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

        # Normalize
        img = img.astype("float32") / 255.0
        img = img.reshape(1, 28, 28, 1)

        # Show processed input
        st.image(img.reshape(28, 28), width=150, caption="Model Input")

        # Predict
        prediction = model.predict(img)
        pred_class = np.argmax(prediction)

        st.success(f"🔢 Predicted Digit: **{pred_class}**")
    else:
        st.warning("Please draw a digit before predicting.")
