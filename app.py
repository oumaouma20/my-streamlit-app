# app.py

import streamlit as st
import requests
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

# ✅ STEP 1: UI Inputs
st.title("🍅 Tomato Farming Advisory AI System")
st.markdown("Upload a **tomato leaf image**, get a **disease diagnosis**, and view **weather-based advice**.")

lang = st.selectbox("🌍 Select Language", ["english", "swahili"])
location = st.text_input("📍 Enter Location (e.g., Kerugoya)", value="Kerugoya")

# ✅ STEP 2: Weather API Call
api_key = "1d5af17ac5484fae2f35780c93a44fb7"  # Replace with your actual key
humidity = 0
temperature = 0

if location:
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
        response = requests.get(url)
        data = response.json()
        humidity = data["main"]["humidity"]
        temperature = data["main"]["temp"]
        st.success(f"📍 Location: {location} | 🌡️ {temperature}°C | 💧 {humidity}% humidity")
    except:
        st.warning("⚠️ Could not fetch weather data. Check location or API key.")

# ✅ STEP 3: Load model
@st.cache_resource
def load_tomato_model():
    return load_model("tomato_model.keras")

model = load_tomato_model()

# ✅ STEP 4: Upload and Predict
uploaded_file = st.file_uploader("📤 Upload a tomato leaf image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="🖼️ Uploaded Image", use_column_width=True)

    # Preprocess
    img = img.resize((256, 256))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    class_labels = ['Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_healthy']
    predicted_index = np.argmax(prediction)
    predicted_label = class_labels[predicted_index]

    st.subheader(f"✅ Prediction: {predicted_label.replace('_', ' ')}")

    # ✅ STEP 5: Explanation
    def explain_diagnosis(label, humidity, lang):
        if label == 'Tomato_Early_blight':
            if lang == 'swahili':
                st.markdown("🦠 *Ugonjwa: Early Blight (Alternaria solani)*")
                if humidity > 80:
                    st.error("🚨 Unyevu mwingi! Hatari ya kuenea kwa ugonjwa huu.")
                    st.info("🛠️ Tumia dawa ya kuvu kama chlorothalonil.")
                else:
                    st.info("⚠️ Tibu ugonjwa. Hakikisha majani ni makavu.")
            else:
                st.markdown("🦠 *Disease: Early Blight (Alternaria solani)*")
                if humidity > 80:
                    st.error("🚨 High humidity! Favorable for early blight.")
                    st.info("🛠️ Apply chlorothalonil fungicide.")
                else:
                    st.info("⚠️ Treat early blight. Maintain dry leaves.")

        elif label == 'Tomato_Late_blight':
            if lang == 'swahili':
                st.markdown("🦠 *Ugonjwa: Late Blight (Phytophthora infestans)*")
                if humidity > 80:
                    st.error("🚨 Unyevu mwingi sana! Hatari ya kuenea kwa late blight.")
                    st.info("🛠️ Tumia dawa za copper.")
                else:
                    st.info("⚠️ Hali ya hewa ni ya wastani.")
            else:
                st.markdown("🦠 *Disease: Late Blight (Phytophthora infestans)*")
                if humidity > 80:
                    st.error("🚨 ALERT: Ideal for late blight outbreak.")
                    st.info("🛠️ Apply copper-based fungicide.")
                else:
                    st.info("⚠️ Moderate risk. Maintain airflow.")

        elif label == 'Tomato_healthy':
            if lang == 'swahili':
                st.success("✅ Mimea ina afya!")
                if humidity > 80:
                    st.warning("⚠️ Unyevu mwingi. Hatari ya ugonjwa ipo.")
                else:
                    st.info("🌞 Hali ya hewa ni nzuri.")
            else:
                st.success("✅ Plant is healthy!")
                if humidity > 80:
                    st.warning("⚠️ High humidity. Fungal risk exists.")
                else:
                    st.info("🌞 Weather is favorable.")

    explain_diagnosis(predicted_label, humidity, lang)
