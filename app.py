import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load Model
model = load_model("fruit_freshness_model.h5")

# Class Labels (Adjust if needed)
class_labels = [
    'freshapples',
    'freshbanana',
    'freshoranges',
    'rottenapples',
    'rottenbanana',
    'rottenoranges'
]

# Streamlit UI
st.set_page_config(page_title="Fruit Freshness Classifier", layout="centered")
st.markdown(
    """
    <style>
    /* Make sidebar wider */
    [data-testid="stSidebar"] {
        width: 400px;
        min-width: 400px;
    }

    /* Optional: adjust main content to avoid overlap */
    [data-testid="stMainContent"] {
        margin-left: 410px;
    }

    /* Increase font size for sidebar titles */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        font-size: 28px;  /* for titles/headers */
    }

    /* Increase font size for regular sidebar text */
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] li {
        font-size: 20px;  /* for text and list items */
    }
    "<h2 style='text-align: center; font-weight: bold;'>This is a bold centered sentence</h2>"
    </style>
    """,
    unsafe_allow_html=True
)
st.sidebar.title("AICW (Artificial Intellegence Career For Women) By Microsoft and SAP in collaboration with Edunet Foundation")
st.sidebar.title("👥 Team Information")
  # Replace with your guide's name

st.sidebar.write("*Team Members:*")
st.sidebar.write("- Somireddy Devika")  # Replace with actual names
st.sidebar.write("- Gali Anusha Devi")
st.sidebar.write("- Nemala Haritha Lavanya")
st.sidebar.write("- Pyda Tejasai Bharathi")
st.sidebar.title("PROJECT GUIDE")
st.sidebar.write("### Abdul Aziz Md")
st.sidebar.write("Master Trainer-Edunet Foundation")

st.title("🍎 Fruit Freshness Classification")
st.write("Upload a fruit image to check whether it is Fresh or Rotten")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Resize (⚠️ Must match training size)
    img = img.resize((150,150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Prediction
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    result = class_labels[class_index]
    confidence = np.max(prediction) * 100

    st.subheader("🔍 Prediction Result")

    if "fresh" in result:
        st.success(f"✅ The fruit is {result}")
    else:
        st.error(f"❌ The fruit is {result}")

    st.info(f"Confidence: {confidence:.2f}%")
