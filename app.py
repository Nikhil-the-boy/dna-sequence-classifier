import streamlit as st
import os
import gdown
import numpy as np
from model_utils import load_model, one_hot_encode_seq

# Set up Streamlit page
st.set_page_config(page_title="DNA Classifier", layout="wide")
st.title("🧬 DNA Sequence Classifier (Random Forest)")
st.write("Input a DNA sequence to classify it.")

# Google Drive download (only once)
MODEL_URL = "https://drive.google.com/uc?id=1nGbtMUYtJAbBtVNI7roWDvN8NcY_Sa0R"
ENCODER_URL = "https://drive.google.com/uc?id=1Fz8ddcyaUJVmIMLcojZxAqf1fKHn_6Cn"

if not os.path.exists("dna_classifier_rf.pkl"):
    st.info("⏳ Downloading model...")
    gdown.download(MODEL_URL, "dna_classifier_rf.pkl", quiet=False)

if not os.path.exists("label_encoder.pkl"):
    st.info("⏳ Downloading encoder...")
    gdown.download(ENCODER_URL, "label_encoder.pkl", quiet=False)

# Load model + encoder
model, encoder = load_model("dna_classifier_rf.pkl", "label_encoder.pkl")

if model is None or encoder is None:
    st.error("❌ Failed to load model or encoder.")
else:
    seq = st.text_area("Enter DNA Sequence", height=150)
    if st.button("Predict"):
        if seq.strip() == "":
            st.warning("Please enter a sequence.")
        else:
            encoded = one_hot_encode_seq(seq)
            X = np.array(encoded).reshape(1, -1)
            try:
                pred = model.predict(X)
                label = encoder.inverse_transform(pred)[0]
                st.success(f"✅ Predicted Class: {label}")
            except Exception as e:
                st.error(f"Prediction Error: {e}")
