import streamlit as st
import joblib
import numpy as np
import os
import gdown

# ------------------ Google Drive Downloads ------------------
MODEL_URL = "https://drive.google.com/uc?id=1nGbtMUYtJAbBtVNI7roWDvN8NcY_Sa0R"
ENCODER_URL = "https://drive.google.com/uc?id=1U6Jv93VYVRoOxRSZWrkAyMrzO_cWku9V"

if not os.path.exists("dna_classifier_rf.pkl"):
    gdown.download(MODEL_URL, "dna_classifier_rf.pkl", quiet=False)

if not os.path.exists("label_encoder.pkl"):
    gdown.download(ENCODER_URL, "label_encoder.pkl", quiet=False)

# ------------------ Load Model and Encoder ------------------
model = joblib.load("dna_classifier_rf.pkl")
le = joblib.load("label_encoder.pkl")

# ------------------ Constants ------------------
FIXED_MAX_LEN = 916  # ⛔️ DO NOT change this unless you retrain model

# ------------------ One-hot Encoder ------------------
def one_hot_encode_seq(seq, max_len):
    mapping = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1],
        'N': [0, 0, 0, 0]
    }
    seq = seq.upper()
    seq = seq[:max_len]
    padded_seq = seq.ljust(max_len, 'N')
    encoded = [mapping.get(base, [0, 0, 0, 0]) for base in padded_seq]
    return np.array(encoded).flatten()

# ------------------ Streamlit App ------------------
st.title("🧬 DNA Classifier (Dynamic Input ✅)")
st.markdown("Paste your DNA sequence (A/C/G/T only). It will be auto-truncated/padded as needed.")

user_input = st.text_area("📝 Enter DNA sequence")

if st.button("🚀 Predict"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a DNA sequence.")
    else:
        try:
            seq_len = len(user_input.strip())
            encoded = one_hot_encode_seq(user_input.strip(), FIXED_MAX_LEN)
            final_input = np.append(encoded, seq_len).reshape(1, -1)

            prediction = model.predict(final_input)[0]
            predicted_label = le.inverse_transform([prediction])[0]

            st.success(f"✅ Prediction: **{predicted_label}**")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
