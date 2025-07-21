import streamlit as st
import joblib
import numpy as np

# ------------------ Load Model and Label Encoder ------------------
model = joblib.load("dna_classifier_rf.pkl")
le = joblib.load("label_encoder.pkl")

# 🔍 Dynamically determine max sequence length from model input size
EXPECTED_INPUT_SIZE = model.n_features_in_  # e.g., 3665
MAX_SEQ_LEN = EXPECTED_INPUT_SIZE // 4      # Each base has 4 values

# ------------------ One-Hot Encoding Function ------------------
def one_hot_encode_seq(seq, max_len):
    mapping = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1],
        'N': [0, 0, 0, 0]  # Unknown base
    }
    seq = seq.upper()
    seq = seq[:max_len]                      # Trim if too long
    padded_seq = seq.ljust(max_len, 'N')     # Pad with 'N' if short
    encoded = [mapping.get(base, [0, 0, 0, 0]) for base in padded_seq]
    return np.array(encoded).flatten()

# ------------------ Streamlit UI ------------------
st.title("🧬 DNA Sequence Classifier")
st.markdown("Enter a DNA sequence (A, C, G, T) to predict its **gene type**.")

# Input from user
user_input = st.text_area("📥 Enter DNA sequence", height=150)

if st.button("🔍 Predict"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a DNA sequence first.")
    else:
        try:
            # One-hot encode + extra feature
            encoded_seq = one_hot_encode_seq(user_input.strip(), MAX_SEQ_LEN)
            seq_len = len(user_input.strip())   # Extra feature
            final_input = np.append(encoded_seq, seq_len)  # (3664 + 1 = 3665)

            input_data = final_input.reshape(1, -1)
            prediction = model.predict(input_data)[0]
            predicted_label = le.inverse_transform([prediction])[0]

            st.success(f"🧠 Predicted Gene Type: **{predicted_label}**")
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and scikit-learn")
