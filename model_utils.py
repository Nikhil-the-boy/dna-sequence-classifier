import joblib
import numpy as np

# Load the model and encoder
def load_model(model_path='dna_classifier_rf.pkl', encoder_path='label_encoder.pkl'):
    try:
        model = joblib.load(model_path)
        encoder = joblib.load(encoder_path)
        return model, encoder
    except Exception as e:
        print(f"❌ Error loading model or encoder: {e}")
        return None, None

# One-hot encode a DNA sequence
def one_hot_encode_seq(seq, max_len=1000):
    mapping = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1]
    }
    encoded = [mapping.get(char.upper(), [0, 0, 0, 0]) for char in seq]
    
    # Pad or truncate
    if len(encoded) < max_len:
        encoded += [[0, 0, 0, 0]] * (max_len - len(encoded))
    else:
        encoded = encoded[:max_len]

    return np.array(encoded).flatten()
