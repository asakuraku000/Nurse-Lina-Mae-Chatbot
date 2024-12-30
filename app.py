from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import pickle
import random
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'medical_chatbot_model.h5')
METADATA_PATH = os.path.join(MODEL_DIR, 'chatbot_metadata.pickle')

# Create models directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Load the trained model and metadata
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    
    with open(METADATA_PATH, 'rb') as handle:
        metadata = pickle.load(handle)
    
    tokenizer = metadata['tokenizer']
    label_mapping = metadata['label_mapping']
    responses = metadata['responses']
    max_sequence_length = metadata['max_sequence_length']
    
    print("Model and metadata loaded successfully!")
    
except Exception as e:
    print(f"Error loading model or metadata: {str(e)}")
    print("Please ensure you have trained the model first using train_model.py")

def get_response(text):
    # Tokenize and pad the input text
    sequences = tokenizer.texts_to_sequences([text])
    padded_seq = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')
    
    # Get prediction
    prediction = model.predict(padded_seq)
    predicted_label_index = np.argmax(prediction)
    
    # Convert back to label
    reverse_mapping = {idx: label for label, idx in label_mapping.items()}
    predicted_label = reverse_mapping[predicted_label_index]
    
    # Get random response for the predicted intent
    possible_responses = responses[predicted_label]
    response = random.choice(possible_responses)
    
    return {
        'response': response,
        'intent': predicted_label,
        'confidence': float(prediction[0][predicted_label_index])
    }

@app.route('/')
def home():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    message = request.json.get('message', '')
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    try:
        result = get_response(message)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Change this to bind to 0.0.0.0 and use environment port
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
