import json
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import pickle

# Medical tips dataset
medical_data = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": [
                "Hi", "Hello", "Good day", "Hey there", "How are you"
            ],
            "responses": [
                "Hello! I'm Nurse Lina Mae, how can I help you today?",
                "Hi there! This is Nurse Lina Mae. What medical concerns can I address for you?",
                "Good day! I'm Nurse Lina Mae. How may I assist you with your health questions?"
            ]
        },
        {
            "tag": "blood_pressure",
            "patterns": [
                "What is normal blood pressure?",
                "Is my blood pressure too high?",
                "Blood pressure guidelines",
                "What should my BP be?"
            ],
            "responses": [
                "Normal blood pressure is generally considered to be below 120/80 mmHg. If you're concerned about your readings, please consult your healthcare provider.",
                "A healthy blood pressure range is typically between 90/60 mmHg and 120/80 mmHg. Regular monitoring is important for your health."
            ]
        },
        {
            "tag": "hydration",
            "patterns": [
                "How much water should I drink?",
                "Signs of dehydration",
                "Daily water intake",
                "Am I drinking enough water?"
            ],
            "responses": [
                "It's recommended to drink about 8 glasses (2 liters) of water daily. Remember to increase intake during exercise or hot weather.",
                "Watch for signs of dehydration like dark urine, thirst, dry mouth, and fatigue. Most adults need 2-3 liters of water daily."
            ]
        },
        {
            "tag": "exercise",
            "patterns": [
                "How often should I exercise?",
                "Best exercises for beginners",
                "Physical activity guidelines",
                "Exercise recommendations"
            ],
            "responses": [
                "Adults should aim for at least 150 minutes of moderate aerobic activity or 75 minutes of vigorous activity weekly, plus strength training twice a week.",
                "Start with 30 minutes of moderate exercise most days. This could include brisk walking, swimming, or cycling. Always consult your doctor before starting a new exercise routine."
            ]
        },
        {
            "tag": "sleep",
            "patterns": [
                "How much sleep do I need?",
                "Tips for better sleep",
                "Insomnia help",
                "Sleep hygiene"
            ],
            "responses": [
                "Adults typically need 7-9 hours of sleep per night. Maintain a regular sleep schedule and create a relaxing bedtime routine.",
                "For quality sleep, keep your bedroom cool and dark, avoid screens before bedtime, and try to go to bed at the same time each night."
            ]
        },
        {
            "tag": "nutrition",
            "patterns": [
                "Healthy diet tips",
                "What should I eat?",
                "Balanced meal advice",
                "Nutrition guidelines"
            ],
            "responses": [
                "Focus on a balanced diet with plenty of fruits, vegetables, whole grains, lean proteins, and healthy fats. Limit processed foods and added sugars.",
                "Try to eat a rainbow of vegetables daily, choose whole grains over refined ones, and include protein with each meal. Stay hydrated and watch portion sizes."
            ]
        }
    ]
}

# Save dataset
with open('medical_dataset.json', 'w') as f:
    json.dump(medical_data, f, indent=4)

def prepare_training_data(data):
    training_sentences = []
    training_labels = []
    labels = []
    responses = {}
    
    for intent in data['intents']:
        for pattern in intent['patterns']:
            training_sentences.append(pattern)
            training_labels.append(intent['tag'])
        responses[intent['tag']] = intent['responses']
        if intent['tag'] not in labels:
            labels.append(intent['tag'])
    
    return training_sentences, training_labels, labels, responses

# Prepare training data
training_sentences, training_labels, labels, responses = prepare_training_data(medical_data)

# Tokenize the input sentences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(training_sentences)
total_words = len(tokenizer.word_index) + 1

# Create training sequences and pad them
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences, padding='post')

# Process labels
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)

# Convert string labels to numerical indices
label_mapping = {label: idx for idx, label in enumerate(labels)}
numerical_labels = [label_mapping[label] for label in training_labels]

# Convert to one-hot encoded format
training_label_seq = to_categorical(numerical_labels, num_classes=len(labels))

# Build the model
model = Sequential([
    Embedding(total_words, 100, input_length=padded_sequences.shape[1]),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(len(labels), activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', 
             optimizer=Adam(learning_rate=0.001), 
             metrics=['accuracy'])

print("Model Summary:")
model.summary()

# Train the model
print("\nTraining the model...")
epochs = 500
history = model.fit(padded_sequences, training_label_seq, 
                   epochs=epochs, verbose=1)

# Save the model and necessary data
print("\nSaving the model and data...")
model.save('medical_chatbot_model.h5')

# Save additional data needed for inference
metadata = {
    'tokenizer': tokenizer,
    'label_mapping': label_mapping,
    'responses': responses,
    'max_sequence_length': padded_sequences.shape[1]
}

with open('chatbot_metadata.pickle', 'wb') as handle:
    pickle.dump(metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("\nTraining completed and model saved successfully!")

# Add a test function to verify the model works
def test_model(text, tokenizer, label_mapping, model, max_sequence_length):
    # Tokenize and pad the input text
    sequences = tokenizer.texts_to_sequences([text])
    padded_seq = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')
    
    # Get prediction
    prediction = model.predict(padded_seq)
    predicted_label_index = np.argmax(prediction)
    
    # Convert back to label
    reverse_mapping = {idx: label for label, idx in label_mapping.items()}
    predicted_label = reverse_mapping[predicted_label_index]
    
    return predicted_label, prediction[0][predicted_label_index]

# Test the model with a few examples
print("\nTesting the model with sample queries:")
test_queries = [
    "Hello, how are you?",
    "What should my blood pressure be?",
    "How much water should I drink daily?"
]

for query in test_queries:
    predicted_label, confidence = test_model(
        query, 
        metadata['tokenizer'],
        metadata['label_mapping'],
        model,
        metadata['max_sequence_length']
    )
    print(f"\nQuery: {query}")
    print(f"Predicted Intent: {predicted_label}")
    print(f"Confidence: {confidence:.2f}")

# Print sample response
print("\nSample response from the model:")
for query in test_queries:
    predicted_label, _ = test_model(
        query,
        metadata['tokenizer'],
        metadata['label_mapping'],
        model,
        metadata['max_sequence_length']
    )
    possible_responses = responses[predicted_label]
    response = np.random.choice(possible_responses)
    print(f"\nQuery: {query}")
    print(f"Response: {response}")