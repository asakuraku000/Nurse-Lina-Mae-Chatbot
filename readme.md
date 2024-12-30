# Nurse Lina Mae Medical Chatbot

A medical chatbot powered by deep learning that provides health-related information and advice. The chatbot takes on the persona of Nurse Lina Mae to provide friendly and professional medical guidance.

## Features

- Medical knowledge base covering:
  - Blood pressure information
  - Hydration guidelines
  - Exercise recommendations
  - Sleep advice
  - Nutrition tips
- Interactive web interface
- Real-time responses
- Professional and caring communication style

## Installation

1. Clone the repository:
```bash
git clone https://github.com/asakuraku000/Nurse-Lina-Mae-Chatbot.git
cd Nurse-Lina-Mae-Chatbot
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the training script to generate the model:
```bash
python train_model.py
```

4. Start the Flask server:
```bash
python app.py
```

5. Open your browser and navigate to `http://localhost:5000`

## Project Structure

```
Nurse-Lina-Mae-Chatbot/
├── app.py                  # Flask application
├── train_model.py         # Model training script
├── requirements.txt       # Project dependencies
├── templates/
│   └── chat.html         # Chat interface
├── static/
│   └── css/
│       └── style.css     # Custom styles
└── models/               # Trained model files
```

## Usage

1. Type your medical question in the chat interface
2. Press Enter or click Send
3. Receive professional medical information from Nurse Lina Mae

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.# Nurse-Lina-Mae-Chatbot
