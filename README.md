# Sign Language Translator

A real-time sign language recognition system that translates hand gestures into text using computer vision and machine learning.

## Features
- Real-time hand gesture detection
- Translates sign language to text
- Trained on custom dataset

## Tech Stack
- Python
- OpenCV
- MediaPipe
- Flask
- Scikit-learn

## How to Run
1. Clone the repo
   git clone https://github.com/Sakshamm0111/sign_language_translator
2. Install dependencies
   pip install -r requirements.txt
3. Run the app
   python app.py
4. Open browser at http://localhost:5000

## Project Structure
- app.py - Main Flask application
- train_model.py - Model training script
- collect_data.py - Data collection script
- models/ - Saved trained models
- data/ - Training dataset
