An AI-powered application that detects facial emotions using computer vision and recommends songs based on the detected mood.
Features
Detects facial emotions from images/webcam input
Uses a trained deep learning model for emotion recognition
Recommends songs according to the detected emotion
Simple and easy-to-use interface
Technologies Used
Python
OpenCV
TensorFlow / Keras
NumPy
Haar Cascade Classifier
Machine Learning
Project Structure
main.py – Main application file
test.py – Testing script
testdata.py – Test data processing
my_model.h5 – Trained emotion detection model
haarcascade_frontalface_default.xml – Face detection classifier
How to Run

Clone the repository:

git clone https://github.com/renusaklani8-tech/FacialEmotionDetector-SongRecommendator.git

Install the required dependencies:

pip install opencv-python tensorflow numpy

Run the application:

python main.py
Note
The training dataset (archive/) has been excluded from this repository to keep the repository lightweight.

Author
Renu Saklani
