import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ✅ Load trained model
model = load_model('my_model.h5')

# ✅ Load face detector (use FULL PATH if needed)
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# ✅ Emotion labels
labels_dict = {
    0:'Angry',1:'Disgust',2:'Fear',
    3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'
}

# ✅ Try different camera index (0 or 1)
video = cv2.VideoCapture(0)

# 🔍 Check camera
if not video.isOpened():
    print("❌ Camera not detected. Trying another index...")
    video = cv2.VideoCapture(1)

while True:
    ret, frame = video.read()

    
    if not ret:
        print("❌ Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = faceDetect.detectMultiScale(gray, 1.3, 3)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]

        resized = cv2.resize(face_img, (48,48))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1,48,48,1))

        result = model.predict(reshaped, verbose=0)
        label = np.argmax(result)

        # Draw box
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        # Show emotion
        cv2.putText(frame, labels_dict[label], (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,255,255),2)

    # Show video
    cv2.imshow("Emotion Detector", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release
video.release()
cv2.destroyAllWindows()