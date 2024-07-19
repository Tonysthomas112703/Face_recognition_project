

import cv2 as cv
import numpy as np
import pickle
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.metrics import accuracy_score
import pyttsx3
import threading

# Load the trained SVM model from the pickle file
with open('s_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the label encoder
with open('label_encoder.pkl', 'rb') as le_file:
    encoder = pickle.load(le_file)

# Load FaceNet for face embedding
embedder = FaceNet()

# Initialize MTCNN for face detection
detector = MTCNN()

engine = pyttsx3.init()

# Function to extract face embeddings from an image
def get_embedding(face_img):
    face_img = face_img.astype('float32')  # 3D (160x160x3)
    face_img = np.expand_dims(face_img, axis=0)  # 4D (Nonex160x160x3)
    embeddings = embedder.embeddings(face_img)
    return embeddings[0]  # 512D image (1x1x512)

# Function to perform speech synthesis in a separate thread
def speak_label(label):
    engine.say(label)
    engine.runAndWait()

# Function to perform live video recognition
def live_video_recognition():
    # Open video capture device (0 for webcam)
    cap = cv.VideoCapture(0)

    y_true = []  # True labels
    y_preds = []  # Predicted labels
    prev="Not_in_data"
   
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Detect faces in the frame
        faces = detector.detect_faces(frame)

        # Iterate over detected faces
        for result in faces:
            # Extract bounding box coordinates
            x, y, w, h = result['box']

            # Extract face region
            face = frame[y:y+h, x:x+w]

            # Resize face image to FaceNet input size
            face_resized = cv.resize(face, (160, 160))

            # Get face embeddings
            embedding = get_embedding(face_resized)

            # Perform prediction using the SVM model
            prediction = model.predict([embedding])[0]
           
            # Decode the predicted label using the label encoder
            predicted_label = encoder.inverse_transform([prediction])[0]
           
            # Append true and predicted labels for evaluation
            y_true.append('true_label')  # Replace 'true_label' with actual true label
            y_preds.append(predicted_label)

            # Get the confidence score corresponding to the predicted label
            confidence_score = model.predict_proba([embedding])[0][prediction]

            # Concatenate label and accuracy score
            label_accuracy_text = f"{predicted_label}: {confidence_score:.2f}"

           
            # Speak the predicted label
            if prev!=predicted_label and predicted_label!="Not_in_data":
                prev=predicted_label
                # Create a thread for speech synthesis
                speech_thread = threading.Thread(target=speak_label, args=(predicted_label,))
                speech_thread.start()

            # Draw bounding box and label with accuracy on the frame
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(frame, label_accuracy_text, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the resulting frame
        cv.imshow('Live Video Recognition', frame)

        # Exit loop if 'q' key is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture device and close windows
    cap.release()
    cv.destroyAllWindows()

# Run live video recognition
live_video_recognition()
