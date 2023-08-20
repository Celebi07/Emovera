import cv2
import os
import uuid
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
from tensorflow.keras.metrics import Precision, Recall

# Siamese L1 Distance class
class L1Dist(Layer):
    
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()
       
    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)
    
def preprocess(file_path):
    
    # Read in image from file path
    byte_img = tf.io.read_file(file_path)
    # Load in the image 
    img = tf.io.decode_jpeg(byte_img)
    
    # Preprocessing steps - resizing the image to be 105x105x3
    img = tf.image.resize(img, (105,105))
    # Scale image to be between 0 and 1 
    img = img / 255.0

    # Return image
    return img


# Ask user for name and ID
user_name = input("Enter your name: ")
user_id = input("Enter your ID: ")

# Construct the path to the user's image folder
user_image_folder = os.path.join('VerificationImages', 'Images', user_name + "_" + user_id)

# Check if the user folder already exists
if os.path.exists(user_image_folder):
    print("User already registered. Photos will not be captured.")
else:
    os.makedirs(user_image_folder, exist_ok=True)
    print(f"New user folder created: {user_image_folder}")

    # Load the pre-trained Haar cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Establish a connection to the webcam
    cap = cv2.VideoCapture(0)

    capturing = False
    capture_count = 0

    while cap.isOpened():
        ret, frame = cap.read()

        # Cut down frame to 250x250px
        frame = frame[120:120+250, 200:200+250, :]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        display_frame = frame.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('Face Detection', display_frame)

        key = cv2.waitKey(30)
        if key & 0xFF == ord('v') and len(faces) > 0:
            capturing = True 

        if capturing and len(faces) > 0 and capture_count < 2:
            imgname = os.path.join(user_image_folder, f'{uuid.uuid1()}.jpg')
            cv2.imwrite(imgname, frame)
            cv2.imshow('Image Collection', frame)
            capture_count += 1
            cv2.waitKey(30)

        if capture_count >= 2:
            capturing = False
            print("Capturing complete.")
            break

        if key & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



siamese_model = tf.keras.models.load_model('snn_kash.h5', 
                                   custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})

# # Ask for user name and ID
# user_name = input("Enter the user's name: ")
# user_id = input("Enter the user's ID: ")
# user_folder = f"{user_name}_{user_id}"
# print(user_folder)
# # Construct the path to the user's image folder
# user_image_folder = os.path.join('VerificationImages', 'Images', user_folder)
# # VerificationImages\Images\bho_2
# # Check if the user's folder exists
# if not os.path.exists(user_image_folder):
#     print("User folder not found.")
# else:
#     # List all images in the user's folder
#     for image in os.listdir(os.path.join('VerificationImages', 'Images',user_folder)):
#         validation_img = os.path.join(user_folder, image)
#         print(validation_img) 

def verify(model, detection_threshold, verification_threshold):                             
    # Ask user for name and ID to identify the folder
    # user_name = input("Enter user name for verification: ")
    # user_id = input("Enter user ID for verification: ")
    user_folder = f"{user_name}_{user_id}"
    
    # Construct the path to the user's image folder
    user_image_folder = os.path.join('VerificationImages', 'Images', user_folder)
    
    # Check if the user's folder exists
    if not os.path.exists(user_image_folder):
        print("User folder not found.")
        return
    # Build results array
    results = []
    for image in os.listdir(user_image_folder):
        input_img = preprocess(os.path.join('VerificationImages', 'InputImage', user_folder, 'input_image.jpg'))
        validation_img = preprocess(os.path.join(user_image_folder, image))
        
        # Make Predictions 
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)
    
    # Detection Threshold: Metric above which a prediciton is considered positive 
    detection = np.sum(np.array(results) > detection_threshold)
    
    # Verification Threshold: Proportion of positive predictions / total positive samples 
    verification = detection / len(os.listdir(user_image_folder)) 
    verified = verification > verification_threshold
    
    return results, verified 



# Ask user for name and ID to identify the input image folder
user_name = input("Enter your name: ")
user_id = input("Enter your user ID: ")

# Construct the path to the user's input image folder
input_image_folder = os.path.join('VerificationImages', 'InputImage', f'{user_name}_{user_id}')

# Create the input image folder if it doesn't exist
if not os.path.exists(input_image_folder):
    os.makedirs(input_image_folder)
    print(f"Input image folder created: {input_image_folder}")
else:
    print("Input image folder already exists.")


# Load the pre-trained Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the default camera (camera index 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    frame_roi = frame[120:120+250, 200:200+250, :]
    
    # Display the verification frame
    cv2.imshow('Verification', frame_roi)

    key = cv2.waitKey(10) & 0xFF

    if key == ord('v'):
        # Construct the path to the user's input image folder
        input_image_folder = os.path.join('VerificationImages', 'InputImage', f'{user_name}_{user_id}')
    
    # Create the input image folder if it doesn't exist
        if not os.path.exists(input_image_folder):
            os.makedirs(input_image_folder)
    
        input_image_path = os.path.join(input_image_folder, 'input_image.jpg')
        cv2.imwrite(input_image_path, frame_roi)
        print(f"Input image saved as {input_image_path}")
    
    if key == ord('q'):
        break

cap.release() 
cv2.destroyAllWindows()


# Run verification using your 'verify' function and 'siamese_model'
results, verified = verify(siamese_model, 0.5, 0.5) 
print(verified) 