import cv2
import numpy as np
from flask import Flask, render_template, Response
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
import requests
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import random

app = Flask(__name__)

# Store emotion scores over a period of time
emotion_scores = {'angry': [], 'disgust': [], 'fear': [], 'happy': [], 'sad': [], 'surprise': [], 'neutral': []}

def gen():
    model = model_from_json(open("model.json", "r").read())
    model.load_weights('best_model.h5')
    face_haar_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    while True:
        valid, test_image = cap.read()
        if not valid:
            break

        gray_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

        faces_detected = face_haar_cascade.detectMultiScale(gray_image)
        for (x, y, w, h) in faces_detected:
            cv2.rectangle(test_image, (x, y), (x + w, y + h), (255, 0, 0)) 
            roi_gray = gray_image[y:y + w, x:x + h]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            image_pixels = img_to_array(roi_gray)
            image_pixels = np.expand_dims(image_pixels, axis=0)
            predictions = model.predict(image_pixels)
            max_index = np.argmax(predictions[0])
            
            emotion_detection = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            emotion_prediction = emotion_detection[max_index]
            score = predictions[0][max_index]  # Get the confidence score for the predicted emotion
            
            # Draw the emotion label and score on the frame
            label = f"{emotion_prediction}: {score:.2f}"
            cv2.putText(test_image, label, (int(x), int(y) + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Store the score in the dictionary
            emotion_scores[emotion_prediction].append(score)
            
            resize_image = cv2.resize(test_image, (1000, 700))
            _, jpeg = cv2.imencode('.jpg', resize_image)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def get_emotion_based_books(emotion, limit):
    base_url = 'https://www.dbooks.org/api/search/'

    emotion_to_genre = {
        'happy': 'Fiction',
        'sad': 'Drama',
        'excited': 'Adventure',
        'calm': 'Self-Help',
        'mysterious': 'Mystery',
        # Add more emotions and genres as needed
    }

    genre = emotion_to_genre.get(emotion.lower())
    if not genre:
        return []

    try:
        api_url = f'{base_url}{genre}?count={limit}'
        response = requests.get(api_url)
        response.raise_for_status()

        data = response.json()

        book_list = []
        for book in data.get('books', [])[:limit]:
            book_info = {
                'title': book.get('title', 'Unknown Title'),
                'author': book.get('authors', 'Unknown Author'),
                'published_year': book.get('year', 'Unknown Year'),
                'cover_image': book.get('image', ''),
                'link': book.get('url', '')
            }
            book_list.append(book_info)

        return book_list
    except requests.exceptions.RequestException as req_err:
        print(f"Request error occurred: {req_err}")
        return []

def get_random_songs(genre, limit):
    random.seed()
    client_id = 'YOUR_SPOTIFY_CLIENT_ID'
    client_secret = 'YOUR_SPOTIFY_CLIENT_SECRET'
    client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    results = sp.search(q='genre:' + genre, type='track', limit=limit)
    tracks = results['tracks']['items']
    random.shuffle(tracks)

    song_list = []
    for track in tracks:
        song = {
            'name': track['name'],
            'artists': [artist['name'] for artist in track['artists']],
            'album': track['album']['name'],
            'release_year': track['album']['release_date'][:4],
            'cover_photo': track['album']['images'][0]['url'],
            'spotify_link': track['external_urls']['spotify']
        }
        song_list.append(song)

    return song_list

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/recommendations/<emotion>')
def recommendations(emotion):
    recommended_books = get_emotion_based_books(emotion, limit=5)
    recommended_songs = get_random_songs(emotion, limit=5)
    return render_template('recommendations.html', emotion=emotion, books=recommended_books, songs=recommended_songs)

if __name__ == '__main__':
    app.run(debug=True)
    # Calculate and display average scores in the terminal after the app is closed
    for emotion, scores in emotion_scores.items():
        if scores:
            average_score = sum(scores) / len(scores)
            print(f"Average {emotion} score: {average_score:.2f}")
        else:
            print(f"No {emotion} scores recorded.")
