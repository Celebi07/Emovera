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
emotion_scores = {'angry': [], 'disgust': [], 'fear': [], 'happy': [], 'sad': [], 'surprise': [], 'neutral': [], 'calm': []}

# Set up your credentials
client_id = '8dbb11386d6c469ca24674e25da5a9f5'
client_secret = '2ca4fa4a31724ff2af8cdc7e32704501'

# Create an instance of the Spotify client
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

@app.route('/')
def index():
    return render_template('index.html')

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

            emotion_detection = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral', 'calm')
            emotion_prediction = emotion_detection[max_index]
            score = predictions[0][max_index]

            label = f"{emotion_prediction}: {score:.2f}"
            cv2.putText(test_image, label, (int(x), int(y) + h+ 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            # Store the score in the dictionary
            emotion_scores[emotion_prediction].append(score)

            # Get recommended songs and books based on the highest emotion score
            if score > 0.5:  # Adjust this threshold as needed
                recommended_songs = get_random_songs(emotion_prediction, limit=5)
                recommended_books = get_emotion_based_books(emotion_prediction, limit=5)
                display_recommendations(emotion_prediction, recommended_songs, recommended_books)

            resize_image = cv2.resize(test_image, (1000, 700))
            _, jpeg = cv2.imencode('.jpg', resize_image)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Existing book recommendation code
def get_emotion_based_books(emotion, limit):
    base_url = 'https://www.dbooks.org/api/search/'

    emotion_to_genre = {
        'happy': 'Fiction',
        'sad': 'Drama',
        'excited': 'Adventure',
        'calm': 'Self-Help',
        'mysterious': 'Mystery',
        'angry': 'Psychology',
        'disgust': 'Horror',
        'fear': 'Thriller',
        'surprise': 'Suspense',
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

# Existing song recommendation code
def get_random_songs(genre, limit):
    random.seed()

    # Search for tracks in the given genre
    results = sp.search(q='genre:' + genre, type='track', limit=limit)
    tracks = results['tracks']['items']

    # Shuffle the tracks randomly
    random.shuffle(tracks)

    # Extract the desired metadata for each track
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

def display_recommendations(emotion, recommended_songs, recommended_books):
    print(f"Emotion: {emotion}")
    print("Recommended Songs:")
    for song in recommended_songs:
        print(f"  - {song['name']} by {', '.join(song['artists'])}")
    print("Recommended Books:")
    for book in recommended_books:
        print(f"  - {book['title']} by {book['author']}")

if __name__ == '__main__':
    app.run(debug=True)
    # Calculate and display average scores in the terminal after the app is closed
    for emotion, scores in emotion_scores.items():
        if scores:
            average_score = sum(scores) / len(scores)
            print(f"Average {emotion} score: {average_score:.2f}")
        else:
            print(f"No {emotion} scores recorded.")
