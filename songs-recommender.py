import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import random

# Set up your credentials
client_id = '9856682f7d6646968b877568499c908c'
client_secret = '449f66c050ba4970aacb28a3e0e4c346'

# Create an instance of the Spotify client
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def get_random_songs(genre, limit):
    random.seed()  # Set a new random seed for each request
    
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

# Usage example
# genres = ['rock', 'pop', 'hip hop', 'jazz', 'country', 'reggae', 'classical', 
#           'electronic', 'indie', 'alternative', 'r&b', 'metal', 'folk', 'punk', 
#           'dance', 'blues', 'soul', 'latin', 'world']

genre = 'world'  # Replace with the genre you want
random_songs = get_random_songs(genre, limit=50)
con = 1
for song in random_songs:
    print(con)
    con = con+1
    print('Name:', song['name'])
    print('Artists:', ', '.join(song['artists']))
    print('Album:', song['album'])
    print('Release Year:', song['release_year'])
    print('Cover Photo:', song['cover_photo'])
    print('Spotify Link:', song['spotify_link'])
    print('--------------------------------------------------------------')