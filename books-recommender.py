import requests

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
        for book in data.get('books', [])[:limit]:  # Limit the number of books to the specified 'limit'
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

# Usage example
emotion = 'happy'  # Replace with the emotion you want
recommended_books = get_emotion_based_books(emotion, limit=5)  # Recommend 5 books
con = 1
for book in recommended_books:
    print(con)
    con = con + 1
    print('Title:', book['title'])
    print('Author:', book['author'])
    print('Published Year:', book['published_year'])
    print('Cover Image:', book['cover_image']) 
    print('Link:', book['link'])
    print('--------------------------------------------------------------')
