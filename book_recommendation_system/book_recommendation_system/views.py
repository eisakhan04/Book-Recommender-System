from django.shortcuts import render ,HttpResponse
import pandas as pd
import pickle
import numpy as np

# Load the DataFrame
popular = pd.read_pickle(r"C:\Users\Mega Computers\Most Advance Machine Learning Projects\Clustering Project Book Recommender System\popular_df.pkl")
model = pickle.load(open(r"C:\Users\Mega Computers\Most Advance Machine Learning Projects\Clustering Project Book Recommender System\model.pkl", 'rb'))
book_pivot = pd.read_pickle(r"C:\Users\Mega Computers\Most Advance Machine Learning Projects\Clustering Project Book Recommender System\book_pivot.pkl")
book = pickle.load(open(r"C:\Users\Mega Computers\Most Advance Machine Learning Projects\Clustering Project Book Recommender System\book.pkl", 'rb'))

def home(request):
    # Prepare data for the template
    books = [
        {
            'title': title,
            'author': author,
            'image': img,
            'votes': vote,
            'rating': rate
        }
        for title, author, img, vote, rate in zip(
            popular['title'],
            popular['author'],
            popular['Image_URL_M'],
            popular['num_rating'],
            popular['avg_rating']
        )
    ]

    context = {
        'books': books
    }

    return render(request, 'home.html', context)

def recommend(request):
    return render(request, 'recomend.html')

def recommend_book(request):
    user_input = request.GET.get('user_input')

    if user_input and user_input in book_pivot.index:
        # Get the book ID from the pivot table index
        book_id = np.where(book_pivot.index == user_input)[0][0]

        # Get distances and suggestions (neighbors)
        distances, suggestions = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)

        data = []
        for i in range(1, len(suggestions[0])):  # Start from 1 to skip the input book itself
            suggested_book_title = book_pivot.index[suggestions[0][i]]
            temp_df = book[book['title'] == suggested_book_title]
            item = {
                'title': temp_df.drop_duplicates('title')['title'].values[0],
                'author': temp_df.drop_duplicates('title')['author'].values[0],
                'image_url': temp_df.drop_duplicates('title')['Image_URL_M'].values[0]
            }
            data.append(item)

        context = {
            'books': data,
        }
        return render(request, 'recomend.html', context)
    
    else:
        return HttpResponse("Book not found or no recommendations available.")