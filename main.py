from flask import Flask, render_template
from flask_restful import Api

import os
import csv
import pandas as pd
import numpy as np

from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
api = Api(app)

@app.route("/")
def home():
	print("im home")
	compute_cosine_matrix()
	return render_template("home.html")

# @app.route("/upload_image", methods=['GET'])
# def 

def weighted_rating(x,):
	C = 5.618207215133889
	m = 160.0
	v = x['vote_count']
	R = x['vote_average']
	return (v/(v+m) * R) + (m/(m+v) * C)

def get_recommendations(title, cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]

    movie_indices = [i[0] for i in sim_scores]

    return metadata['title'].iloc[movie_indices]

def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        if len(names) > 3:
            names = names[:3]
        return names
    return []

def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

def compute_cosine_matrix():
	metadata = pd.read_csv('data/movies_metadata.csv', low_memory=False)
	keywords = pd.read_csv('data/keywords.csv')
	# credits = pd.read_csv('data/credits.csv')
	credits_1 = pd.read_csv('data/credits_1.csv')
	credits_2 = pd.read_csv('data/credits_2.csv')
	credits_3 = pd.read_csv('data/credits_3.csv')
	credits = pd.concat([credits_1, credits_2, credits_3])

	credits = credits.reset_index()
	del credits['index']

	C = metadata['vote_average'].mean()
	m = metadata['vote_count'].quantile(0.90)

	q_movies = metadata.copy().loc[metadata['vote_count'] >= m]
	q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
	q_movies = q_movies.sort_values('score', ascending=False)
	q_movies[['title', 'vote_count', 'vote_average', 'score']].head()

	indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()

	metadata = metadata.drop([19730, 29503, 35587])
	keywords['id'] = keywords['id'].astype('int')
	credits['id'] = credits['id'].astype('int')
	metadata['id'] = metadata['id'].astype('int')

	metadata = metadata.merge(credits, on='id')
	metadata = metadata.merge(keywords, on='id')

	features = ['cast', 'crew', 'keywords', 'genres']
	for feature in features:
	    metadata[feature] = metadata[feature].apply(literal_eval)

	metadata['director'] = metadata['crew'].apply(get_director)

	features = ['cast', 'keywords', 'genres']
	for feature in features:
	    metadata[feature] = metadata[feature].apply(get_list)

	features = ['cast', 'keywords', 'director', 'genres']

	for feature in features:
	    metadata[feature] = metadata[feature].apply(clean_data)

	metadata['soup'] = metadata.apply(create_soup, axis=1)

	count = CountVectorizer(stop_words='english')
	count_matrix = count.fit_transform(metadata['soup'])
	cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
	metadata = metadata.reset_index()
	indices = pd.Series(metadata.index, index=metadata['title'])
	print(get_recommendations('Toy Story', cosine_sim2))



if __name__ == "__main__":
	# app.run(debug=True)
	app.run(debug=True, host='0.0.0.0', port=80)