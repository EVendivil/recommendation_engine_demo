from flask import Flask, render_template, request
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

cosine_sim = ""
names = ""

@app.route("/")
def home():
	print("im home")
	compute_cosine_matrix()
	return render_template("home.html")

# @app.route("/upload_image", methods=['GET'])
# def 

@app.route("/recommend", methods=['POST'])
def get_movie_recom():
	movie = request.form['movie_title']
	print(movie)
	recomm_movies = get_recommendations(movie)
	return render_template("recom.html", data=recomm_movies)

def get_recommendations(title):
    idx = names[names.str.lower().str.contains(title.lower(), na=False)].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]

    movie_indices = [i[0] for i in sim_scores]

    return metadata_cleaned['title'].iloc[movie_indices].to_list()

def compute_cosine_matrix():
	# metadata_cleaned = pd.read_csv('data/metadata_cleaned.csv')
	metadata_cleaned_1 = pd.read_csv('data/metadata_cleaned_1.csv')
	metadata_cleaned_2 = pd.read_csv('data/metadata_cleaned_2.csv')
	metadata_cleaned = pd.concat([metadata_cleaned_1, metadata_cleaned_2])
	metadata_cleaned = metadata_cleaned.reset_index()
	del metadata_cleaned['Unnamed: 0']
	del metadata_cleaned['index']


	# indices = pd.Series(metadata_cleaned.index, index=metadata_cleaned['title']).drop_duplicates()
	count = CountVectorizer(stop_words='english')
	count_matrix = count.fit_transform(metadata_cleaned['soup'])
	cosine_sim = cosine_similarity(count_matrix, count_matrix)
	metadata_cleaned = metadata_cleaned.reset_index()
	names = pd.Series(metadata_cleaned['title'],metadata_cleaned.index).drop_duplicates()
	# print(get_recommendations('Toy Story', cosine_sim))

if __name__ == "__main__":
	# app.run(debug=True)
	app.run(debug=True, host='0.0.0.0', port=80)