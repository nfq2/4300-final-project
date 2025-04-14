import json
import os
import re
import numpy as np
from collections import Counter, defaultdict
from flask import Flask, render_template, request
from flask_cors import CORS
import pandas as pd
from math import radians, cos, sin, asin, sqrt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))
current_directory = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(current_directory, 'init.json')

STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 'when', 
    'where', 'how', 'of', 'to', 'in', 'for', 'with', 'by', 'about', 'against', 
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 
    'from', 'up', 'down', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 
    'once', 'here', 'there', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 
    'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now',
    'hotel', 'room', 'rooms', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing'
}

with open(json_file_path, 'r', encoding='utf-8') as file:
    hotel_data = json.load(file)
    hotels_df = pd.DataFrame(hotel_data)

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text):
    words = text.split()
    return {word for word in words if word not in STOPWORDS and len(word) > 1}

hotel_tokens = []

def initialize_hotel_tokens():
    global hotels_df, vectorizer, svd_model, doc_vectors

    hotels_df['combined_text'] = ''
    if 'Description' in hotels_df.columns:
        hotels_df['combined_text'] += hotels_df['Description'].fillna('').apply(preprocess_text)
    if 'HotelFacilities' in hotels_df.columns:
        hotels_df['combined_text'] += ' ' + hotels_df['HotelFacilities'].fillna('').apply(preprocess_text)
    if 'Attractions' in hotels_df.columns:
        hotels_df['combined_text'] += ' ' + hotels_df['Attractions'].fillna('').apply(preprocess_text)

    vectorizer = TfidfVectorizer(stop_words=list(STOPWORDS))
    X = vectorizer.fit_transform(hotels_df['combined_text'])

    n_components = 100 if X.shape[1] >= 100 else (X.shape[1] - 1 if X.shape[1] > 1 else 1)
    svd_model = TruncatedSVD(n_components=n_components)
    doc_vectors = svd_model.fit_transform(X)

    
    print(f"Tokenized {len(hotels_df)} hotels")

def cosine_similarity(v1, v2):
    """
    Compute cosine similarity between a single vector v1 (1D) and each row in v2 (2D).
    """
    dot_products = np.dot(v2, v1.T).flatten()
    v1_norm = np.linalg.norm(v1)
    v2_norms = np.linalg.norm(v2, axis=1)
    v2_norms[v2_norms == 0] = 1e-10
    if v1_norm == 0:
        v1_norm = 1e-10
    return dot_products / (v2_norms * v1_norm)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

def apply_rocchio(query_vector, doc_vectors, similarities, top_k=5, alpha=1.0, beta=0.75):
    top_indices = np.argsort(similarities)[-top_k:]
    if len(top_indices) > 0:
        relevant_mean = np.mean(doc_vectors[top_indices], axis=0)
        modified_query = alpha * query_vector + beta * relevant_mean
        return modified_query
    return query_vector

def json_search(query, user_lat=None, user_lon=None, unit="km", sort_order="default", top_n=10):
    global hotels_df, vectorizer, svd_model, doc_vectors

    if doc_vectors is None or not query:
        return '[]'

    try:
        processed_query = preprocess_text(query)
        query_tfidf = vectorizer.transform([processed_query])
        query_vector = svd_model.transform(query_tfidf)[0]

        initial_similarities = cosine_similarity(query_vector, doc_vectors)

        modified_query_vector = apply_rocchio(query_vector, doc_vectors, initial_similarities, top_k=5, alpha=1.0, beta=0.75)

        similarities = cosine_similarity(modified_query_vector, doc_vectors)

        distances = []
        for idx in range(len(hotels_df)):
            if user_lat is not None and user_lon is not None:
                try:
                    map_val = hotels_df.iloc[idx].get('Map', '')
                    lat_str, lon_str = map_val.split('|')
                    hotel_lat = float(lat_str)
                    hotel_lon = float(lon_str)
                    dist_km = haversine(user_lat, user_lon, hotel_lat, hotel_lon)
                    dist = dist_km if unit == "km" else dist_km * 0.621371
                    distances.append(round(dist, 1))
                except Exception:
                    distances.append(None)
            else:
                distances.append(None)

        results_df = hotels_df.copy()
        results_df['similarity_score'] = similarities
        results_df['distance_km'] = distances

        results_df = results_df.sort_values('similarity_score', ascending=False)
        top_results = results_df.head(top_n)

        if sort_order in ['asc', 'desc'] and user_lat is not None and user_lon is not None:
            top_results = top_results.sort_values('distance_km', ascending=(sort_order == 'asc'))

        if user_lat is not None and user_lon is not None:
            def add_distance(row):
                desc = row.get('Description', '')
                if row['distance_km'] is not None:
                    return f"{desc} ({row['distance_km']} {'km' if unit == 'km' else 'mi'} away from you)"
                return desc
            top_results['Description'] = top_results.apply(add_distance, axis=1)

        if 'HotelWebsiteUrl' in top_results.columns:
            top_results['imageSearchLink'] = top_results['HotelWebsiteUrl'].apply(
                    lambda url: f"https://www.google.com/search?tbm=isch&q={url}" if isinstance(url, str) else ""
    )
            top_results['hotelSearchLink'] = top_results['HotelWebsiteUrl'].apply(
                lambda url: url if isinstance(url, str) and url.startswith("http") else "#"
    )


        columns_to_include = ['HotelName', 'Description', 'HotelFacilities', 'cityName', 'hotelSearchLink', 'countyName','similarity_score']
        if 'imageSearchLink' in top_results.columns:
            columns_to_include.append('imageSearchLink')
        return top_results[columns_to_include].to_json(orient='records')

    except Exception as e:
        print(f"Error in json_search: {str(e)}")
        return json.dumps({"error": str(e)})


app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return render_template('base.html', title="sample html")

@app.route("/hotels", methods=['GET'])
def hotels_search():
    app.logger.info("Initializing hotel tokens")
    initialize_hotel_tokens()
    app.logger.info("Processing search request")
    text = request.args.get("query", "")
    user_lat = request.args.get("lat", type=float)
    user_lon = request.args.get("lon", type=float)
    unit = request.args.get("unit", default="km")
    sort_order = request.args.get("sort", default="default")
    return json_search(text, user_lat=user_lat, user_lon=user_lon, unit=unit, sort_order=sort_order)

@app.route("/episodes", methods=['GET'])
def episodes_search():
    text = request.args.get("query", "")
    return json_search(text)

@app.route("/ping", methods=['GET'])
def ping():
    app.logger.info("pong")
    return "pong"

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)