import json
import os
import re
import numpy as np
from collections import Counter, defaultdict
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
from math import radians, cos, sin, asin, sqrt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Add these new imports
from datetime import datetime
import pickle

os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))
current_directory = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(current_directory, 'init.json')

# Add feedback file path
feedback_file_path = os.path.join(current_directory, 'user_feedback.pkl')

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

# Add global user feedback dictionary
# Structure: {query_text: {'relevant': [hotel_ids], 'non_relevant': [hotel_ids]}}
user_feedback = {}

# Load existing feedback if available
if os.path.exists(feedback_file_path):
    try:
        with open(feedback_file_path, 'rb') as f:
            user_feedback = pickle.load(f)
    except Exception as e:
        print(f"Error loading feedback file: {str(e)}")

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

# Modified Rocchio algorithm to incorporate user feedback
def apply_rocchio(query_vector, doc_vectors, similarities, query_text, alpha=1.0, beta=0.75, gamma=0.15, top_k=5):
    """
    Apply the Rocchio algorithm for query modification using relevance feedback.
    
    Parameters:
    - query_vector: Original query vector
    - doc_vectors: Document vectors matrix
    - similarities: Current similarity scores
    - query_text: The original text query (used to look up user feedback)
    - alpha: Weight for original query (typically 1.0)
    - beta: Weight for relevant documents (typically 0.75)
    - gamma: Weight for non-relevant documents (typically 0.15)
    - top_k: Number of top documents to consider if no explicit feedback
    
    Returns:
    - Modified query vector
    """
    modified_query = query_vector.copy()
    
    # Check if we have explicit feedback for this query
    if query_text in user_feedback:
        feedback = user_feedback[query_text]
        
        # Get relevant document indices
        relevant_indices = feedback.get('relevant', [])
        if relevant_indices:
            relevant_mean = np.mean(doc_vectors[relevant_indices], axis=0)
            modified_query += beta * relevant_mean
        
        # Get non-relevant document indices
        non_relevant_indices = feedback.get('non_relevant', [])
        if non_relevant_indices:
            non_relevant_mean = np.mean(doc_vectors[non_relevant_indices], axis=0)
            modified_query -= gamma * non_relevant_mean
    else:
        # If no explicit feedback, use top-k documents as pseudo-relevant
        top_indices = np.argsort(similarities)[-top_k:]
        if len(top_indices) > 0:
            relevant_mean = np.mean(doc_vectors[top_indices], axis=0)
            modified_query = alpha * query_vector + beta * relevant_mean
    
    return modified_query

def json_search(query, user_lat=None, user_lon=None, unit="km", sort_order="default", top_n=10, session_id=None):
    global hotels_df, vectorizer, svd_model, doc_vectors

    if doc_vectors is None or not query:
        return '[]'

    try:
        processed_query = preprocess_text(query)
        query_tfidf = vectorizer.transform([processed_query])
        query_vector = svd_model.transform(query_tfidf)[0]

        initial_similarities = cosine_similarity(query_vector, doc_vectors)

        # Apply Rocchio algorithm using feedback
        modified_query_vector = apply_rocchio(query_vector, doc_vectors, initial_similarities, query)

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
        results_df['hotel_index'] = results_df.index  # Store original index for feedback

        results_df = results_df.sort_values('similarity_score', ascending=False)
        top_results = results_df.head(top_n)
        if sort_order == 'rating':
            rating_map = {
                'OneStar': 1,
                'TwoStar': 2,
                'ThreeStar': 3,
                'FourStar': 4,
                'All': 5
            }
            top_results = top_results.copy()
            top_results['numeric_rating'] = top_results['HotelRating'].map(rating_map).fillna(0)
            top_results = top_results.sort_values('numeric_rating', ascending=False)

        elif sort_order in ['asc', 'desc'] and user_lat is not None and user_lon is not None:
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
                lambda url: f"{url}" if isinstance(url, str) else ""
            )

        columns_to_include = ['HotelName', 'Description', 'HotelFacilities', 'cityName', 'countyName',
                             'similarity_score', 'HotelRating', 'hotel_index']
        if 'imageSearchLink' in top_results.columns:
            columns_to_include.append('imageSearchLink')
            
        # Store query for session if provided
        if session_id:
            session_queries[session_id] = query
            
        return top_results[columns_to_include].to_json(orient='records')

    except Exception as e:
        print(f"Error in json_search: {str(e)}")
        return json.dumps({"error": str(e)})

# Keep track of sessions and their queries
session_queries = {}

app = Flask(__name__)
CORS(app)
initialize_hotel_tokens()

@app.route("/")
def home():
    return render_template('base.html', title="TripTune - Hotel Finder")

@app.route("/hotels", methods=['GET'])
def hotels_search():
    app.logger.info("Processing search request")
    text = request.args.get("query", "")
    user_lat = request.args.get("lat", type=float)
    user_lon = request.args.get("lon", type=float)
    unit = request.args.get("unit", default="km")
    sort_order = request.args.get("sort", default="default")
    session_id = request.args.get("session_id", None)
    
    return json_search(text, user_lat=user_lat, user_lon=user_lon, unit=unit, sort_order=sort_order, session_id=session_id)

@app.route("/feedback", methods=['POST'])
def record_feedback():
    """Handle user feedback on hotel results"""
    global user_feedback
    
    try:
        data = request.json
        query = data.get('query')
        hotel_index = data.get('hotel_index')
        is_relevant = data.get('is_relevant')  # True for thumbs up, False for thumbs down
        
        if not query or hotel_index is None or is_relevant is None:
            return jsonify({"error": "Missing required data"}), 400
        
        # Convert hotel_index to integer
        hotel_index = int(hotel_index)
        
        # Initialize feedback for this query if not exists
        if query not in user_feedback:
            user_feedback[query] = {'relevant': [], 'non_relevant': []}
        
        # Add feedback based on relevance
        if is_relevant:
            if hotel_index not in user_feedback[query]['relevant']:
                user_feedback[query]['relevant'].append(hotel_index)
            # Remove from non-relevant if it was there
            if hotel_index in user_feedback[query]['non_relevant']:
                user_feedback[query]['non_relevant'].remove(hotel_index)
        else:
            if hotel_index not in user_feedback[query]['non_relevant']:
                user_feedback[query]['non_relevant'].append(hotel_index)
            # Remove from relevant if it was there
            if hotel_index in user_feedback[query]['relevant']:
                user_feedback[query]['relevant'].remove(hotel_index)
        
        # Save feedback to file
        with open(feedback_file_path, 'wb') as f:
            pickle.dump(user_feedback, f)
        
        return jsonify({"success": True, "message": "Feedback recorded"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/episodes", methods=['GET'])
def episodes_search():
    text = request.args.get("query", "")
    return json_search(text)

@app.route("/ping", methods=['GET'])
def ping():
    app.logger.info("pong")
    return "pong"

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5001)