import json
import os
import re
import numpy as np
from collections import Counter, defaultdict
from flask import Flask, render_template, request
from flask_cors import CORS
import pandas as pd
from math import radians, cos, sin, asin, sqrt

# Set up ROOT_PATH relative to the project
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))
current_directory = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(current_directory, 'init.json')

# Define stopwords (you can adjust as needed)
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

# Load hotel data
with open(json_file_path, 'r', encoding='utf-8') as file:
    hotel_data = json.load(file)
    hotels_df = pd.DataFrame(hotel_data)

# Preprocessing functions
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

# Global variables for hotel tokens
hotel_tokens = []

def initialize_hotel_tokens():
    global hotel_tokens, hotels_df
    # Create a combined text column from different fields
    hotels_df['combined_text'] = ''
    if 'Description' in hotels_df.columns:
        hotels_df['combined_text'] += hotels_df['Description'].fillna('').apply(preprocess_text)
    if 'HotelFacilities' in hotels_df.columns:
        hotels_df['combined_text'] += ' ' + hotels_df['HotelFacilities'].fillna('').apply(preprocess_text)
    if 'Attractions' in hotels_df.columns:
        hotels_df['combined_text'] += ' ' + hotels_df['Attractions'].fillna('').apply(preprocess_text)
    
    # Tokenize each hotel's text into a set of unique tokens
    hotel_tokens = [tokenize(doc) for doc in hotels_df['combined_text']]
    
    print(f"Tokenized {len(hotels_df)} hotels")

def jaccard_similarity(set_a, set_b):
    """Calculate Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    
    if union == 0:
        return 0.0
        
    return intersection / union

def haversine(lat1, lon1, lat2, lon2):
    # Earth radius in kilometers
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

def json_search(query, user_lat=None, user_lon=None, unit="km", sort_order="default", top_n=10):
    global hotels_df

    if not hotel_tokens or not query:
        return '[]'

    try:
        processed_query = preprocess_text(query)
        query_tokens = tokenize(processed_query)

        similarity_scores = []
        distances = []

        for idx, hotel_token_set in enumerate(hotel_tokens):
            score = jaccard_similarity(query_tokens, hotel_token_set)
            similarity_scores.append(score)

            if user_lat is not None and user_lon is not None:
                try:
                    map_val = hotels_df.iloc[idx].get('Map', '')
                    lat_str, lon_str = map_val.split('|')
                    hotel_lat = float(lat_str)
                    hotel_lon = float(lon_str)
                    dist_km = haversine(user_lat, user_lon, hotel_lat, hotel_lon)
                    dist = dist_km if unit == "km" else dist_km * 0.621371
                    distances.append(round(dist, 1))
                except:
                    distances.append(None)

        results_df = hotels_df.copy()
        results_df['similarity_score'] = similarity_scores
        results_df['distance_km'] = distances

        results_df = results_df.sort_values('similarity_score', ascending=False)
        top_results = results_df.head(top_n)
        if sort_order in ['asc', 'desc'] and user_lat is not None and user_lon is not None:
            top_results = top_results.sort_values('distance_km', ascending=(sort_order == 'asc'))

        # Append distance info to description
        if user_lat is not None and user_lon is not None:
            def add_distance(row):
                desc = row.get('Description', '')
                if row['distance_km'] is not None:
                    return f"{desc} ({row['distance_km']} {'km' if unit == 'km' else 'mi'} away from you)"                
                return desc
            top_results['Description'] = top_results.apply(add_distance, axis=1)
        # Add hotel images
        if 'HotelWebsiteUrl' in top_results.columns:
            top_results['imageSearchLink'] = top_results['HotelWebsiteUrl'].apply(
                lambda url: f"https://www.google.com/search?tbm=isch&q={url}" if isinstance(url, str) else ""
            )
        columns_to_include = ['HotelName', 'similarity_score']
        if 'imageSearchLink' in top_results.columns:
          columns_to_include.append('imageSearchLink')
        return top_results[columns_to_include].to_json(orient='records')

    except Exception as e:
        print(f"Error in json_search: {str(e)}")
        return json.dumps({"error": str(e)})

# Create Flask app
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