import json
import os
import re
import numpy as np
from collections import Counter, defaultdict
from flask import Flask, render_template, request
from flask_cors import CORS
import pandas as pd

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

def json_search(query, top_n=10):
    if not hotel_tokens or not query:
        return '[]'
    try:
        # Process the query
        processed_query = preprocess_text(query)
        query_tokens = tokenize(processed_query)
        
        # Calculate Jaccard similarity between query and each hotel
        similarity_scores = []
        for idx, hotel_token_set in enumerate(hotel_tokens):
            score = jaccard_similarity(query_tokens, hotel_token_set)
            similarity_scores.append(score)
        
        # Add similarity scores to results dataframe
        results_df = hotels_df.copy()
        results_df['similarity_score'] = similarity_scores
        results_df = results_df.sort_values('similarity_score', ascending=False)
        top_results = results_df.head(top_n)
        
        columns_to_include = ['HotelName', 'similarity_score']
        for col in ['Description', 'HotelFacilities', 'cityName', 'HotelRating']:
            if col in top_results.columns:
                columns_to_include.append(col)
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
    return json_search(text)

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