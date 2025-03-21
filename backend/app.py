import json
import os
import re
import numpy as np
import math
from collections import Counter
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd

os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))

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

with open(json_file_path, 'r') as file:
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
    words = [word for word in words if word not in STOPWORDS and len(word) > 1]
    return words

vocabulary = {}
idf_values = {}
hotel_vectors = None

def initialize_vectorization():
    global vocabulary, idf_values, hotel_vectors
    hotels_df['combined_text'] = ''
    if 'Description' in hotels_df.columns:
        hotels_df['combined_text'] += hotels_df['Description'].fillna('').apply(preprocess_text)
    if 'HotelFacilities' in hotels_df.columns:
        hotels_df['combined_text'] += ' ' + hotels_df['HotelFacilities'].fillna('').apply(preprocess_text)
    if 'Attractions' in hotels_df.columns:
        hotels_df['combined_text'] += ' ' + hotels_df['Attractions'].fillna('').apply(preprocess_text)
    
    tokenized_docs = [tokenize(doc) for doc in hotels_df['combined_text']]
    doc_freq = Counter()
    for doc_tokens in tokenized_docs:
        unique_tokens = set(doc_tokens)
        for token in unique_tokens:
            doc_freq[token] += 1
    
    max_features = 5000
    doc_count = len(hotels_df)
    sorted_terms = sorted(doc_freq.items(), key=lambda x: x[1], reverse=True)
    if max_features > 0 and max_features < len(sorted_terms):
        sorted_terms = sorted_terms[:max_features]
    vocabulary = {term: idx for idx, (term, _) in enumerate(sorted_terms)}
    
    idf_values = {}
    for term, df in doc_freq.items():
        if term in vocabulary:
            idf_values[term] = math.log((doc_count + 1) / (df + 1)) + 1
    hotel_vectors = np.zeros((len(hotels_df), len(vocabulary)))
    
    for doc_idx, doc_tokens in enumerate(tokenized_docs):
        term_freq = Counter(doc_tokens)
        doc_length = len(doc_tokens)
        
        if doc_length == 0:
            continue
        for term, freq in term_freq.items():
            if term in vocabulary:
                term_idx = vocabulary[term]
                tf = freq / doc_length
                hotel_vectors[doc_idx, term_idx] = tf * idf_values.get(term, 0)
    
    print(f"Initialized vectors for {len(hotels_df)} hotels with {len(vocabulary)} features")

def create_query_vector(query_text):
    processed_query = preprocess_text(query_text)
    query_tokens = tokenize(processed_query)
    term_freq = Counter(query_tokens)
    query_length = len(query_tokens)
    
    query_vector = np.zeros(len(vocabulary))
    if query_length == 0:
        return query_vector
    for term, freq in term_freq.items():
        if term in vocabulary:
            term_idx = vocabulary[term]
            tf = freq / query_length
            query_vector[term_idx] = tf * idf_values.get(term, 0)

    return query_vector

def compute_cosine_similarity(vec_a, matrix_b):
    dot_products = np.dot(matrix_b, vec_a)
    magnitude_a = np.sqrt(np.sum(vec_a**2))
    magnitudes_b = np.sqrt(np.sum(matrix_b**2, axis=1))
    
    magnitudes_b[magnitudes_b == 0] = 1
    if magnitude_a == 0:
        magnitude_a = 1
    similarities = dot_products / (magnitudes_b * magnitude_a)
    
    return similarities

def json_search(query, top_n=10):
    if hotel_vectors is None or not query:
        return '[]'
    
    try:
        query_vector = create_query_vector(query)
        similarity_scores = compute_cosine_similarity(query_vector, hotel_vectors)
        results_df = hotels_df.copy()
        results_df['similarity_score'] = similarity_scores
        results_df = results_df.sort_values('similarity_score', ascending=False)
        top_results = results_df.head(top_n)
        
        columns_to_include = ['HotelName', 'similarity_score']
        for col in ['Description', 'HotelFacilities', 'cityName', 'HotelRating']:
            if col in top_results.columns:
                columns_to_include.append(col)
        matches_filtered = top_results[columns_to_include]
        matches_filtered_json = matches_filtered.to_json(orient='records')
        return matches_filtered_json
        
    except Exception as e:
        print(f"Error in json_search: {str(e)}")
        return json.dumps({"error": str(e)})

app = Flask(__name__)
CORS(app)

# @app.before_first_request
# def before_first_request():
#     initialize_vectorization()

@app.route("/")
def home():
    return render_template('base.html', title="sample html")

@app.route("/hotels")
def hotels_search():
    text = request.args.get("query", "")
    return json_search(text)

@app.route("/episodes")
def episodes_search():
    text = request.args.get("query", "")
    return json_search(text)

if 'DB_NAME' not in os.environ:
    initialize_vectorization()
    app.run(debug=True, host="0.0.0.0", port=5000)
import json
import os
import re
import numpy as np
import math
from collections import Counter
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd

os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))

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
    words = [word for word in words if word not in STOPWORDS and len(word) > 1]
    return words

vocabulary = {}
idf_values = {}
hotel_vectors = None

def initialize_vectorization():
    global vocabulary, idf_values, hotel_vectors
    hotels_df['combined_text'] = ''
    if 'Description' in hotels_df.columns:
        hotels_df['combined_text'] += hotels_df['Description'].fillna('').apply(preprocess_text)
    if 'HotelFacilities' in hotels_df.columns:
        hotels_df['combined_text'] += ' ' + hotels_df['HotelFacilities'].fillna('').apply(preprocess_text)
    if 'Attractions' in hotels_df.columns:
        hotels_df['combined_text'] += ' ' + hotels_df['Attractions'].fillna('').apply(preprocess_text)
    
    tokenized_docs = [tokenize(doc) for doc in hotels_df['combined_text']]
    doc_freq = Counter()
    for doc_tokens in tokenized_docs:
        unique_tokens = set(doc_tokens)
        for token in unique_tokens:
            doc_freq[token] += 1
    
    max_features = 5000
    doc_count = len(hotels_df)
    sorted_terms = sorted(doc_freq.items(), key=lambda x: x[1], reverse=True)
    if max_features > 0 and max_features < len(sorted_terms):
        sorted_terms = sorted_terms[:max_features]
    vocabulary = {term: idx for idx, (term, _) in enumerate(sorted_terms)}
    
    idf_values = {}
    for term, df in doc_freq.items():
        if term in vocabulary:
            idf_values[term] = math.log((doc_count + 1) / (df + 1)) + 1
    hotel_vectors = np.zeros((len(hotels_df), len(vocabulary)))
    
    for doc_idx, doc_tokens in enumerate(tokenized_docs):
        term_freq = Counter(doc_tokens)
        doc_length = len(doc_tokens)
        
        if doc_length == 0:
            continue
        for term, freq in term_freq.items():
            if term in vocabulary:
                term_idx = vocabulary[term]
                tf = freq / doc_length
                hotel_vectors[doc_idx, term_idx] = tf * idf_values.get(term, 0)
    
    print(f"Initialized vectors for {len(hotels_df)} hotels with {len(vocabulary)} features")

def create_query_vector(query_text):
    processed_query = preprocess_text(query_text)
    query_tokens = tokenize(processed_query)
    term_freq = Counter(query_tokens)
    query_length = len(query_tokens)
    
    query_vector = np.zeros(len(vocabulary))
    if query_length == 0:
        return query_vector
    for term, freq in term_freq.items():
        if term in vocabulary:
            term_idx = vocabulary[term]
            tf = freq / query_length
            query_vector[term_idx] = tf * idf_values.get(term, 0)

    return query_vector

def compute_cosine_similarity(vec_a, matrix_b):
    dot_products = np.dot(matrix_b, vec_a)
    magnitude_a = np.sqrt(np.sum(vec_a**2))
    magnitudes_b = np.sqrt(np.sum(matrix_b**2, axis=1))
    
    magnitudes_b[magnitudes_b == 0] = 1
    if magnitude_a == 0:
        magnitude_a = 1
    similarities = dot_products / (magnitudes_b * magnitude_a)
    
    return similarities

def json_search(query, top_n=10):
    if hotel_vectors is None or not query:
        return '[]'
    
    try:
        query_vector = create_query_vector(query)
        similarity_scores = compute_cosine_similarity(query_vector, hotel_vectors)
        results_df = hotels_df.copy()
        results_df['similarity_score'] = similarity_scores
        results_df = results_df.sort_values('similarity_score', ascending=False)
        top_results = results_df.head(top_n)
        
        columns_to_include = ['HotelName', 'similarity_score']
        for col in ['Description', 'HotelFacilities', 'cityName', 'HotelRating']:
            if col in top_results.columns:
                columns_to_include.append(col)
        matches_filtered = top_results[columns_to_include]
        matches_filtered_json = matches_filtered.to_json(orient='records')
        return matches_filtered_json
        
    except Exception as e:
        print(f"Error in json_search: {str(e)}")
        return json.dumps({"error": str(e)})

app = Flask(__name__)
CORS(app)

# @app.before_first_request
# def before_first_request():
#     initialize_vectorization()

@app.route("/")
def home():
    return render_template('base.html', title="sample html")

@app.route('/hotels', methods=['GET'])
def hotels_search():
    initialize_vectorization()
    text = request.args.get("query", "")
    return json_search(text)

@app.route("/episodes")
def episodes_search():
    text = request.args.get("query", "")
    return json_search(text)

@app.route('/ping', methods=['GET'])
def ping():
    return "pong"

if 'DB_NAME' not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)