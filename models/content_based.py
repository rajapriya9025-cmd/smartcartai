# content_based.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity



# -----------------------------
# 📁 LOAD DATA
# -----------------------------
products = pd.read_csv(r"D:\smart_cart_ai\data\products.csv")

# -----------------------------
# 🧹 DATA CLEANING
# -----------------------------

# Convert product_id to string (avoid mismatch errors)
products["product_id"] = products["product_id"].astype(str)

# Fill missing values
products["product_name"] = products["product_name"].fillna('')
products["category"] = products["category"].fillna('')
products["description"] = products["description"].fillna('')
products["brand"] = products["brand"].fillna('')

# -----------------------------
# 🧠 FEATURE ENGINEERING
# -----------------------------

# Combine text features (improves recommendation quality)
products["features"] = (
    products["product_name"] + " " +
    products["category"] + " " +
    products["brand"] + " " +
    products["description"]
)

# -----------------------------
# 🔢 TF-IDF + COSINE SIMILARITY
# -----------------------------
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(products["features"])

from sklearn.metrics.pairwise import cosine_similarity

def recommend_similar(product_id, tfidf_matrix, product_index_map, top_n=10):
    # Get vector for the product
    idx = product_index_map[product_id]
    product_vec = tfidf_matrix[idx]

    # Compute similarity only with this vector
    sim_scores = cosine_similarity(product_vec, tfidf_matrix).flatten()

    # Get top-N similar products
    similar_indices = sim_scores.argsort()[::-1][1:top_n+1]
    return [list(product_index_map.keys())[i] for i in similar_indices]


# -----------------------------
# 🎯 RECOMMENDATION FUNCTION
# -----------------------------
def recommend_similar(product_id, top_n=5):
    
    # Convert input to string
    product_id = str(product_id)

    # Check if product exists
    matches = products[products["product_id"] == product_id]

    if matches.empty:
        print("❌ Product ID not found:", product_id)
        return []

    # Get index safely
    idx = matches.index[0]

    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort by similarity
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get top similar items (skip itself)
    sim_indices = [i[0] for i in sim_scores[1:top_n+1]]

    # Return product IDs
    return products.iloc[sim_indices]["product_id"].tolist()

# -----------------------------
# 🧪 TEST FUNCTION
# -----------------------------
if __name__ == "__main__":
    
    # Take a valid product_id from dataset
    sample_id = products["product_id"].iloc[0]

    print("Testing Product ID:", sample_id)
    
    recommendations = recommend_similar(sample_id)

    print("Recommended Product IDs:", recommendations)