from flask import Flask, request, jsonify
from pymongo import MongoClient
from bson import ObjectId
import pandas as pd
import joblib
import os
import numpy as np

from recommendation_model import recommend_for_user
from smart_search_model import smart_search_and_rank

app = Flask(__name__)

# ==============================
# LOAD MODEL FILES
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ENCODER_PATH = os.path.join(BASE_DIR, "encoder.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

encoder = joblib.load(ENCODER_PATH)
scaler = joblib.load(SCALER_PATH)

# ==============================
# CONNECT TO MONGODB ATLAS
# ==============================

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "lmgtech"

if not MONGO_URI:
    raise ValueError("MONGO_URI environment variable not set")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

properties_collection = db["assets"]
users_collection = db["users"]

# ==============================
# HELPER FUNCTIONS
# ==============================

def serialize_doc(doc):
    if isinstance(doc, list):
        return [serialize_doc(item) for item in doc]
    if isinstance(doc, dict):
        return {k: serialize_doc(v) for k, v in doc.items()}
    if isinstance(doc, ObjectId):
        return str(doc)
    return doc

# ==============================
# TEST ROUTE
# ==============================

@app.route("/")
def home():
    return "AI Service Running Successfully"

# ==============================
# DEBUG ROUTE (see real user IDs)
# ==============================

@app.route("/debug-users")
def debug_users():
    users = list(users_collection.find({}, {"_id": 1}).limit(5))
    return jsonify([str(u["_id"]) for u in users])


@app.route("/debug-collections")
def debug_collections():
    return jsonify(db.list_collection_names())

@app.route("/debug-assets")
def debug_assets():
    assets = list(properties_collection.find().limit(5))
    return jsonify(serialize_doc(assets))
# ==============================
# RECOMMENDATION ROUTE
# ==============================

@app.route("/recommend/<user_id>", methods=["GET"])
def recommend(user_id):

    # validate ObjectId format
    if not ObjectId.is_valid(user_id):
        return jsonify({"error": "Invalid user ID format"}), 400

    user = users_collection.find_one({"_id": ObjectId(user_id)})

    if user is None:
        return jsonify({"error": "User not found"}), 404

    # fetch properties
    properties = list(properties_collection.find({}, {"_id": 0}))
    df = pd.DataFrame(properties)

    if df.empty:
        return jsonify({"error": "No properties found in database"}), 404

    categorical_cols = ['category', 'location', 'condition']
    numerical_cols = ['price_per_day', 'popularity']

    # Ensure all required columns exist in DataFrame to prevent KeyError
    for col in categorical_cols:
        if col not in df.columns:
            df[col] = "Good"  # Default value for missing categorical data
    for col in numerical_cols:
        if col not in df.columns:
            df[col] = 0.0     # Default value for missing numerical data

    # Precompute property vectors for similarity search
    prop_cat = encoder.transform(df[categorical_cols])
    prop_num = scaler.transform(df[numerical_cols])
    property_vectors = np.hstack([prop_cat, prop_num])

    # run recommendation model
    results = recommend_for_user(
        user,
        df,
        encoder,
        scaler,
        property_vectors,
        categorical_cols,
        numerical_cols,
        top_n=5
    )

    serialized_results = serialize_doc(results.to_dict(orient="records"))
    return jsonify(serialized_results)

# ==============================
# SMART SEARCH ROUTE
# ==============================

@app.route("/smart-search", methods=["POST"])
def smart_search():

    query = request.json or {}

    properties = list(properties_collection.find({}, {"_id": 0}))
    df = pd.DataFrame(properties)

    if df.empty:
        return jsonify({"error": "No properties found"}), 404

    # Ensure required columns exist to prevent KeyErrors
    required_categorical = ['category', 'location', 'condition']
    required_numerical = ['price_per_day', 'popularity']

    for col in required_categorical:
        if col not in df.columns:
            df[col] = "Unknown"
    for col in required_numerical:
        if col not in df.columns:
            df[col] = 0.0

    results = smart_search_and_rank(df, query, top_n=5)

    serialized_results = serialize_doc(results.to_dict(orient="records"))
    return jsonify(serialized_results)

# ==============================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)