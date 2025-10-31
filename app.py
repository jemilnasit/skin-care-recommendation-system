import streamlit as st
import pandas as pd
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

# ================================
# 1. Load Data & Models
# ================================
@st.cache_data
def load_data():
    df = pd.read_csv("product_info_processed.csv")
    return df

@st.cache_resource
def load_embeddings_and_index():
    # Load embeddings
    with open("product_embeddings.pkl", "rb") as f:
        product_embeddings = pickle.load(f)

    # Load FAISS index
    index = faiss.read_index("product_faiss.index")

    # Load model for queries
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return product_embeddings, index, model

df = load_data()
product_embeddings, index, model = load_embeddings_and_index()

st.title("💄 Skincare Product Recommender")
st.write("Search for products or browse with Query.")

# ================================
# 2. Sidebar Filters
# ================================
st.sidebar.header("🔍 Filters")

# Price range filter
if "price_usd" in df.columns:
    min_price, max_price = float(df["price_usd"].min()), float(df["price_usd"].max())
    price_range = st.sidebar.slider("Price Range (USD)", min_price, max_price, (min_price, max_price))
else:
    price_range = (None, None)

# Rating filter
if "rating" in df.columns:
    min_rating, max_rating = float(df["rating"].min()), float(df["rating"].max())
    rating_filter = st.sidebar.slider("Minimum Rating", min_rating, max_rating, min_rating)
else:
    rating_filter = None

# Category filter
category_filter = None
if "primary_category" in df.columns:
    categories = df["primary_category"].dropna().unique().tolist()
    category_filter = st.sidebar.multiselect("Select Category", categories)

# Ingredient keyword filter
ingredient_keyword = st.sidebar.text_input("Ingredient Keyword")

# ================================
# 3. Search Box
# ================================
query = st.text_input("Enter a skincare need (or leave empty to browse):", "")

# ================================
# 4. Recommendation Logic
# ================================
def recommend_products(query, top_n=5):
    """Returns top N recommended products based on query."""
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_n)
    results = df.iloc[indices[0]].copy()
    results["distance"] = distances[0]
    display_cols = [c for c in ['product_name','price_usd','rating','distance','primary_category'] if c in results.columns]
    return results[display_cols]

def apply_filters(df):
    """Apply sidebar filters to DataFrame."""
    filtered = df.copy()

    if price_range[0] is not None:
        filtered = filtered[(filtered["price_usd"] >= price_range[0]) & (filtered["price_usd"] <= price_range[1])]

    if rating_filter is not None and "rating" in filtered.columns:
        filtered = filtered[filtered["rating"] >= rating_filter]

    if category_filter:
        filtered = filtered[filtered["primary_category"].isin(category_filter)]

    if ingredient_keyword and "ingredients" in filtered.columns:
        filtered = filtered[filtered["ingredients"].str.contains(ingredient_keyword, case=False, na=False)]

    return filtered

# ================================
# 5. Show Results
# ================================
if query.strip():  # If user entered a query
    st.subheader(f"🔹 Top 5 Recommendations for: {query}")
    results = recommend_products(query, top_n=5)
    st.dataframe(results)
else:
    st.subheader("🛍 Browse Products with Query")
    filtered_df = apply_filters(df)
    st.write(f"Showing {len(filtered_df)} products after filtering:")
    st.dataframe(filtered_df[['product_name','price_usd','rating','primary_category']] if len(filtered_df)>0 else filtered_df)