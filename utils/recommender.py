import pandas as pd
import csv
import os

USER_FILE = "data/users.csv"

# -------------------------------
# USER FUNCTIONS
# -------------------------------

def load_users():
    if not os.path.exists(USER_FILE):
        return pd.DataFrame(columns=["user_id", "location", "preference", "password"])
    return pd.read_csv(USER_FILE)


def get_user(user_id):
    users = load_users()
    user = users[users["user_id"] == user_id]
    if user.empty:
        return None
    return user.iloc[0].to_dict()


def user_exists(user_id):
    users = load_users()
    return user_id in users["user_id"].values


def validate_user(user_id, password):
    user = get_user(user_id)
    if not user:
        return False
    return user["password"] == password


def register_user(user_id, location, preference, password):
    file_exists = os.path.isfile(USER_FILE)

    with open(USER_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["user_id", "location", "preference", "password"])
        writer.writerow([user_id, location, preference, password])


# -------------------------------
# PRODUCT FUNCTIONS (SIMPLE)
# -------------------------------

def load_products():
    return pd.read_csv("dataset/products.csv")


def get_all_products():
    return load_products().to_dict(orient="records")


def get_product_by_id(product_id):
    df = load_products()
    p = df[df["product_id"] == int(product_id)]
    if p.empty:
        return None
    return p.iloc[0].to_dict()


def search_products(query):
    df = load_products()
    return df[df["product_name"].str.contains(query, case=False)].to_dict(orient="records")


def get_products_by_category(category, top_n=10):
    df = load_products()
    return df[df["category"] == category].head(top_n).to_dict(orient="records")


def get_trending_products(top_n=10):
    df = load_products()
    return df.sample(n=min(top_n, len(df))).to_dict(orient="records")


# -------------------------------
# RECOMMENDATION (SIMPLE LOGIC)
# -------------------------------

def recommend_by_preference(preference, top_n=8):
    return get_products_by_category(preference, top_n)


def recommend_svd(user_id, top_n=8):
    return get_trending_products(top_n)


def recommend_similar(product_id, top_n=6):
    return get_trending_products(top_n)