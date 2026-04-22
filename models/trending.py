import pandas as pd

# -----------------------------
# 📥 LOAD DATA (USE FULL PATH)
# -----------------------------
interactions = pd.read_csv(r"D:\smart_cart_ai\data\interactions.csv")
products = pd.read_csv(r"D:\smart_cart_ai\data\products.csv")

print("✅ Data Loaded!")

# -----------------------------
# 🔧 FIX RATING TYPE
# -----------------------------
interactions["rating"] = pd.to_numeric(interactions["rating"], errors="coerce")
interactions = interactions.dropna(subset=["rating"])

print("✅ Ratings cleaned!")

# -----------------------------
# 🔥 TRENDING FUNCTION
# -----------------------------
def get_trending_products(top_n=5):
    trending = interactions.groupby("product_id")["rating"].count()
    trending = trending.sort_values(ascending=False).head(top_n)

    product_ids = list(trending.index)

    result = products[
        products["product_id"].isin(product_ids)
    ][["product_id", "product_name", "price"]]

    return result

# -----------------------------
# ✅ TEST RUN (IMPORTANT)
# -----------------------------
if __name__ == "__main__":
    print("\n🔥 TOP TRENDING PRODUCTS:\n")
    print(get_trending_products(5))