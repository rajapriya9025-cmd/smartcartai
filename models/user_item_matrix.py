import pandas as pd

# -----------------------------
# 📥 LOAD DATA (use full path)
# -----------------------------
interactions = pd.read_csv(r"D:\smart_cart_ai\data\interactions.csv")

print("✅ Data Loaded!")

# -----------------------------
# 🔧 FIX RATING TYPE (IMPORTANT)
# -----------------------------
interactions["rating"] = pd.to_numeric(interactions["rating"], errors="coerce")
interactions = interactions.dropna(subset=["rating"])

print("✅ Ratings converted to numeric!")

# -----------------------------
# 🧠 CREATE USER-ITEM MATRIX
# -----------------------------
user_item_matrix = interactions.pivot_table(
    index="user_id",
    columns="product_id",
    values="rating"
).fillna(0)

# -----------------------------
# 📊 PRINT OUTPUT (IMPORTANT)
# -----------------------------
print("\n📊 USER-ITEM MATRIX SHAPE:")
print(user_item_matrix.shape)

print("\n👤 SAMPLE USERS:")
print(user_item_matrix.index[:5])

print("\n🛍️ SAMPLE PRODUCTS:")
print(user_item_matrix.columns[:5])

print("\n🔍 SAMPLE MATRIX (FIRST 5 ROWS):")
print(user_item_matrix.head())

print("\n✅ Matrix created successfully!")