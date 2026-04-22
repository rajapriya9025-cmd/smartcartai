import pandas as pd

# Load cleaned dataset
df = pd.read_csv(r"D:\smart_cart_ai\data\cleaned_data.csv")

print("Loaded data:", df.shape)

# -----------------------------
# 🛍️ PRODUCTS TABLE
# -----------------------------
products = df[[
    "product_name",
    "category",
    "description",
    "image_url",
    "brand",
    "price"
]].drop_duplicates()

# Create product_id
products["product_id"] = range(1, len(products) + 1)

# Reorder columns
products = products[[
    "product_id",
    "product_name",
    "category",
    "description",
    "image_url",
    "brand",
    "price"
]]

print("Products:", products.shape)


# -----------------------------
# 👤 USERS TABLE
# -----------------------------
users = df[["user_id"]].drop_duplicates()

# Add simulated features (important for your UI)
users["location"] = "India"
users["preference"] = "General"
users["password"]="riya@123"

print("Users:", users.shape)


# -----------------------------
# 🔄 INTERACTIONS TABLE
# -----------------------------

# Merge product_id into original df
df = df.merge(products[["product_id", "product_name"]], on="product_name", how="left")

interactions = df[[
    "user_id",
    "product_id",
    "rating",
    "timestamp"
]]

print("Interactions:", interactions.shape)


# -----------------------------
# 💾 SAVE FILES
# -----------------------------
products.to_csv("products.csv", index=False)
users.to_csv("users.csv", index=False)
interactions.to_csv("interactions.csv", index=False)

print("✅ All files created successfully!")