import pandas as pd

# Load dataset
df = pd.read_csv(r"D:\smart_cart_ai\data\amazon_reviews.csv")


print("Original Shape:", df.shape)

# Keep only required columns
df = df[[
    "userName",
    "itemName",
    "category",
    "description",
    "image",
    "brand",
    "price",
    "rating",
    "reviewTime"
]]

# Rename columns
df.columns = [
    "user_id",
    "product_name",
    "category",
    "description",
    "image_url",
    "brand",
    "price",
    "rating",
    "timestamp"
]

# Drop missing values
df = df.dropna()

# Reduce dataset size (important)
df = df.head(50000)

print("Cleaned Shape:", df.shape)

# Save cleaned data
df.to_csv("cleaned_data.csv", index=False)

print("✅ cleaned_data.csv created!")