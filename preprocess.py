import pandas as pd

def create_engine_data():
    # 1. Load the unzipped CSVs
    items = pd.read_csv('data/olist_order_items_dataset.csv')
    products = pd.read_csv('data/olist_products_dataset.csv')
    reviews = pd.read_csv('data/olist_order_reviews_dataset.csv')

    # 2. Merge: Link Items to their Categories
    # Result: Each sale now has a Category name
    df = pd.merge(items[['order_id', 'product_id', 'price']], 
                  products[['product_id', 'product_category_name']], 
                  on='product_id')

    # 3. Merge: Link Sales to their Ratings
    # Result: Each sale now has a 1-5 star score
    df = pd.merge(df, reviews[['order_id', 'review_score']], on='order_id')

    # 4. Cleaning: Fill missing categories and drop duplicates
    df['product_category_name'] = df['product_category_name'].fillna('General')
    df = df.drop_duplicates()

    # 5. Aggregate: Create the "Intelligence Table"
    # This gives us the average rating and price for every single product
    master_data = df.groupby('product_id').agg({
        'product_category_name': 'first',
        'price': 'mean',
        'review_score': 'mean'
    }).reset_index()

    # Save for your App
    master_data.to_csv('cleaned_engine_data.csv', index=False)
    print("✅ Success! 'cleaned_engine_data.csv' is ready for the engine.")

create_engine_data()