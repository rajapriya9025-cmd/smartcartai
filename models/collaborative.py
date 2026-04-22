import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

def build_user_item_matrix(interactions):
    # Build sparse matrix instead of dense pivot
    user_item = interactions.pivot_table(
        index='user_id',
        columns='product_id',
        values='rating',
        fill_value=0
    )

    # Convert to sparse
    sparse_matrix = csr_matrix(user_item.values)
    return user_item, sparse_matrix

def recommend_svd(interactions, user_id, products, n_components=50, top_n=10):
    user_item, sparse_matrix = build_user_item_matrix(interactions)

    # Fit SVD on sparse matrix
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced = svd.fit_transform(sparse_matrix)

    # Reconstruct approximate ratings
    reconstructed = np.dot(reduced, svd.components_)

    # Map user index
    try:
        user_idx = user_item.index.get_loc(user_id)
    except KeyError:
        return []

    scores = reconstructed[user_idx]
    top_items = np.argsort(scores)[::-1][:top_n]

    recs = []
    for pid in user_item.columns[top_items]:
        product = products.loc[products['product_id'] == pid].to_dict('records')
        if product:
            recs.append(product[0])
    return recs
