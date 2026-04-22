from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import os, re, ast

app = Flask(__name__)
app.secret_key = 'smartcartai_secret_2024'

USD_TO_INR = 84.0

# ─── DATA HELPERS ─────────────────────────────────────────────────────────────

def parse_first_image(raw):
    if not raw or pd.isna(raw): return ''
    s = str(raw).strip()
    if s in ('[]','','nan'): return ''
    try:
        result = ast.literal_eval(s)
        if isinstance(result, list):
            for item in result:
                url = str(item).strip()
                if url.startswith('http'): return url
    except: pass
    m = re.search(r'https?://[^\s\'">,\]]+', s)
    return m.group(0) if m else ''

def upgrade_image(url):
    """Upgrade Amazon thumbnail (_SS40_ 40px) to high-res (_SX500_ 500px)"""
    if not url: return ''
    return re.sub(r'\._[A-Z0-9_,]+_\.', '._SX500_.', url)

def parse_price_inr(raw):
    if not raw or pd.isna(raw): return None
    s = str(raw).strip()
    m = re.match(r'^\$?([\d,]+\.?\d*)\s*(?:-\s*\$?[\d,]+\.?\d*)?$', s)
    if m:
        try: return round(float(m.group(1).replace(',','')) * USD_TO_INR, 2)
        except: return None
    return None

def parse_description(raw):
    if not raw or pd.isna(raw): return ''
    s = str(raw).strip()
    try:
        result = ast.literal_eval(s)
        if isinstance(result, list):
            return ' '.join(str(x).strip() for x in result if str(x).strip())[:800]
        return str(result)[:800]
    except:
        clean = re.sub(r'<[^>]+>', ' ', s)
        return re.sub(r'\s+', ' ', clean).strip()[:800]

def clean_category(c):
    if pd.isna(c): return ''
    s = str(c).strip()
    if s.startswith("['http") or s.startswith('.a-') or s.startswith('\n') or s.startswith('<'): return ''
    return s

# ─── LOAD & CLEAN DATA ────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Try clean CSVs first, fall back to raw
def load_csv(clean_name, raw_name):
    clean_path = os.path.join(DATA_DIR, clean_name)
    raw_path   = os.path.join(DATA_DIR, raw_name)
    if os.path.exists(clean_path):
        return pd.read_csv(clean_path)
    return pd.read_csv(raw_path)

products     = load_csv('products_clean.csv', 'products.csv')
users        = load_csv('users_clean.csv', 'users.csv')
interactions = load_csv('interactions_clean.csv', 'interactions.csv')

# Clean products
products['product_id']   = products['product_id'].astype(str)
products['product_name'] = products['product_name'].fillna('').astype(str).str.strip()
products['brand']        = products['brand'].fillna('').astype(str).str.strip()
products['description']  = products['description'].apply(parse_description)
products['category']     = products['category'].apply(clean_category)
products['image_url']    = products['image_url'].apply(parse_first_image).apply(upgrade_image)
products['price']        = products['price'].apply(parse_price_inr)
products = products.dropna(subset=['price'])
products = products[products['product_name'].str.len() > 0]
products = products[products['category'].str.len() > 0]
products['price'] = products['price'].round(2)

# Clean users
users['user_id']    = users['user_id'].astype(str).str.strip()
users['password']   = users['password'].fillna('').astype(str).str.strip()
users['preference'] = users['preference'].fillna('General').astype(str).str.strip()
users = users.drop_duplicates(subset=['user_id'])

# Clean interactions
interactions['user_id']    = interactions['user_id'].astype(str).str.strip()
interactions['product_id'] = interactions['product_id'].astype(str).str.strip()
interactions['rating']     = pd.to_numeric(interactions['rating'], errors='coerce')
interactions               = interactions.dropna(subset=['rating'])

# ─── CATEGORIES ───────────────────────────────────────────────────────────────
# Use ACTUAL categories from the dataset with friendly display names
CATEGORY_MAP = {
    'Pet Supplies':          'Pet_Supplies',
    'Food & Grocery':        'Grocery_and_Gourmet_Food|Prime_Pantry',
    'Office & Tech':         'Office_Products|Cell_Phones|Video_Games|Software|Industrial',
    'Arts, Garden & Music':  'Arts_Crafts|Patio_Lawn|Musical|All_Beauty|Appliances',
}
CATEGORIES = list(CATEGORY_MAP.keys())
CAT_EMOJIS = {
    'Pet Supplies':         '🐾',
    'Food & Grocery':       '🛒',
    'Office & Tech':        '💻',
    'Arts, Garden & Music': '🎨',
}

# ─── TF-IDF ───────────────────────────────────────────────────────────────────
products['features'] = (
    products['product_name'] + ' ' +
    products['category']     + ' ' +
    products['brand']        + ' ' +
    products['description'].str[:300]
)
tfidf          = TfidfVectorizer(stop_words='english', max_features=8000)
tfidf_matrix   = tfidf.fit_transform(products['features'])
prod_id_to_idx = {pid: i for i, pid in enumerate(products['product_id'])}

def content_recommend(product_id, top_n=6):
    pid = str(product_id)
    if pid not in prod_id_to_idx: return []
    idx = prod_id_to_idx[pid]
    sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    top = sim.argsort()[::-1][1:top_n+1]
    return products.iloc[top].to_dict('records')

# ─── SVD ──────────────────────────────────────────────────────────────────────
def collab_recommend(user_id, top_n=10):
    try:
        ui = interactions.pivot_table(
            index='user_id', columns='product_id', values='rating', fill_value=0)
        if user_id not in ui.index: return []
        sparse = csr_matrix(ui.values)
        nc = min(50, sparse.shape[0]-1, sparse.shape[1]-1)
        if nc < 1: return []
        svd = TruncatedSVD(n_components=nc, random_state=42)
        rec = np.dot(svd.fit_transform(sparse), svd.components_)
        scores   = rec[ui.index.get_loc(user_id)]
        top_pids = np.argsort(scores)[::-1][:top_n]
        recs = []
        for pid in ui.columns[top_pids]:
            r = products[products['product_id'] == pid]
            if not r.empty: recs.append(r.iloc[0].to_dict())
        return recs
    except: return []

def get_trending(top_n=10):
    t = interactions.groupby('product_id')['rating'].count().sort_values(ascending=False).head(top_n)
    return products[products['product_id'].isin(t.index)].to_dict('records')

def get_category_products(category, top_n=8):
    pattern = CATEGORY_MAP.get(category, category)
    mask    = products['category'].str.contains(pattern, case=False, na=False)
    subset  = products[mask]
    if subset.empty: subset = products.sample(min(top_n, len(products)))
    else:            subset = subset.sample(min(top_n, len(subset)))
    return subset.to_dict('records')

# ─── SMART SEARCH (returns grouped by category) ───────────────────────────────
def smart_search(query, top_n=40):
    """
    Score products by relevance. Return both:
      - flat list sorted by score (for grid display)
      - grouped by category (for category-grouped view)
    """
    q = query.strip().lower()
    if not q: return [], {}

    df = products.copy()

    def ww(text, word):
        return bool(re.search(r'\b' + re.escape(word) + r'\b', str(text).lower()))

    df['_score'] = 0
    # Whole-word in name → highest priority
    df.loc[df['product_name'].apply(lambda x: ww(x, q)), '_score'] += 5
    # Substring in name
    df.loc[df['product_name'].str.contains(q, case=False, na=False) &
           ~df['product_name'].apply(lambda x: ww(x, q)), '_score'] += 2
    # Brand
    df.loc[df['brand'].apply(lambda x: ww(x, q)), '_score'] += 3
    # Category
    df.loc[df['category'].str.contains(q, case=False, na=False), '_score'] += 2
    # Description
    df.loc[df['description'].str.contains(q, case=False, na=False), '_score'] += 1

    df = df[df['_score'] > 0].sort_values('_score', ascending=False).head(top_n)
    flat = df.to_dict('records')

    # Group by category
    grouped = {}
    for row in flat:
        cat = row.get('category', 'Other')
        if cat not in grouped: grouped[cat] = []
        grouped[cat].append(row)

    return flat, grouped

# ─── ROUTES ───────────────────────────────────────────────────────────────────
@app.route('/')
def index(): return redirect(url_for('login'))

@app.route('/login', methods=['GET','POST'])
def login():
    error = None
    if request.method == 'POST':
        uid = request.form.get('username','').strip()
        pwd = request.form.get('password','').strip()
        row = users[users['user_id'] == uid]
        if not row.empty:
            if str(row.iloc[0].get('password','')) == pwd:
                session['user_id'] = uid
                session['is_new']  = False
                return redirect(url_for('home'))
            error = 'Incorrect password.'
        else:
            error = 'User not found. Please sign up.'
    return render_template('login.html', error=error)

@app.route('/register', methods=['GET','POST'])
def register():
    error = None
    if request.method == 'POST':
        uid   = request.form.get('username','').strip()
        prefs = request.form.getlist('preferences')
        if users[users['user_id'] == uid].empty:
            session['user_id']    = uid
            session['is_new']     = True
            session['preference'] = ','.join(prefs)
            return redirect(url_for('home'))
        error = 'User ID already exists.'
    return render_template('register.html', error=error,
                           categories=CATEGORIES, cat_emojis=CAT_EMOJIS)

@app.route('/home')
def home():
    if 'user_id' not in session: return redirect(url_for('login'))
    uid    = session['user_id']
    is_new = session.get('is_new', False)
    prefs  = session.get('preference','')
    if is_new and prefs:
        recs = get_category_products(prefs.split(',')[0], top_n=8)
    else:
        recs = collab_recommend(uid, top_n=8) or get_trending(top_n=8)
    fresh = products.sample(min(6, len(products))).to_dict('records')
    return render_template('home.html', user_id=uid, recommendations=recs,
                           fresh_picks=fresh, categories=CATEGORIES,
                           cat_emojis=CAT_EMOJIS)

@app.route('/search')
def search():
    if 'user_id' not in session: return redirect(url_for('login'))
    query = request.args.get('q','').strip()
    flat, grouped = smart_search(query, top_n=40) if query else ([], {})
    return render_template('search.html', query=query,
                           results=flat, grouped=grouped,
                           user_id=session['user_id'])

@app.route('/product/<product_id>')
def product_page(product_id):
    if 'user_id' not in session: return redirect(url_for('login'))
    row = products[products['product_id'] == str(product_id)]
    if row.empty: return redirect(url_for('home'))
    product    = row.iloc[0].to_dict()
    cross_sell = content_recommend(product_id, top_n=6)
    return render_template('product.html', product=product,
                           cross_sell=cross_sell, user_id=session['user_id'])

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ─── JSON APIs ────────────────────────────────────────────────────────────────
@app.route('/api/fresh')
def api_fresh():
    picks = products.sample(min(6,len(products)))[
        ['product_id','product_name','image_url','price','category']
    ].to_dict('records')
    return jsonify(picks)

@app.route('/api/category/<cat>')
def api_category(cat):
    items = get_category_products(cat, top_n=8)
    return jsonify([{k:v for k,v in p.items()
        if k in ['product_id','product_name','image_url','price','category','brand']}
        for p in items])

@app.route('/api/search')
def api_search():
    q = request.args.get('q','').strip()
    flat, _ = smart_search(q, top_n=8) if q else ([],{})
    return jsonify([{k:v for k,v in p.items()
        if k in ['product_id','product_name','image_url','price','category','brand']}
        for p in flat])

if __name__ == '__main__':
    app.run(debug=True)