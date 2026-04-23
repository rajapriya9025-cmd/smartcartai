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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, 'data')

# ─── LAZY GLOBALS ─────────────────────────────────────────────────────────────
_products     = None
_users        = None
_interactions = None
_tfidf_matrix = None
_tfidf        = None
_prod_id_idx  = None

# ─── DATA HELPERS ─────────────────────────────────────────────────────────────

def parse_first_image(raw):
    if not raw or pd.isna(raw): return ''
    s = str(raw).strip()
    if s in ('[]', '', 'nan'): return ''
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
    if not url: return ''
    return re.sub(r'\._[A-Z0-9_,]+_\.', '._SX500_.', url)

def parse_price_inr(raw):
    if not raw or pd.isna(raw): return None
    s = str(raw).strip()
    m = re.match(r'^\$?([\d,]+\.?\d*)\s*(?:-\s*\$?[\d,]+\.?\d*)?$', s)
    if m:
        try: return round(float(m.group(1).replace(',', '')) * USD_TO_INR, 2)
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
    if s.startswith("['http") or s.startswith('.a-') or s.startswith('\n') or s.startswith('<'):
        return ''
    return s

# ─── CATEGORIES ───────────────────────────────────────────────────────────────
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

# ─── LAZY DATA LOADER ─────────────────────────────────────────────────────────

def get_data():
    global _products, _users, _interactions
    global _tfidf_matrix, _tfidf, _prod_id_idx

    if _products is not None:
        return _products, _users, _interactions

    # Download files if missing
    import gdown
    files = {
        "data/products.csv":       "1yqbLHWgk6laINBX6QgG8OZfAW2cHfz0q",
        "data/users.csv":          "1iTdUZglcz7T4IDzyxvvF7y1gSJM4IULK",
        "data/interactions.csv":   "1M9SlJdKKcitoroncJ4nur7GGlsOH7F_g",
        "data/cleaned_data.csv":   "16f2n3C-Zrfns20VCzLuYksd1F03yBran",
        "data/amazon_reviews.csv": "10vgH6bfeQQZZV8DTEvEsXgU7wNyIuuAO",
    }
    os.makedirs('data', exist_ok=True)
    for path, fid in files.items():
        if not os.path.exists(path):
            try:
                gdown.download(id=fid, output=path, quiet=False, fuzzy=True)
                print(f"Downloaded: {path}")
            except Exception as e:
                print(f"Warning: Could not download {path}: {e}")

    # Load CSVs
    def load_csv(clean_name, raw_name):
        clean_path = os.path.join(DATA_DIR, clean_name)
        raw_path   = os.path.join(DATA_DIR, raw_name)
        if os.path.exists(clean_path):
            return pd.read_csv(clean_path)
        return pd.read_csv(raw_path)

    p = load_csv('products_clean.csv',     'products.csv')
    u = load_csv('users_clean.csv',        'users.csv')
    i = load_csv('interactions_clean.csv', 'interactions.csv')

    # Clean products
    p['product_id']   = p['product_id'].astype(str)
    p['product_name'] = p['product_name'].fillna('').astype(str).str.strip()
    p['brand']        = p['brand'].fillna('').astype(str).str.strip()
    p['description']  = p['description'].apply(parse_description)
    p['category']     = p['category'].apply(clean_category)
    p['image_url']    = p['image_url'].apply(parse_first_image).apply(upgrade_image)
    p['price']        = p['price'].apply(parse_price_inr)
    p = p.dropna(subset=['price'])
    p = p[p['product_name'].str.len() > 0]
    p = p[p['category'].str.len() > 0]
    p['price'] = p['price'].round(2)

    # Clean users
    u['user_id']    = u['user_id'].astype(str).str.strip()
    u['password']   = u['password'].fillna('').astype(str).str.strip()
    u['preference'] = u['preference'].fillna('General').astype(str).str.strip()
    u = u.drop_duplicates(subset=['user_id'])

    # Clean interactions
    i['user_id']    = i['user_id'].astype(str).str.strip()
    i['product_id'] = i['product_id'].astype(str).str.strip()
    i['rating']     = pd.to_numeric(i['rating'], errors='coerce')
    i = i.dropna(subset=['rating'])

    # TF-IDF
    p['features'] = (
        p['product_name'] + ' ' +
        p['category']     + ' ' +
        p['brand']        + ' ' +
        p['description'].str[:300]
    )
    tfidf     = TfidfVectorizer(stop_words='english', max_features=8000)
    tfidf_mat = tfidf.fit_transform(p['features'])
    pid_idx   = {pid: idx for idx, pid in enumerate(p['product_id'])}

    _products, _users, _interactions = p, u, i
    _tfidf, _tfidf_matrix, _prod_id_idx = tfidf, tfidf_mat, pid_idx

    print("Data loaded successfully!")
    return _products, _users, _interactions

# ─── RECOMMENDATION FUNCTIONS ─────────────────────────────────────────────────

def content_recommend(product_id, top_n=6):
    products, _, _ = get_data()
    pid = str(product_id)
    if pid not in _prod_id_idx: return []
    idx = _prod_id_idx[pid]
    sim = cosine_similarity(_tfidf_matrix[idx], _tfidf_matrix).flatten()
    top = sim.argsort()[::-1][1:top_n+1]
    return products.iloc[top].to_dict('records')

def collab_recommend(user_id, top_n=10):
    products, _, interactions = get_data()
    try:
        ui = interactions.pivot_table(
            index='user_id', columns='product_id', values='rating', fill_value=0)
        if user_id not in ui.index: return []
        sparse = csr_matrix(ui.values)
        nc = min(50, sparse.shape[0]-1, sparse.shape[1]-1)
        if nc < 1: return []
        svd    = TruncatedSVD(n_components=nc, random_state=42)
        rec    = np.dot(svd.fit_transform(sparse), svd.components_)
        scores = rec[ui.index.get_loc(user_id)]
        top_pids = np.argsort(scores)[::-1][:top_n]
        recs = []
        for pid in ui.columns[top_pids]:
            r = products[products['product_id'] == pid]
            if not r.empty: recs.append(r.iloc[0].to_dict())
        return recs
    except: return []

def get_trending(top_n=10):
    products, _, interactions = get_data()
    t = interactions.groupby('product_id')['rating'].count().sort_values(ascending=False).head(top_n)
    return products[products['product_id'].isin(t.index)].to_dict('records')

def get_category_products(category, top_n=8):
    products, _, _ = get_data()
    pattern = CATEGORY_MAP.get(category, category)
    mask    = products['category'].str.contains(pattern, case=False, na=False)
    subset  = products[mask]
    if subset.empty: subset = products.sample(min(top_n, len(products)))
    else:            subset = subset.sample(min(top_n, len(subset)))
    return subset.to_dict('records')

def smart_search(query, top_n=40):
    products, _, _ = get_data()
    q = query.strip().lower()
    if not q: return [], {}

    df = products.copy()

    def ww(text, word):
        return bool(re.search(r'\b' + re.escape(word) + r'\b', str(text).lower()))

    df['_score'] = 0
    df.loc[df['product_name'].apply(lambda x: ww(x, q)), '_score'] += 5
    df.loc[df['product_name'].str.contains(q, case=False, na=False) &
           ~df['product_name'].apply(lambda x: ww(x, q)), '_score'] += 2
    df.loc[df['brand'].apply(lambda x: ww(x, q)), '_score'] += 3
    df.loc[df['category'].str.contains(q, case=False, na=False), '_score'] += 2
    df.loc[df['description'].str.contains(q, case=False, na=False), '_score'] += 1

    df = df[df['_score'] > 0].sort_values('_score', ascending=False).head(top_n)
    flat = df.to_dict('records')

    grouped = {}
    for row in flat:
        cat = row.get('category', 'Other')
        if cat not in grouped: grouped[cat] = []
        grouped[cat].append(row)

    return flat, grouped

# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        _, users, _ = get_data()
        uid = request.form.get('username', '').strip()
        pwd = request.form.get('password', '').strip()
        row = users[users['user_id'] == uid]
        if not row.empty:
            if str(row.iloc[0].get('password', '')) == pwd:
                session['user_id'] = uid
                session['is_new']  = False
                return redirect(url_for('home'))
            error = 'Incorrect password.'
        else:
            error = 'User not found. Please sign up.'
    return render_template('login.html', error=error)

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == 'POST':
        _, users, _ = get_data()
        uid   = request.form.get('username', '').strip()
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
    if 'user_id' not in session:
        return redirect(url_for('login'))
    products, _, _ = get_data()
    uid    = session['user_id']
    is_new = session.get('is_new', False)
    prefs  = session.get('preference', '')
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
    if 'user_id' not in session:
        return redirect(url_for('login'))
    query = request.args.get('q', '').strip()
    flat, grouped = smart_search(query, top_n=40) if query else ([], {})
    return render_template('search.html', query=query,
                           results=flat, grouped=grouped,
                           user_id=session['user_id'])

@app.route('/product/<product_id>')
def product_page(product_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    products, _, _ = get_data()
    row = products[products['product_id'] == str(product_id)]
    if row.empty:
        return redirect(url_for('home'))
    product    = row.iloc[0].to_dict()
    cross_sell = content_recommend(product_id, top_n=6)
    return render_template('product.html', product=product,
                           cross_sell=cross_sell,
                           user_id=session['user_id'])

@app.route('/cart')
def cart():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('cart.html', user_id=session['user_id'])

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ─── JSON APIs ────────────────────────────────────────────────────────────────

@app.route('/api/fresh')
def api_fresh():
    products, _, _ = get_data()
    picks = products.sample(min(6, len(products)))[
        ['product_id', 'product_name', 'image_url', 'price', 'category']
    ].to_dict('records')
    return jsonify(picks)

@app.route('/api/category/<cat>')
def api_category(cat):
    items = get_category_products(cat, top_n=8)
    return jsonify([{k: v for k, v in p.items()
        if k in ['product_id', 'product_name', 'image_url', 'price', 'category', 'brand']}
        for p in items])

@app.route('/api/search')
def api_search():
    q = request.args.get('q', '').strip()
    flat, _ = smart_search(q, top_n=8) if q else ([], {})
    return jsonify([{k: v for k, v in p.items()
        if k in ['product_id', 'product_name', 'image_url', 'price', 'category', 'brand']}
        for p in flat])

# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)