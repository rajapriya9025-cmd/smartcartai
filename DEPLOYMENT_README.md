# SmartCartAI – Deployment Guide

## 📁 Final Project Structure

```
smart_cart_ai/
├── app.py                    ← Main Flask app (replace yours)
├── requirements.txt          ← Python dependencies
├── Procfile                  ← For Render deployment
├── render.yaml               ← Render config
├── data/
│   ├── products_clean.csv    ← Clean product data (use this!)
│   ├── users_clean.csv       ← Clean user data
│   └── interactions_clean.csv← Clean interaction data
├── templates/
│   ├── login.html
│   ├── register.html
│   ├── home.html             ← Updated (replace yours)
│   ├── search.html           ← Updated (replace yours)
│   └── product.html
└── static/
    └── (any static assets)
```

---

## 🖥️ Run Locally (Test First)

```bash
cd D:\smart_cart_ai
pip install -r requirements.txt
python app.py
```
Open: http://127.0.0.1:5000

---

## 🚀 Deploy on Render (FREE – Recommended for sharing with group)

### Step 1 – Push to GitHub
```bash
# In your project folder
git init
git add .
git commit -m "SmartCartAI final"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/smartcartai.git
git push -u origin main
```

### Step 2 – Deploy on Render
1. Go to **https://render.com** → Sign up free
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repo
4. Fill in:
   - **Name:** smartcartai
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`
5. Click **"Create Web Service"**
6. Wait ~3 minutes for build
7. Your app will be live at: `https://smartcartai.onrender.com`

### Share with your group:
Copy the URL from Render and share it. Anyone can access it from anywhere.

---

## ⚠️ Important: Upload Data Files to Render

Since data CSVs can't go on GitHub (too large), use Render's **disk** feature:

**Option A – Render Persistent Disk (easiest):**
1. In Render dashboard → your service → "Disks"
2. Add disk, mount path: `/opt/render/project/src/data`
3. Upload your 3 clean CSVs via Render Shell:
   ```bash
   # In Render Shell (under your service)
   ls data/   # verify files exist
   ```

**Option B – Upload small dataset sample:**
If CSVs are under 100MB, just commit them to GitHub:
```bash
git add data/products_clean.csv data/users_clean.csv data/interactions_clean.csv
git commit -m "add dataset"
git push
```

---

## 🔑 Login with Existing Users

Your dataset has users with:
- **Password for all:** `riya@123`
- Try any `user_id` from `users_clean.csv` (e.g., `Amazon Customer`, `Carla`, `Jaclyn`)

---

## 🔍 Search Tips for Your Dataset

This Amazon dataset contains:
| Category | Search examples |
|---|---|
| Pet Supplies | `dog food`, `cat toy`, `pet collar`, `leash` |
| Food & Grocery | `chocolate`, `tea`, `coffee`, `protein bar` |
| Office & Tech | `headphones`, `USB`, `keyboard`, `mouse` |
| Arts & Crafts | `pencil`, `paint`, `craft`, `sewing` |
| Garden | `seeds`, `garden hose`, `planter` |
| Music | `guitar`, `drum`, `piano` |

> Note: This dataset doesn't contain actual laptops/phones as products —
> it contains accessories FOR them (stands, cases, cables, etc.)

---

## 📊 Dataset Summary (after cleaning)

| File | Records | Notes |
|---|---|---|
| products_clean.csv | 24,035 | 20,412 with images; prices in INR |
| users_clean.csv | 19,929 | All passwords: riya@123 |
| interactions_clean.csv | 53,082 | Ratings 1.0–5.0 |
