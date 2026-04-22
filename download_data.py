import gdown
import os

os.makedirs('data', exist_ok=True)

files = {
    "data/products.csv":      "1yqbLHWgk6laINBX6QgG8OZfAW2cHfz0q",
    "data/users.csv":         "1iTdUZglcz7T4IDzyxvvF7y1gSJM4IULK",
    "data/interactions.csv":  "1M9SlJdKKcitoroncJ4nur7GGlsOH7F_g",
    "data/cleaned_data.csv":  "16f2n3C-Zrfns20VCzLuYksd1F03yBran",
    "data/amazon_reviews.csv":"10vgH6bfeQQZZV8DTEvEsXgU7wNyIuuAO",
}

for path, file_id in files.items():
    if not os.path.exists(path):
        try:
            gdown.download(id=file_id, output=path, quiet=False, fuzzy=True)
            print(f"Downloaded: {path}")
        except Exception as e:
            print(f"Warning: Could not download {path}: {e}")
    else:
        print(f"Already exists: {path}")