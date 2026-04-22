import gdown
import os
os.makedirs('data',exist_ok=True)

files ={
    "data/products.csv": "https://drive.google.com/file/d/1yqbLHWgk6laINBX6QgG8OZfAW2cHfz0q/view?usp=drive_link",
    "data/users.csv": "https://drive.google.com/file/d/1iTdUZglcz7T4IDzyxvvF7y1gSJM4IULK/view?usp=sharing",
    "data/interactions.csv": "https://drive.google.com/file/d/1M9SlJdKKcitoroncJ4nur7GGlsOH7F_g/view?usp=sharing",
    "data/cleaned_data.csv":"https://drive.google.com/file/d/16f2n3C-Zrfns20VCzLuYksd1F03yBran/view?usp=sharing",
    "data/amazon_reviews.csv":"https://drive.google.com/file/d/10vgH6bfeQQZZV8DTEvEsXgU7wNyIuuAO/view?usp=sharing"

}
for path,file_id in files.items():
    if not os.path.exists(path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url,path,quiet=False)
        print(f"Downloaded: {path}")
    else:
        print(f"Already exists:{path}")
