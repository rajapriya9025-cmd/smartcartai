import pandas as pd

def load_data():
    products = pd.read_csv(r"D:\smart_cart_ai\data\products.csv")
    users = pd.read_csv(r"D:\smart_cart_ai\data\users.csv")
    interactions = pd.read_csv(r"D:\smart_cart_ai\data\interactions.csv")
    return products, users, interactions
import csv

def load_users():
    users = []
    with open('data/users.csv', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            users.append(row)
    return users
import csv

def load_users():
    users = []
    with open('data/users.csv', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            users.append(row)
    return users

def save_user(user_id, location, preference, password):
    with open('data/users.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([user_id, location, preference, password])
