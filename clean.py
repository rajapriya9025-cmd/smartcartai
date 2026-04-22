with open('d:/smart_cart_ai/data/users.csv', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if 19950 <= i <= 19960:
            print(i, line)
            
            pd.read_csv('d:/smart_cart_ai/data/users.csv', on_bad_lines='skip')