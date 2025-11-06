import pandas as pd

df = pd.read_csv("Dataset2.csv")

df['horsepower_hp'] = df['horsepower_hp'].astype(str).str.replace('แรงม้า', '', regex=False).str.strip()

df.to_csv("Dataset2.csv", index=False, encoding='utf-8-sig')

print("ลบคำว่าแรงม้าและอัปเดตไฟล์ Dataset2.csv ")
