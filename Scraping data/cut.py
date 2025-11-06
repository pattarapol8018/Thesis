import pandas as pd
import re

df = pd.read_csv("Dataset.csv")

df.columns = df.columns.str.strip().str.lower()

if 'make' in df.columns:
    df['series'] = df.apply(
        lambda x: re.sub(rf"^{x['make']}\s+", "", x['series'], flags=re.IGNORECASE),
        axis=1
    )
else:
    brands = ["Ford", "Honda", "Toyota", "Nissan", "Mazda", "Isuzu", "Mitsubishi", "MG"]
    pattern = r"^(" + "|".join(brands) + r")\s+"
    df['series'] = df['series'].str.replace(pattern, "", regex=True)

df.to_csv("Dataset_clean.csv", index=False, encoding="utf-8-sig")
print("ไฟล์ใหม่ชื่อ Dataset_clean.csv")
