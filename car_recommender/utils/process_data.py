from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import numpy as np
import os
import re

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path, encoding="utf-8-sig")

    # รีเนมคอลัมน์หลัก
    df = df.rename(columns={
        "Model Name": "full_name",
        "Price": "price_thb",
        "Details": "description",
        "model": "make",
        "engine": "engine",
        "horsepower_hp": "horsepower_hp",
        "gears": "gears",
        "fuel_type": "fuel_type",
    })

    #ทำความสะอาดราคา 
    df = df.dropna(subset=["full_name", "price_thb", "description", "make"])
    df = df[df["description"].astype(str).str.strip() != ""]

    df["price_thb"] = (
        df["price_thb"].astype(str)
        .str.replace("บาท", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("฿", "", regex=False)
        .str.strip()
    )
    df["price_thb"] = pd.to_numeric(df["price_thb"], errors="coerce")
    df = df.dropna(subset=["price_thb"])
    df["price_thb"] = df["price_thb"].astype(int)

    for col in ["engine_l", "horsepower_hp", "gears", "fuel_type"]:
        if col not in df.columns:
            df[col] = np.nan

    #horsepower_hp
    df["horsepower_hp"] = (
        df["horsepower_hp"]
        .astype(str)
        .str.replace("แรงม้า", "", regex=False)
        .str.extract(r"(\d+)", expand=False)
    )
    df["horsepower_hp"] = pd.to_numeric(df["horsepower_hp"], errors="coerce")

    #gears
    df["gears"] = (
        df["gears"]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df.loc[df["gears"].str.lower().isin(["nan", "none", ""]), "gears"] = np.nan

    #engine_cc:
    if "engine_cc" in df.columns:
        cc_num = (
            df["engine_cc"].astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("CC", "", regex=False)
            .str.extract(r"(\d+)", expand=False)
        )
        cc_num = pd.to_numeric(cc_num, errors="coerce")
        df["engine_cc"] = cc_num.astype("Int64")

        liters_from_cc = cc_num / 1000.0
        df["engine_l"] = df["engine_l"].fillna(liters_from_cc)

    #engine: 
    if "engine" in df.columns:
        def _parse_engine_l(x):
            if pd.isna(x):
                return np.nan
            s = str(x).replace(" .", ".")
            m = re.search(r"(\d+(?:\.\d+)?)\s*(?:L|ลิตร)", s, flags=re.I)
            if m:
                try:
                    return float(m.group(1))
                except:
                    return np.nan
            m2 = re.search(r"(\d)\s*\.\s*(\d)", s)  
            if m2:
                try:
                    return float(f"{m2.group(1)}.{m2.group(2)}")
                except:
                    return np.nan
            return np.nan

        df["engine_l"] = df["engine_l"].fillna(df["engine"].apply(_parse_engine_l))

    #year
    if "year_from_model_name" in df.columns:
        year_num = pd.to_numeric(df["year_from_model_name"], errors="coerce")
        df["year"] = year_num.astype("Int64")
    elif "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    else:
        df["year"] = pd.Series([pd.NA] * len(df), dtype="Int64")

    #series
    if "series" in df.columns:
        df["series"] = df["series"].astype(str).str.strip()
        df.loc[df["series"].isin(["", "nan", "None"]), "series"] = np.nan
    else:
        df["series"] = np.nan

    #normalize fuel_type 
    def _normalize_fuel(x):
        if pd.isna(x): return x
        s = str(x).strip().lower()
        mapping = {
            "ดีเซล": "Diesel", "diesel": "Diesel",
            "เบนซิน": "Petrol", "gasoline": "Petrol", "petrol": "Petrol",
            "ไฟฟ้า": "EV", "electric": "EV", "bev": "EV", "ev": "EV",
            "ไฮบริด": "Hybrid", "hybrid": "Hybrid", "hev": "Hybrid", "e:hev": "Hybrid",
            "ปลั๊กอินไฮบริด": "PHEV", "plug-in hybrid": "PHEV", "phev": "PHEV",
            "e20": "E20", "e85": "E85", "cng": "CNG", "lpg": "LPG", "mhev": "MHEV", "mild hybrid": "MHEV"
        }
        return mapping.get(s, x)
    df["fuel_type"] = df["fuel_type"].apply(_normalize_fuel)
    #ody type
    CANONICAL_TYPES = {
        "sedan": "sedan", "ซีดาน": "sedan", "รถเก๋ง": "sedan", "เก๋ง": "sedan",
        "hatchback": "hatchback", "แฮทช์แบ็ก": "hatchback", "แฮทช์แบ็ค": "hatchback",
        "pickup": "pickup", "ปิคอัพ": "pickup", "ปิกอัพ": "pickup", "กระบะ": "pickup",
        "suv": "suv", "เอสยูวี": "suv", "รถอเนกประสงค์": "suv",
        "mpv": "mpv", "รถตู้": "van", "van": "van", "wagon": "wagon",
        "coupe": "coupe", "คูเป้": "coupe", "convertible": "convertible"
    }

    def _normalize_type(x: str):
        if pd.isna(x):
            return x
        s = str(x).strip().lower()
        s = re.sub(r"\s+", " ", s)
        if s in CANONICAL_TYPES:
            return CANONICAL_TYPES[s]
        for key, val in CANONICAL_TYPES.items():
            if key in s:
                return val
        return s 

    if "type" in df.columns:
        df["type"] = df["type"].apply(_normalize_type)
    else:
        df["type"] = np.nan

    df = df.drop_duplicates(subset=["full_name", "make", "price_thb", "description"])

    return df



def build_faiss_index(df, save_path="embeddings/faiss_index.idx"):
    if df.empty:
        print("ไม่มีข้อมูลพร้อมใช้งานสำหรับสร้าง FAISS index")
        return None

    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    descriptions = df["description"].tolist()
    embeddings = model.encode(descriptions, show_progress_bar=True).astype("float32")

    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)   
    index.add(embeddings)

    os.makedirs("embeddings", exist_ok=True)
    faiss.write_index(index, save_path)

    
    df.to_csv("embeddings/clean_data.csv", index=False, encoding="utf-8-sig", float_format="%.0f")
    print("สร้างเสร็จเรียบร้อยแล้ว")
    print(f"rows: {len(df)}  |  index.ntotal: {index.ntotal}")
    return index

if __name__ == "__main__":
    df = load_and_clean_data("data/Dataset.csv")
    build_faiss_index(df)
