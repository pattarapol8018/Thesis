import pandas as pd
import re

# ---- 1) โหลดข้อมูล ----
df = pd.read_csv("Dataset.csv", encoding="utf-8-sig")
# บางไฟล์ใช้ utf-8 ปกติ
# df = pd.read_csv("Dataset.csv", encoding="utf-8")

# ---- 2) เตรียมข้อความสำหรับตรวจจับประเภท ----
def norm(s):
    if pd.isna(s):
        return ""
    # ลบช่องว่างพิเศษและทำเป็นตัวพิมพ์เล็ก
    return re.sub(r"\s+", " ", str(s)).strip().lower()

series_norm  = df.get("series", "").apply(norm)
details_norm = df.get("Details", "").apply(norm)

text = (series_norm + " " + details_norm).str.strip()

# ---- 3) กฎจัดประเภท (เรียงจากเฉพาะทาง → กว้าง) ----
# หมายเหตุ: ปรับ/เติม keyword ได้ตาม dataset จริง
PICKUP_KEYWORDS = [
    r"\bpickup\b", "รถกระบะ", "กระบะ", "raptor", "revo", "triton", "navara",
    "ranger", "d-max", "bt-50", "wildtrak", "double cab", "open cab", "standard cab"
]
SUV_KEYWORDS = [
    r"\bsuv\b", "เอสยูวี", "ครอสโอเวอร์", "crossover",
    "fortuner", "pajero", "mu-x", "cr-v", "hr-v", "cx-5", "cx-8",
    "x-trail", "corolla cross", "crosstrek", "outlander", "xpv", "xpv cross"
]
SEDAN_KEYWORDS = [
    "sedan", "ซีดาน", "เก๋ง", "camry", "accord", "altis", "civic",
    "city e:hev", "almera", "attrage", "mazda 3", "sylphy"
]
HATCHBACK_KEYWORDS = [
    "hatchback", "แฮทช์แบ็ก", "แฮทช์แบ็ค", "yaris", "swift", "jazz"
]
MPV_KEYWORDS = [
    "mpv", "รถครอบครัว", "7 ที่นั่ง", "stargazer", "avanza", "ertiga", "xpander"
]

def detect_type(s: str) -> str:
    t = s
    # Pickup
    if any(re.search(k, t) if k.startswith(r"\b") else (k in t) for k in PICKUP_KEYWORDS):
        return "Pickup"
    # SUV
    if any(re.search(k, t) if k.startswith(r"\b") else (k in t) for k in SUV_KEYWORDS):
        return "SUV"
    # Sedan
    if any(k in t for k in SEDAN_KEYWORDS):
        return "Sedan"
    # Hatchback
    if any(k in t for k in HATCHBACK_KEYWORDS):
        return "Hatchback"
    # MPV
    if any(k in t for k in MPV_KEYWORDS):
        return "MPV"
    return "Other"

df["type"] = text.apply(detect_type)

# ---- 4) สรุปจำนวนต่อประเภทรถ ----
counts = df["type"].value_counts().rename_axis("vehicle_type").reset_index(name="count")
print(counts)

# ---- 5) บันทึกผลลัพธ์ ----
counts.to_csv("type_counts.csv", index=False, encoding="utf-8-sig")
df.to_csv("Dataset_with_type.csv", index=False, encoding="utf-8-sig")

print("\nSaved:")
print(" - type_counts.csv (ยอดรวมต่อประเภทรถ)")
print(" - Dataset_with_type.csv (เพิ่มคอลัมน์ type แล้ว)")
