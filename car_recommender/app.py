from flask import Flask, request, render_template, jsonify, session
import pandas as pd
import numpy as np
import faiss, re, os, json
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import random

app = Flask(__name__)
app.secret_key = "change_this_to_a_secure_random_string"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



df = pd.read_csv("embeddings/clean_data.csv")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
index = faiss.read_index("embeddings/faiss_index.idx")


if "row_id" not in df.columns:
    df["row_id"] = df.index

DEFAULT_PREFS = {
    "answers": {},
    "asked": [],
    "pending_key": "price",
    "last_question": None,
    "_skip": [],
    "extra_done": False,
    "stage": None,
    "last_results": []
}

def reset_state_all():
    """รีเซ็ตสถานะทั้งหมด (prefs + session) ใช้ได้ตอนมี request context"""
    try:
        session.clear()
    except Exception:
        pass
    save_prefs(DEFAULT_PREFS.copy())
    return DEFAULT_PREFS.copy()

boot_done = False
@app.before_request
def boot_reset_once():
    global boot_done
    if not boot_done:
        reset_state_all()
        boot_done = True

def json_safe(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    return obj

def get_prefs():
    prefs = session.get("prefs")
    if not isinstance(prefs, dict):
        prefs = {"answers": {}, "asked": [], "pending_key": None, "last_question": None}
    return prefs

def save_prefs(prefs):
    session["prefs"] = json_safe(prefs)

def safe_int(v):
    try:
        return int(v) if pd.notna(v) else None
    except:
        return None

def safe_float(v):
    try:
        return float(v) if pd.notna(v) else None
    except:
        return None

def parse_gears(v):
    if v is None:
        return None
    n = pd.to_numeric(v, errors="coerce")
    if pd.notna(n):
        try:
            return int(n)
        except:
            pass
    m = re.search(r"(\d+)", str(v))
    return int(m.group(1)) if m else None

def extract_price_range(query: str):
    if not query:
        return (None, None)
    s = (query or "").lower().strip()
    thai_digits = "๐๑๒๓๔๕๖๗๘๙"
    s = (
        s.translate(str.maketrans(thai_digits, "0123456789"))
        .replace(",", "")
        .replace(" ", "")
    )

    def to_number(token: str):
        m = re.match(r"(?P<num>\d+(\.\d+)?)(?P<u>ล้าน|แสน|หมื่น|พัน|k)?", token)
        if not m:
            return None
        num = float(m.group("num"))
        u = m.group("u")
        mul = 1
        if u == "ล้าน":
            mul = 1_000_000
        elif u == "แสน":
            mul = 100_000
        elif u == "หมื่น":
            mul = 10_000
        elif u in ("พัน", "k"):
            mul = 1_000
        return int(num * mul)
    m_no_more = re.search(r"ไม่เกิน\s*(\d+(\.\d+)?(ล้าน|แสน|หมื่น|พัน|k)?)", s)
    if m_no_more:
        x = to_number(m_no_more.group(1))
        if x:
            return (0, x)

    m_between = re.search(
        r"(ระหว่าง|ช่วง|ตั้งแต่)?(\d+(\.\d+)?(ล้าน|แสน|หมื่น|พัน|k)?)\s*(ถึง|-)\s*(\d+(\.\d+)?(ล้าน|แสน|หมื่น|พัน|k)?)",
        s,
    )
    if m_between:
        a = to_number(m_between.group(2))
        b = to_number(m_between.group(6))
        if a and b:
            return (min(a, b), max(a, b))

    m_budget_unbounded = re.search(r"(งบ\s*)?(?<!ไม่)(เกิน|มากกว่า|>)+\s*(\d+(\.\d+)?(ล้าน|แสน|หมื่น|พัน|k)?)", s)
    if m_budget_unbounded:
        return (None, None)

    m_one = re.search(r"(\d+(\.\d+)?(ล้าน|แสน|หมื่น|พัน|k)?)", s)
    if m_one:
        x = to_number(m_one.group(1))
        if x:
            return (max(0, x - 100_000), x + 100_000)

    return (None, None)

def extract_make_from_query(query):
    if not query:
        return None
    q = str(query).strip().lower()
    if not q:
        return None

    found = {}

    
    makes = df["make"].dropna().astype(str).str.lower().unique().tolist()
    for mk in makes:
        if mk in q:
            found["make"] = mk
            break  
    if "series" in df.columns:
        all_series = (
            df["series"].dropna().astype(str).str.lower().unique().tolist()
        )
        for sr in all_series:
            root = sr.split()[0]  
            if sr in q or root in q:
                found["series"] = root  
                break
    return found or None

df["type"] = df.get("type").fillna("").astype(str)
df["type_norm"] = df["type"].str.lower().str.strip()

BODY_MAP = {
    "pickup": "pickup", "truck": "pickup", "กระบะ": "pickup",
    "sedan": "sedan", "saloon": "sedan", "เก๋ง": "sedan", "รถเก๋ง": "sedan",
    "suv": "suv",
    "mpv": "mpv", "van": "mpv",
    "hatchback": "hatchback"
}

def fuel_to_thai(x):
    s = str(x).strip().lower()
    mapping = {
        "diesel": "ดีเซล",
        "petrol": "เบนซิน",
        "gasoline": "เบนซิน",
        "ev": "ไฟฟ้า",
        "bev": "ไฟฟ้า",
        "hybrid": "ไฮบริด",
        "hev": "ไฮบริด",
        "e:hev": "ไฮบริด",
        "phev": "ปลั๊กอินไฮบริด",
        "plug-in hybrid": "ปลั๊กอินไฮบริด",
        "e20": "E20",
        "e85": "E85",
        "cng": "CNG",
        "lpg": "LPG",
        "mhev": "ไฮบริดแบบ MHEV",
    }
    return mapping.get(s, None)

USAGE_HINTS = [
    "ครอบครัว",
    "เดินทางไกล",
    "ในเมือง",
    "ออฟโรด",
    "บรรทุก",
    "ประหยัดน้ำมัน",
    "แรง",
    "กว้าง",
    "คอมแพค",
    "suv",
    "ซีดาน",
    "กระบะ",
    "mpv",
    "ขนของ",
    "ขึ้นเขา",
]
TRANS_HINTS = {
    "AT": ["ออโต้", "อัตโนมัติ", "auto", "at", "cvt", "dct", "เกียร์อัตโนมัติ"],
    "MT": ["ธรรมดา", "เกียร์ธรรมดา", "manual", "mt"],
}
BODY_HINTS = {
    "suv": ["suv", "เอสยูวี"],
    "sedan": ["ซีดาน", "sedan","เก๋ง","รถเก๋ง"],
    "hatchback": ["แฮทช์", "hatch"],
    "mpv": ["mpv", "ครอบครัว", "7ที่นั่ง", "7 ที่นั่ง", "อเนกประสงค์"],
    "pickup": ["กระบะ", "ปิคอัพ", "ปิกอัพ", "pickup","รถกระบะ"],
}
FUEL_HINTS = {
    "diesel": ["ดีเซล", "diesel"],
    "petrol": ["เบนซิน", "gasoline", "petrol"],
    "hybrid": ["ไฮบริด", "hev", "mhev", "phev", "ปลั๊กอิน"],
    "ev": ["ไฟฟ้า", "ev", "bev"],
}
NO_WORDS = ("ไม่มี", "ไม่เน้น", "อะไรก็ได้", "เฉยๆ", "เฉย ๆ", "ยัง", "ไม่", "ข้าม")

def _extract_usage(text):
    return [k for k in USAGE_HINTS if k in (text or "").lower()]


def _extract_transmission(text):
    s = (text or "").lower()
    for k, kws in TRANS_HINTS.items():
        if any(w in s for w in kws):
            return k
    return None

def _extract_fuel(text):
    s = (text or "").lower()
    for name, kws in FUEL_HINTS.items():
        if any(w in s for w in kws):
            return name
    return None


ASK_ORDER = [
    "make",       
    "usage",       
    "trans",       
    "price",       
    "fuel",
    "drive",        
    "extra",       
]

ASK_VARIANTS = {
    "make": [
        "อยากดูรถรุ่นไหนเป็นพิเศษไหมครับ?",
        "พอจะมีรุ่นที่สนใจอยู่ในใจไหมครับ?",
    ],
    "usage_text": [
        "อยากได้รถไว้ใช้เดินทางไกล ขึ้นเขา หรือใช้งานในเมืองครับ?",
        "ลักษณะการใช้งานหลัก ๆ เป็นแบบไหนครับ เช่น เดินทางบ่อยหรือใช้งานทั่วไป?",
        "ส่วนใหญ่จะใช้รถเพื่ออะไรครับ เช่น ไปทำงาน หรือท่องเที่ยว?",
    ],
    "trans": [
        "ชอบแบบเกียร์อัตโนมัติหรือธรรมดาครับ?",
        "เรื่องระบบเกียร์มีความชอบแบบไหนเป็นพิเศษไหมครับ?",
    ],
    "price": [
        "งบประมาณที่ตั้งไว้สำหรับรถคันนี้อยู่ที่ประมาณเท่าไหร่ครับ?",
        "อยากได้รถในช่วงราคาประมาณกี่บาทครับ?",
        "คุณมีงบราคาที่ตั้งไว้ไหมครับ?",
    ],
    "fuel": [
        "อยากได้รถที่ใช้น้ำมันประเภทไหนครับ เช่น เบนซิน ดีเซล หรืออิ่นๆ",
    ],
    "drive": [
        "อยากได้ระบบขับเคลื่อนแบบไหนครับ เช่น ขับหน้า , ขับหลัง  หรือขับสี่ ครับ?",
    ],
    "extra": [
        "มีรายละเอียดเพิ่มเติมไหมครับ เช่น cc แรงม้า สี หรือฟังก์ชันอื่น ๆ?",
    ],
}
SYSTEM_QUESTIONER = """
คุณเป็นผู้ช่วยถามคำถามสั้น ๆ ต่อผู้ใช้
พูดกระชับ ไม่เกินหนึ่งประโยค
ห้ามใช้คำเกริ่น เช่น 'แน่นอนครับ', 'เข้าใจแล้วครับ'
ต้องลงท้ายด้วยเครื่องหมาย '?' เท่านั้น
"""

def llm_generate_question(
    field: str, answers: dict, last_question: str = ""
) -> str | None:
    """
    ให้ LLM แต่งคำถามสด ๆ ตาม field และ context ที่รู้
    คืน None หากเกิดข้อผิดพลาด เพื่อให้ fallback
    """
    try:
        ctx = {
            "ask_for": field,
            "known_answers": answers,
            "last_question": last_question or "",
        }
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.9,  
            max_tokens=60,
            messages=[
                {"role": "system", "content": SYSTEM_QUESTIONER},
                {"role": "user", "content": f"บริบท: {ctx}\nโปรดสร้างคำถามเดียว"},
            ],
        )
        q = (resp.choices[0].message.content or "").strip()
        
        if not q or q == last_question or len(q.split()) < 3:
            return None
        return q
    except Exception:
        return None
def format_question_for_ui(raw_q: str, answers: dict) -> str:
    user_make = answers.get("make")
    series    = answers.get("series")

    ctx = ""
    if user_make:
        ctx += f"เกี่ยวกับ {user_make}"
        if series:
            ctx += f" รุ่น {series}"

    system_prompt = (
        "คุณเป็นระบบช่วยปรับแต่งประโยคคำถามให้สุภาพ กระชับ และเป็นกันเอง "
        "หน้าที่ของคุณคือ 'ปรับประโยคคำถาม' เท่านั้น "
        "ห้ามตอบกลับเป็นประโยคบรรยาย ห้ามให้คำแนะนำ ห้ามพูดยาว "
        "ให้ตอบเฉพาะประโยคคำถาม 1 ประโยคเท่านั้น "
        "ห้ามใช้คำว่า สวัสดี และห้ามใส่คำอธิบาย"
    )

    user_prompt = f"ประโยคเดิม: {raw_q}\nบริบทเพิ่มเติม: {ctx}\nกรุณาปรับให้เหมาะสมและสุภาพขึ้น"

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.5,
            max_tokens=50,
        )
        new_q = resp.choices[0].message.content.strip()
        return new_q if new_q else raw_q
    except:
        return raw_q


def choose_natural_question(field: str, answers: dict, prefs: dict) -> str:
    last_q = (prefs or {}).get("last_question", "")
    variants = ASK_VARIANTS.get(field, []) or ["ขอรายละเอียดเพิ่มเติมหน่อยครับ?"]
    random.shuffle(variants)
    base_q = variants[0].strip()
    try:
        ctx_parts = []
        for k in ("make","series","usage_text","body","fuel","trans","price","drive"):
            v = answers.get(k)
            if v:
                ctx_parts.append(f"{k}={v}")
        ctx = ", ".join(ctx_parts) or "-"

        prompt = (
            "รีเขียนประโยคคำถามต่อผู้ใช้ให้สุภาพ เป็นกันเอง กระชับ และเป็นธรรมชาติ "
            "ห้ามใช้คำเกริ่นนำ เช่น 'แน่นอนเลย', 'ลองปรับเป็น', 'คุณอาจจะถามว่า' "
            "ห้ามใส่คำพูดสรุป/บรรยายอื่น ๆ ให้ส่ง 'ประโยคคำถามเดียวเท่านั้น' และลงท้ายด้วย '?' \n"
            f"บริบท: {ctx}\n"
            f'คำถามตั้งต้น: "{base_q}"'
        )

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=60,
            messages=[
                {"role": "system", "content": "คุณเป็นผู้ช่วยขายรถยนต์ ตอบเป็น 'คำถามเดียว' ที่สุภาพและเป็นกันเอง"},
                {"role": "user", "content": prompt},
            ],
        )
        q = (resp.choices[0].message.content or "").strip()
        q = re.sub(r'^[\'"“”]+|[\'"“”]+$', '', q) 
        q = re.sub(r'^(แน่นอน(เลย)?!?|ได้เลย(ครับ)?!?|โอเค(ครับ)?!?)[\s,:-]*', '', q, flags=re.IGNORECASE)
        q = re.sub(r'^(ลอง(ปรับ|ถาม)ว่า|คุณอาจจะถามว่า|ตัวอย่างเช่น|เช่น)[:\s-]*', '', q, flags=re.IGNORECASE)
        if "?" in q:
            q = q.split("?")[0].strip() + "?"
        if not q.endswith("?"):
            q = q.rstrip(" .!~") + "?"
        if not q or len(q.split()) < 3 or q.lower() == base_q.lower():
            q = base_q
        if q == last_q:
            q = base_q if base_q != last_q else (variants[1] if len(variants) > 1 else base_q)
        return q

    except Exception:
        return base_q


def _next_missing_field(answers: dict) -> str | None:
    prefs = get_prefs()
    order = prefs.get("_ask_order")

    if not order:
        middle = ["usage_text", "trans", "fuel", "drive"]  
        random.shuffle(middle)
        order = ["price", "make"] + middle + ["extra"]
        prefs["_ask_order"] = order
        save_prefs(prefs)

    skip = set(prefs.get("_skip", []))
    for f in order:
        if answers.get(f) or f in skip:
            continue
        return f
    return None


def _fallback_question(field: str) -> str:
    prefs = get_prefs()
    return choose_natural_question(field, prefs.get("answers", {}), prefs)

SYSTEM_PLANNER = (
    "คุณคือผู้ช่วยด้านรถยนต์ หน้าที่คือ 'ถามต่อหนึ่งคำถามสั้น ๆ' เพื่อคัดเลือกรถได้แม่นขึ้น\n"
    "- ถามเฉพาะเรื่องรถ (งบประมาณ/การใช้งาน/แบรนด์/เชื้อเพลิง/เกียร์/ตัวถัง/ที่นั่ง/ขับเคลื่อน/ขนาด)\n"
    "- ห้ามถามเรื่องส่วนตัว และถามได้ครั้งละ 1 คำถาม\n"
    'ตอบกลับเป็น JSON: {"ask_for":"<หนึ่งในฟิลด์ที่รองรับ>", "question":"<คำถามสั้น>"}'
)

def next_question(user_input: str, answers: dict):
    
    miss = _next_missing_field(answers)
    if not miss:
        return {"ask_for": None, "question": "ขอบคุณครับ เดี๋ยวผมหารถที่เหมาะให้เลยครับ!"}

    variants = ASK_VARIANTS.get(miss, [])
    base_q = random.choice(variants) if variants else "ช่วยให้ข้อมูลเพิ่มเติมหน่อยครับ?"

    try:
        ai_prompt = f"ช่วยแต่งประโยคคำถามให้นุ่มนวลและเป็นธรรมชาติมากขึ้น โดยคงความหมายเดิม: '{base_q}'"
        ai_resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "คุณเป็นผู้ช่วย AI ที่พูดจาสุภาพ เหมือนเพื่อนแนะนำเรื่องรถ"},
                {"role": "user", "content": ai_prompt},
            ],
            temperature=0.5,
            max_tokens=60,
        )
        q = ai_resp.choices[0].message.content.strip()
        for p in ("แน่นอนครับ!", "แน่นอนครับ", "คุณอาจจะถามว่า", "แบบนี้จะฟังดูนุ่มนวลและเป็นธรรมชาติขึ้นครับ"):
            if q.startswith(p):
                q = q[len(p):].lstrip(" ：:–-\"'“” ").strip()
        q = q.strip('“”"\' ')
        if "?" in q:
            q = q.split("?")[0].strip() + "?"
        if not q.endswith("?") or len(q.split()) < 3 or len(q) > 120:
            q = base_q
        if not q or q == base_q or len(q.split()) < 3:
            q = base_q

    except Exception as e:
        q = base_q

    return {"ask_for": miss, "question": q}

def price_llm(text: str) -> tuple[int|None, int|None]:
    
    if not text or not text.strip():
        return (None, None)

    sys = (
        "คุณคือตัวแยกราคาไทย ให้ตอบ JSON เท่านั้น "
        "price_min/price_max เป็นจำนวนเต็มบาท ถ้าเป็น 'ไม่เกิน X' ให้ min=0,max=X; "
        "ถ้า 'ตั้งแต่ X ขึ้นไป' ให้ min=X,max=null; ช่วง X–Y ให้ min=X,max=Y; "
        "รองรับรูปแบบ 1,000,000 / 1ล้าน / 900k / 0.8m / 7-9แสน / ประมาณ 1 ล้าน "
        "อย่าใส่หน่วยลงในตัวเลข"
    )
    user = f"""ข้อความ: "{text}"
ตอบ JSON เท่านั้น เช่น {{"price_min": 0, "price_max": 1000000}} หรือ {{"price_min": 700000, "price_max": 900000}} 
ถ้าระบุไม่ได้ ให้ตอบ {{"price_min": null, "price_max": null}}"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=60,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": user},
            ],
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        mn = data.get("price_min")
        mx = data.get("price_max")
        mn = int(mn) if isinstance(mn, (int, float, str)) and str(mn).strip().isdigit() else None
        mx = int(mx) if isinstance(mx, (int, float, str)) and str(mx).strip().isdigit() else None
        return (mn, mx)
    except Exception:
        return (None, None)

def extract_json(text: str) -> dict:
    """พยายามดึง JSON object ก้อนแรกจากสตริง (กัน LLM ใส่คำพูด/หัวข้อมาเกิน)
    คืน {} ถ้าดึงไม่ได้"""
    if not text:
        return {}
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    import re
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}

def extract_answers(user_input: str, known_answers: dict) -> dict:
    try:
        print("[extract_answers] input:", user_input)
        prompt = f"""คุณเป็นผู้ช่วยขายรถยนต์
แยกข้อมูลที่ผู้ใช้ตอบออกเป็น JSON เท่านั้น (ห้ามมีคำอธิบายอื่น)
คีย์ที่รองรับ: price, usage_text, make, fuel, trans, drive, series
ถ้าไม่พบให้ใส่ค่าว่าง เช่น "" (string ว่าง) ห้ามลบคีย์ทิ้ง

ตัวอย่าง:
ผู้ใช้: "ไม่เกิน 1 ล้าน ขอดู nissan"
คำตอบ:
{{"price": "ไม่เกิน 1 ล้าน", "make": "nissan", "series": "", "usage_text": "", "fuel": "", "trans": "", "drive": ""}}

ผู้ใช้: "{user_input}"
ตอบเป็น JSON object เพียงอย่างเดียว
"""

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,         
            max_tokens=120,
            messages=[
                {"role": "system", "content": "คุณเป็นตัวแยกข้อมูลและต้องตอบเป็น JSON object เท่านั้น"},
                {"role": "user", "content": prompt},
            ],
        )

        llm_text = (resp.choices[0].message.content or "").strip()
        parsed = extract_json(llm_text)  
        
        mk = extract_make_from_query(user_input)
        if mk:
            known_answers.update(mk)
        if not known_answers.get("usage_text") and parsed.get("usage"):
            known_answers["usage_text"] = parsed["usage"]
            parsed.pop("usage", None)      
            
        print("[extract_answers] raw_llm:", llm_text)
        print("[extract_answers] parsed:", parsed)
        
        allow_keys = {"price","usage_text","make","series","fuel","trans","body","drive"}
        for k, v in parsed.items():
            if k in allow_keys and v:
                known_answers[k] = v

        u_text = known_answers.get("usage_text")
        if isinstance(u_text, str) and u_text.strip():
            t = u_text.lower()

            not_usage_keywords = [
                "แรงม้า", "แรงสุด", "เร็ว", "ความเร็ว", "0-100", "แรง",    
                "เกียร์", "ออโต้", "อัตโนมัติ", "ธรรมดา", "manual", "mt", "at", "cvt", "dct"  
            ]
            if any(k in t for k in not_usage_keywords):
                known_answers.pop("usage", None)   
            else:
                hints = _extract_usage(u_text)
                if hints:
                    known_answers["usage"] = hints
                else:
                    known_answers.pop("usage", None)

        btext = f"{known_answers.get('usage_text','')} {user_input}".lower()

        _body_order = ["sedan", "suv", "mpv", "hatchback", "pickup"]
        detected_body = None

        for _b in _body_order:
            if any(w in btext for w in BODY_HINTS[_b]):
                detected_body = _b
                break

        if detected_body:
            known_answers["body"] = detected_body

        
        
        ttext = f"{(known_answers.get('usage_text') or '')} {user_input}".lower()
        tr = _extract_transmission(ttext)  
        if tr:
            known_answers["trans"] = tr

        if (known_answers.get("usage_text","").strip() in
            ("ออโต้","เกียร์ออโต้","เกียร์อัตโนมัติ","ธรรมดา","เกียร์ธรรมดา","ชอบขับออโต้")):
            known_answers["usage_text"] = ""
            
        mn, mx = price_llm(user_input)
        if known_answers.get("price") != (None, None):
            if mn is not None or mx is not None:
                known_answers["price"] = (mn or 0, mx or 10_000_000_000)
    except Exception as e:
        print("extract_answers error:", e)
    return known_answers

def detect_reference(user_input: str, last_results: list[dict]) -> int | None:
    if not last_results:
        return None
    m = re.search(r"คันที่\s*(\d+)", user_input)
    if m:
        idx = int(m.group(1)) - 1
        return idx if 0 <= idx < len(last_results) else None
    t = user_input.lower()
    for i, r in enumerate(last_results):
        n = (r.get("full_name") or "").lower()
        if n and n in t:
            return i
    return None

def is_efficiency_question(user_input: str) -> bool:
    t = user_input.lower()
    return any(k in t for k in ["ประหยัด", "กินน้ำมัน", "อัตราสิ้นเปลือง", "กม/ลิตร", "km/l"])

def is_compare_intent(user_input: str) -> bool:
    t = user_input.lower()
    return ("เทียบ" in t) or ("ต่างกัน" in t) or ("vs" in t) or ("เปรียบเทียบ" in t)

def rag_retrieve_context(user_query: str, answers: dict, top_n: int = 5):
   
    query = user_query or ""
    if isinstance(answers, dict):
        if answers.get("make"):
            query += f" {answers['make']}"
        if answers.get("series"):
            query += f" {answers['series']}"
        if answers.get("usage_text"):
            query += f" {answers['usage_text']}"

        try:
            price_min, price_max = (0, 10_000_000_000)
            if answers.get("price"):
                price_min, price_max = answers["price"]

            results = search_similar_rows(
                user_query=query,
                price_min=price_min,
                price_max=price_max,
                target_make=answers.get("make"),
                top_n=top_n,
                usage_hints=answers.get("usage", []),
                answers=answers,
            )
            return results or []            
        except Exception as e:
            print("RAG error:", e)
            return []
    else:
        return []


def rag_answer_followup(user_input, answers_or_rows) -> str:
    
   
    rows = []
    if isinstance(answers_or_rows, list):
        
        rows = answers_or_rows
    elif isinstance(answers_or_rows, dict):
        retrieved = rag_retrieve_context(user_input, answers_or_rows, top_n=5)
        
        if retrieved and isinstance(retrieved[0], (tuple, list)):
            rows = [r[1] for r in retrieved]
        else:
            rows = retrieved or []
    else:
        rows = []

    if not rows:
        return "ยังไม่พบข้อมูลที่ตรง ลองบอกยี่ห้อ งบประมาณ หรือการใช้งานเพิ่มอีกนิดได้ไหมครับ"

    
    try:
        exps = rag_generate_answer(rows, user_query=user_input) or []
    except Exception as e:
        print(" RAG generate error:", e)
        exps = [""] * len(rows)

   
    lines = []
    for i, row in enumerate(rows[:5]):
        title = str(row.get("full_name") or "-")
        expl  = (exps[i] if i < len(exps) else "") or ""
        if expl:
            lines.append(f"{i+1}. {title} — {expl}")
        else:
            price_txt = row.get("price_thb")
            price_txt = f"{int(price_txt):,} บาท" if isinstance(price_txt, (int, float)) else "-"
            lines.append(f"{i+1}. {title} (ราคา {price_txt})")
    return "\n\n".join(lines)

def search_similar_rows(user_query, price_min=None, price_max=None,
                        target_make=None, top_n=5, usage_hints=None,answers=None,target_series=None):
          
    target_body = (answers or {}).get("body")
    def _row_is_body(row, want):
        t = (row.get("type_norm") or row.get("type") or "").strip().lower()
        t = BODY_MAP.get(t, t)          # map ให้เป็นคีย์มาตรฐาน
        return (t == want) if want else True

    answers = answers or {}
    qemb = model.encode([user_query]).astype("float32")
    faiss.normalize_L2(qemb)
    
    k = min(max(top_n * 50, 200), index.ntotal)
    scores, indices = index.search(qemb, k)  
    

    matched = []
    for s, idx in zip(scores[0], indices[0]):
        if idx >= len(df):
            continue  
        row = df.iloc[idx]
        if not target_body and usage_hints and ("ในเมือง" in usage_hints):
            if _row_is_body(row, "pickup"):
                continue
        if target_body and not _row_is_body(row, target_body):
            continue 
        try:
            price = float(row["price_thb"])
            make_in_row = str(row.get("make", "")).strip().lower()
            series_in_row = str(row.get("series", "")).strip().lower()
        except Exception:
            continue
        trans_pref = (answers or {}).get("trans")
        if trans_pref:
            name_join = f"{row.get('full_name','')} {row.get('series','')} {row.get('description','')} {row.get('gears','')}"
            name_lc = name_join.lower()
            if trans_pref == "MT":
                if any(k in name_lc for k in ["a/t", " auto", "ออโต้", "cvt", "dct"]):
                    continue
                if re.search(r'\b\d+\s*at\b', name_lc) or re.search(r'\bat\b', name_lc):
                    continue
            if trans_pref == "AT":
                if any(k in name_lc for k in ["m/t", " manual", "ธรรมดา"]):
                    continue
                if re.search(r'\b\d+\s*mt\b', name_lc) or re.search(r'\bmt\b', name_lc):
                    continue
        if target_make:
            if isinstance(target_make, str):
                raw = re.split(r'[,\s/&|]+', target_make.lower())
                stop = {"และ", "กับ", "หรือ", "and", "or"}
                makes = [m for m in raw if m and m not in stop]
            elif isinstance(target_make, list):
                makes = [m.lower() for m in target_make]
            else:
                makes = []
            car_name = (str(row.get("full_name", "")) + " " + make_in_row).lower()
            if not any(m in car_name for m in makes):
                continue
        if target_series:
            if str(target_series).strip().lower() not in series_in_row:
                continue
        if price_min is not None and price < float(price_min):
            continue
        if price_max is not None and price > float(price_max):
            continue
        matched.append((float(s), row))
        if len(matched) >= top_n:
            break

    if len(matched) < top_n:
        cand = df.copy()
        if target_make:
            cand = cand[
                cand["make"].astype(str).str.lower().str.contains(target_make, na=False)]
        if price_min is not None:
            cand = cand[pd.to_numeric(cand["price_thb"], errors="coerce") >= float(price_min)]
        if price_max is not None:
            cand = cand[pd.to_numeric(cand["price_thb"], errors="coerce") <= float(price_max)]
        if target_body:
            cand = cand[
                cand["type_norm"].apply(lambda t: BODY_MAP.get(str(t).lower(), str(t).lower()) == target_body)]
        if cand.empty and (price_min is not None or price_max is not None):
            cand = df.copy()
            if target_make:
                cand = cand[
                    cand["make"].astype(str).str.lower().str.contains(target_make, na=False)
                ]
            target = price_max if price_max is not None else price_min
            cand["price_diff"] = (
                pd.to_numeric(cand["price_thb"], errors="coerce") - float(target)
            ).abs()
            cand = cand.sort_values("price_diff", ascending=True)

        cand = cand.sort_values("price_thb", ascending=True).head(top_n)

        real_scored = []
        for ridx, r in cand.iterrows():
            try:
                x = np.asarray(index.reconstruct(int(ridx)), dtype="float32")  
                dist = float(((qemb[0] - x) ** 2).sum())                       
            except Exception:
                dist = 1e9 
            real_scored.append((dist, r))
        matched = real_scored

    hints = usage_hints or []

    def _bonus(row):
        bonus = 0.0
        eng = pd.to_numeric(row.get("engine_l"), errors="coerce")
        gears = pd.to_numeric(row.get("gears"), errors="coerce")
        hp = pd.to_numeric(row.get("horsepower_hp"), errors="coerce")
        fuel = (row.get("fuel_type") or "").lower()
        name = (row.get("full_name") or "").lower()

        if "เดินทางไกล" in hints:
            if pd.notna(eng) and eng >= 1.5: bonus += 0.35
            if pd.notna(gears) and gears >= 6: bonus += 0.2
            if "diesel" in fuel: bonus += 0.35
            if any(k in name for k in ["suv", "pickup", "navara", "terra"]): bonus += 0.2
            if pd.notna(hp) and hp >= 120: bonus += 0.1

        if "ในเมือง" in hints:
            if pd.notna(eng) and eng <= 1.3: bonus += 0.3
            if "hybrid" in fuel or "hev" in fuel: bonus += 0.2
            if any(k in name for k in ["almera", "march", "yaris", "mazda2"]): bonus += 0.15

        return bonus
    
    def _soft_pref_adjust(row, answers):  
        adj = 0.0
        
        min_hp = answers.get("min_hp")
        if min_hp:
            hp = pd.to_numeric(row.get("horsepower_hp"), errors="coerce")
            if pd.notna(hp) and hp < int(min_hp):
                adj += 0.4

        min_cc = answers.get("min_cc")
        if min_cc:
            cc = pd.to_numeric(row.get("engine_cc"), errors="coerce")
            if pd.notna(cc) and cc < int(min_cc):
                adj += 0.3

        return adj

    ranked = []
    for s, r in matched:
        base = float(s)
        penalty = _soft_pref_adjust(r, answers)
        score = base + penalty - _bonus(r)
        rr = r.copy()                 
        rr["_final_sc"] = float(score)
        ranked.append((score, rr))

    ranked.sort(key=lambda x: x[0])     
    return ranked[:top_n]

def rag_generate_answer(rows, user_query=""):
    explanations = []
    for i, row in enumerate(rows, 1):
        eng_l  = row.get('engine_l')
        eng_cc = row.get('engine_cc')
        engine_line = f"{float(eng_l):.1f} L" if pd.notna(eng_l) else "---"
        if pd.notna(eng_cc):
            try:
                engine_line = (engine_line if engine_line != "---" else "") + f" ({int(eng_cc)} cc)"
                engine_line = engine_line.strip()
            except Exception:
                pass

        ctx = (
            f"รุ่น: {row.get('full_name','-')}\n"
            f"ซีรีส์/รุ่นย่อย: {row.get('series','-')}\n"
            f"ปี: {row.get('year','-')}\n"
            f"ราคา: {int(row['price_thb']):,} บาท\n"
            f"เครื่องยนต์: {engine_line}\n"
            f"แรงม้า: {row.get('horsepower_hp','---')}\n"
            f"เชื้อเพลิง: {row.get('fuel_type','---')}\n"
            f"รายละเอียด: {row.get('description','')}\n"
        )

        prompt = (
            f'จากข้อมูลรถ:\n{ctx}\nผู้ใช้ต้องการ: "{user_query}"\n'
            "ช่วยสรุปแบบภาษาคนคุยกัน เป็น 2–3 ประโยค อ่านง่าย ตรงประเด็น "
            "อธิบายว่ารุ่นนี้เด่น/เหมาะเพราะอะไร พร้อมข้อสังเกตสั้น ๆ หากมี "
            "ยึดจากข้อมูลข้างบนเท่านั้น"
        )
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "คุณคือนักขายรถที่อธิบายเก่ง พูดเป็นกันเอง อิงข้อมูลที่ให้เท่านั้น"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=700,
            )
            explanations.append((resp.choices[0].message.content or "").strip())
        except Exception as e:
            print(f"RAG GPT error on row {i}: {e}")
            explanations.append("ไม่สามารถสร้างคำอธิบายได้")
    return explanations

def llm_followup_answer(user_input, last_results):
    def _eng_text(r):
        eng_l  = r.get('engine_l')
        eng_cc = r.get('engine_cc')
        t = f"{eng_l}L" if eng_l not in (None, "", "nan") else "-"
        if eng_cc not in (None, "", "nan"):
            try:
                t = (t if t != "-" else "") + f" ({int(eng_cc)} cc)"
                t = t.strip()
            except Exception:
                pass
        return t

    lines = []
    for i, r in enumerate(last_results, 1):
        line = (
            f"{i}. {r.get('full_name')} | ปี {r.get('year','-')} | ราคา {r.get('price_thb')} | "
            f"เครื่อง {_eng_text(r)} | แรงม้า {r.get('horsepower_hp','-')} | เชื้อเพลิง {r.get('fuel_type','-')} | "
            f"เกียร์ {r.get('gears','-')} | ขับเคลื่อน {r.get('drive','-')}"
        )
        lines.append(line)

    context = "สรุปรถ 5 รุ่นล่าสุด (ย่อ):\n" + "\n".join(lines)

    prompt = (
        f"{context}\n\n"
        f"คำถามผู้ใช้: \"{user_input}\"\n\n"
        "คำสั่งการเขียนคำตอบ (สำคัญมาก):\n"
        "1) ตอบเป็นภาษาไทยแบบคุยธรรมชาติ สั้น กระชับ ไม่เป็นรายงานยาว ๆ\n"
        "2) ถ้าคำถามมีรูปแบบ “คันไหน…ที่สุด/ดีกว่า” ให้เลือกมา **1 คัน** ที่เหมาะสุด พร้อมเหตุผลสั้น ๆ 2–4 ข้อ\n"
        "3) ถ้าผู้ใช้ระบุว่า “เทียบ/เปรียบเทียบ/ทั้งหมด/ทั้ง 5 คัน” ค่อยสรุบเทียบแบบหัวข้อสั้น ๆ ได้\n"
        "4) อิงเฉพาะข้อมูลใน 5 คันนี้เท่านั้น ห้ามอ้างรุ่นอื่นหรือเติมข้อมูลที่ไม่มี\n"
        "5) ปิดท้ายด้วยบรรทัดสรุปสั้น ๆ แนะแนวว่า “เหมาะกับใคร/สถานการณ์ไหน”\n"
    )

    if not client:
        return "ตอนนี้จะตอบแบบย่อให้ก่อนนะครับ: จากชุดข้อมูลล่าสุด ผมจะเลือกคันที่เด่นสุดตามโจทย์และอธิบายเหตุผลสั้น ๆ ให้ครับ"

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "คุณคือที่ปรึกษารถยนต์ที่คุยเป็นกันเอง ตอบสั้นเป็นธรรมชาติ "
                        "ไม่ลิสต์ยาวเกินจำเป็น เว้นแต่ผู้ใช้ต้องการเปรียบเทียบละเอียด "
                        "อ้างอิงเฉพาะข้อมูลที่ให้เท่านั้น หากข้อมูลไม่พอให้บอกอย่างซื่อสัตย์"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,
            max_tokens=1000,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print("LLM followup error:", e)
        return "ขออภัยครับ เกิดข้อผิดพลาดในการสรุปคำตอบ ลองถามอีกครั้งได้ครับ"

def llm_detail_answer(user_input: str, row: dict) -> str:
    eng_l  = row.get('engine_l')
    eng_cc = row.get('engine_cc')
    engine_line = f"{float(eng_l):.1f} L" if eng_l not in (None, "", "nan") else "-"
    if eng_cc not in (None, "", "nan"):
        try:
            engine_line = (engine_line if engine_line != "-" else "") + f" ({int(eng_cc)} cc)"
            engine_line = engine_line.strip()
        except Exception:
            pass

    ctx = (
        f"รุ่น: {row.get('full_name')}\n"
        f"ซีรีส์/รุ่นย่อย: {row.get('series','-')}\n"
        f"ปี: {row.get('year','-')}\n"
        f"ราคา: {row.get('price_thb')} บาท\n"
        f"เครื่องยนต์: {engine_line}\n"
        f"แรงม้า: {row.get('horsepower_hp')}\n"
        f"เชื้อเพลิง: {row.get('fuel_type')}\n"
        f"เกียร์: {row.get('gears')}\n"
        f"ขับเคลื่อน: {row.get('drive')}\n"
        f"รายละเอียด: {row.get('description','')}\n"
    )
    prompt = (
        "สรุปให้เป็นหัวข้อสั้น: จุดเด่นหลัก, ข้อจำกัด, การใช้งานที่เหมาะ (ในเมือง/ทางไกล/ครอบครัว). "
        "อ้างจากข้อมูลที่ให้เท่านั้น ปิดท้ายด้วย **เหมาะกับใคร**.\n\n"
        f"{ctx}\nคำถามผู้ใช้: {user_input}"
    )
    if not client:
        return "สรุปสั้น ๆ: เหมาะการใช้งานทั่วไป ประหยัด ดูแลง่าย"
    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "ผู้เชี่ยวชาญรถยนต์ ตอบสั้น กระชับ จากข้อมูลที่ให้"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=500,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print("LLM detail error:", e)
        return "สรุปสั้น ๆ: เหมาะการใช้งานทั่วไป ประหยัด ดูแลง่าย"

@app.route("/")
def home():
    return render_template("index.html")

START_TRIGGERS = [
    "ช่วยแนะนำรถ",
    "เริ่มใหม่",
    "เริ่มต้น",
    "เริ่มแนะนำ",
    "start",
    "restart",
    "เริ่มต้นใหม่",
    "สวัสดี ช่วยแนะนำรถ",
]

def is_new_start(text: str, answers: dict) -> bool:
    triggers = [
        "เริ่มใหม่", "เริ่มต้นใหม่", "เริ่มต้นการค้นหาใหม่",
        "เริ่มระบบใหม่", "อยากเริ่มใหม่", "อยากได้คำแนะนำใหม่",
        "อยากให้แนะนำใหม่", "แนะนำใหม่", "หารถใหม่ให้หน่อย",
        "อยากหารถใหม่", "ขอเริ่มใหม่", "ขอคำแนะนำใหม่"
    ]
    text = text.lower().strip()
    return any(t in text for t in triggers)

def no_answer(text: str) -> bool:
    t = (text or "").strip().lower()
    return any(w in t for w in NO_WORDS)

WELCOME = "ยินดีต้อนรับสู่ระบบแนะนำรถยนต์ครับ  ผมพร้อมช่วยคุณหารถที่เหมาะกับคุณ!"
FIRST_Q = "เริ่มจากงบประมาณก่อนนะครับ — ตั้งไว้ประมาณเท่าไหร่ดี?"

@app.route("/chat", methods=["POST"])
def chat():

    has_greeted = bool(session.get("has_greeted"))
    data = request.get_json(force=True) or {}
    user_input = (data.get("message") or "").strip()
    ui = user_input.lower()
    handled_pending = False

    RESET_ALL = ["เริ่มใหม่", "เริ่มต้นใหม่", "รีเซ็ต", "reset", "เริ่มระบบใหม่"]
    NEW_RECO  = ["ขอใหม่", "หาใหม่", "แนะนำอีกที", "ขอคำแนะนำใหม่", "สุ่มใหม่", "อีก 5 คัน", "แนะนำชุดใหม่","ขอคำแนะนำใหม่"]

    if data.get("reset") is True or any(t in ui for t in (RESET_ALL + NEW_RECO)):
        reset_state_all()
        session["has_greeted"] = True
        try:
            first_q = (choose_natural_question(
                "price", get_prefs().get("answers", {}), get_prefs()
            ) or FIRST_Q).strip()
        except Exception:
            first_q = FIRST_Q
        first_q = format_question_for_ui(first_q, get_prefs().get("answers", {}))
        prefs = get_prefs() or {}
        prefs["pending_key"] = "price"
        prefs["last_question"] = first_q
        save_prefs(prefs)
        return jsonify({"mode": "intro", "reply": WELCOME, "next": first_q})


    prefs = get_prefs()
    answers = prefs["answers"]
    prefs.setdefault("_skip", [])


    if any(t in ui for t in (RESET_ALL + NEW_RECO)):
        reset_state_all()
        session["has_greeted"] = True
        if has_greeted:
            return jsonify({"mode": "ask", "reply": "โอเค มาใส่ข้อมูลใหม่กันครับ — ตั้งงบไว้ประมาณเท่าไหร่?"})
        else:
            try:
                first_q = (choose_natural_question("price", get_prefs().get("answers", {}), get_prefs()) or FIRST_Q).strip()
            except Exception:
                first_q = FIRST_Q
            prefs = get_prefs() or {}
            prefs["pending_key"] = "price"
            prefs["last_question"] = first_q
            save_prefs(prefs)
            first_q = format_question_for_ui(first_q, answers)
            return jsonify({"mode":"intro","reply":welcome,"next":first_q})



    prefs = get_prefs()
    answers = prefs["answers"]
    prefs.setdefault("_skip", [])
    
    if is_new_start(user_input, answers):
        prefs = {"answers": {}, "asked": [], "pending_key": "price", "last_question": None, "_skip": []}
        save_prefs(prefs)

        welcome = "ยินดีต้อนรับสู่ระบบแนะนำรถยนต์ครับ ถ้าคุณต้องการคำแนะนำ ผมสามารถช่วยคุณค้นหารถที่เหมาะกับคุณได้ครับ 😊"
        try:
            q = (choose_natural_question("price", get_prefs().get("answers", {}), get_prefs()) or FIRST_Q).strip()
        except Exception:
            q = FIRST_Q
        prefs = get_prefs() or {}
        prefs["pending_key"] = "price"
        prefs["last_question"] = q
        save_prefs(prefs)
        return jsonify({
            "mode": "intro",
            "reply": welcome,
            "next": q
        })


    if prefs.get("pending_key"):
        key = prefs["pending_key"].lower()
        ui = user_input.lower()
        prefs.setdefault("_skip", [])

        if no_answer(ui):
            if key not in prefs["_skip"]: prefs["_skip"].append(key)
            prefs["pending_key"] = None
            save_prefs(prefs)
        else:
            if key in ("budget", "price"):
                pmin, pmax = extract_price_range(ui)
                if re.search(r"(งบ\s*)?(เกิน|มากกว่า|>)+\s*\d", ui.replace(" ", "")):
                    answers["price"] = (None, None) 
                else:
                    answers["price"] = (pmin, pmax) if (pmin or pmax) else answers.get("price")
            elif key == "make":
                result = extract_make_from_query(user_input)
                if result:
                    if "make" in result:
                        answers["make"] = result["make"]
                    if "series" in result:         
                        answers["series"] = result["series"]
                else:
                    answers["make"] = answers.get("make")
            elif key in ("trans", "transmission"):
                tr = _extract_transmission(ui)
                answers["trans"] = tr or answers.get("trans")
            elif key == "fuel":
                fu = _extract_fuel(ui)
                answers["fuel"] = fu or answers.get("fuel")
            elif key in ("drive", "drivetrain", "ขับเคลื่อน"):
                if "4wd" in ui or "awd" in ui or "ขับสี่" in ui:
                    answers["drive"] = "4WD/AWD"
                elif "ขับหน้า" in ui or "fwd" in ui:
                    answers["drive"] = "FWD"
                elif "ขับหลัง" in ui or "rwd" in ui:
                    answers["drive"] = "RWD"
            elif key in ("usage", "usage_text"):
                answers["usage_text"] = user_input.strip()
                btxt = answers["usage_text"].lower()
                for _body, _kws in BODY_HINTS.items():
                    if any(w in btxt for w in _kws):
                        answers["body"] = _body
                        break
            elif key == "extra":
                txt = (user_input or "").strip()
                lo  = txt.lower()
                neg = {"ไม่มี", "ไม่", "no", "none"}
                if txt and lo not in neg:
                    prev = (answers.get("extra_text") or "").strip()
                    answers["extra_text"] = (prev + " " + txt).strip() if prev else txt     

                    m_hp = re.search(r'(\d{2,4})\s*(แรงม้า|hp)\b|\b(แรงม้า)\s*(\d{2,4})', lo)
                    if m_hp:
                        val = m_hp.group(1) or m_hp.group(4)
                        answers["min_hp"] = max(int(answers.get("min_hp") or 0), int(val))
                    m_cc = re.search(r"(\d{3,4})\s*cc", lo)
                    if m_cc:
                        answers["min_cc"] = max(int(answers.get("min_cc") or 0), int(m_cc.group(1)))
            elif key == "model":
                mk = extract_make_from_query(user_input)
                if isinstance(mk, dict):
                    if "make" in mk:
                        answers["make"] = mk["make"]
                    if "series" in mk:        
                        answers["series"] = mk["series"]
                prefs.setdefault("_skip", [])
                if "model" not in prefs["_skip"]:
                    prefs["_skip"].append("model")

        prefs.setdefault("asked", []).append(key)
        prefs["answers"] = answers
        prefs["pending_key"] = None
        handled_pending = True
        save_prefs(prefs)

    pmin_now, pmax_now = extract_price_range(user_input)
    if pmin_now or pmax_now:
        answers["price"] = (pmin_now, pmax_now)

    mk_now = extract_make_from_query(user_input)
    if isinstance(mk_now, dict):
        if "make" in mk_now:
            answers["make"] = mk_now["make"]
        if "series" in mk_now:
            answers["series"] = mk_now["series"]
    elif mk_now:
        answers["make"] = mk_now


    save_prefs(prefs)

    if prefs.get("stage") == "results" and session.get("recent_ids") and not prefs.get("pending_key"):
        ids = session["recent_ids"]
        cols = [
            "full_name","price_thb","engine_l","engine_cc","horsepower_hp",
            "fuel_type","gears","drive","brakes","series","description"
        ]
        try:
            last5 = df.loc[ids, cols].to_dict(orient="records")
        except Exception:
            last5 = df.iloc[ids][cols].to_dict(orient="records")

        reply_text = llm_followup_answer(user_input, last5)
        return jsonify({"mode": "followup", "reply": reply_text, "results": []})
    if prefs.get("stage") != "results":
        if not handled_pending:
            prefs["answers"] = extract_answers(user_input, prefs["answers"])
            answers = prefs["answers"]
            save_prefs(prefs)
        else:
            answers = prefs["answers"]
    else:
        answers = prefs["answers"]

    prefs.setdefault("extra_done", False)  

    core_fields = ["make", "usage_text", "trans", "price", "fuel", "drive"]
    skip = set(prefs.get("_skip", []))
    core_flags = {k: (bool(answers.get(k)) or k in skip) for k in core_fields}
    ready = all(core_flags.values())

    print("DEBUG core_flags:", core_flags, "ready:", ready, "extra_done:", prefs.get("extra_done"))

    if not ready:
        plan = next_question(user_input, answers)
        ask_for = (plan.get("ask_for") or "usage_text").strip().lower()
        follow_q = choose_natural_question(ask_for, answers, prefs).strip()

        if (
            (not follow_q)
            or (follow_q == prefs.get("last_question"))
            or (len(follow_q.split()) > 20)
        ):
            missing = _next_missing_field(answers) or "usage"
            ask_for  = missing
            follow_q = choose_natural_question(ask_for, answers, prefs).strip() or _fallback_question(missing)
        prefs["pending_key"] = ask_for
        prefs["last_question"] = follow_q
        save_prefs(prefs)
        follow_q = format_question_for_ui(follow_q, answers)
        return jsonify({"mode": "ask", "reply": follow_q})


    if ready and not prefs.get("extra_done"):
        q = choose_natural_question("extra", answers, prefs)  
        prefs["pending_key"] = "extra"
        prefs["last_question"] = q
        prefs["extra_done"] = True
        save_prefs(prefs)
        return jsonify({"mode": "ask", "reply": q})
   
    if prefs.get("stage") == "results" and not prefs.get("pending_key"):
        ids = session.get("recent_ids") or []
        last_results = []
        if ids:
            cols = [
            "full_name","price_thb","engine_l","engine_cc","horsepower_hp",
            "fuel_type","gears","drive","brakes","description"
        ]
            try:
                last_results = df.loc[ids, cols].to_dict(orient="records")
            except Exception:
                last_results = df.iloc[ids][cols].to_dict(orient="records")


        ref_idx = detect_reference(user_input, last_results)
        if ref_idx is not None:
            ans = llm_detail_answer(user_input, last_results[ref_idx])
            return jsonify({"mode": "followup", "reply": ans})

        if last_results and is_efficiency_question(user_input):
            def score_eff(r):
                try:
                    eng = float(r.get("engine_l") or 99)
                except Exception:
                    eng = 99
                fuel = (r.get("fuel_type") or "").lower()
                bonus = 0
                if any(k in fuel for k in ["hybrid", "hev", "phev", "bev", "ev"]):
                    bonus -= 1.0
                if "diesel" in fuel:
                    bonus -= 0.2
                return eng + bonus

            best = sorted(last_results, key=score_eff)[0]
            msg = (
                f"จากชุดล่าสุด คันที่ประหยัดน้ำมันสุดคือ “{best.get('full_name')}” "
                f"(เครื่อง {best.get('engine_l','-')}L / เชื้อเพลิง {best.get('fuel_type','-')}). "
                f"อยากทราบรายละเอียดเพิ่มไหม?"
            )
            return jsonify({"mode": "followup", "reply": msg})

        if last_results and is_compare_intent(user_input):
            follow = llm_followup_answer(user_input, last_results)
            return jsonify({"mode": "followup", "reply": follow})

        follow = rag_answer_followup(user_input, last_results)
        return jsonify({"mode": "followup", "reply": follow})


    if any(word in user_input for word in ["ไหม", "ดีไหม", "เหมาะไหม", "คุ้มไหม", "หรือเปล่า", "ใช่ไหม"]):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "คุณคือผู้ช่วยด้านรถยนต์ พูดคุยเป็นกันเอง ตอบตรงประเด็น อิงเหตุผลที่เข้าใจง่าย"},
                    {"role": "user", "content": user_input},
                ],
                temperature=0.7,
                max_tokens=600,
            )
            reply = (resp.choices[0].message.content or "").strip()
            return jsonify({"mode": "followup", "reply": reply})
        except Exception as e:
            print("Open-style answer error:", e)
 
    price_min = price_max = None
    if answers.get("price"):
        price_min, price_max = answers["price"]
    target_make = answers.get("make")
    if answers.get("make") and not answers.get("series"):
        ui_lc = user_input.lower().strip()
        mk = str(answers["make"]).lower().strip()
        m = re.search(rf"\b{re.escape(mk)}\s+([a-z0-9ก-๙:.-]+)", ui_lc)
        if m:
            cand = (m.group(1) or "").strip()
            root = cand.split()[0]
            bad = {"กับ", "และ", "หรือ", "and", "or"}
            if root not in bad and re.search(r"[a-z0-9ก-๙]", root) and len(root) >= 2:
                answers["series"] = root
    if answers.get("series"):
        sr = str(answers["series"]).lower()
        if not df["series"].fillna("").astype(str).str.lower().str.contains(sr).any():
            answers.pop("series", None)
    _series = str((answers or {}).get("series", "")).strip().lower()
    s_norm  = _series.replace(" ", "")  
    if _series in {"ไม่มี", "ไม่ระบุ", "อะไรก็ได้", "รุ่นไหนก็ได้", "แล้วแต่", "ไม่กำหนด"} \
    or s_norm in {"อะไรก็ได้".replace(" ", ""), "รุ่นไหนก็ได้".replace(" ", ""), "ไม่กำหนด"}: 
        answers.pop("series", None)
    _body_from_now = None
    btxt_now = f"{answers.get('usage_text','')} {user_input}".lower()
    for _b, _kws in BODY_HINTS.items():
        if any(w in btxt_now for w in _kws):
            _body_from_now = _b
            break
    if _body_from_now:
        answers["body"] = _body_from_now

    combined_query = " ".join([
    user_input,
    answers.get("make", ""),
    answers.get("series", ""),
    answers.get("usage_text", "")
    ]).strip()

    

    results = search_similar_rows(
        user_query=combined_query,
        price_min=price_min,
        price_max=price_max,
        target_make=target_make,
        target_series=answers.get("series"),
        top_n=5,
        usage_hints=answers.get("usage", []),
        answers=answers,
    )


    top_rows = [row for _, row in results]
    if price_max is not None:
        top_rows = [
            r for r in top_rows if float(r.get("price_thb", 0)) <= float(price_max)
        ]
  
    if "min_cc" in answers:
        top_rows = [
            r for r in top_rows
            if r.get("engine_cc") and int(r["engine_cc"]) >= answers["min_cc"]
    ]
    if answers.get("series"):
        sr = str(answers["series"]).lower()
        top_rows = [r for r in top_rows if sr in str(r.get("series", "")).lower()]

    print("=== DEBUG FINAL (exactly the same as UI) ===")
    for i, r in enumerate(top_rows[:5], 1):
        try:
            name  = str(r.get("full_name") or "")
           
            price = float(r.get("price_thb") or 0)
            print(f"FINAL >> {i}. {name} |  ราคา: {price:,.0f} บาท")
        except Exception:
            pass
    gpt_explanations = rag_generate_answer(top_rows, user_input) if top_rows else []

    rendered, serializable = [], []
    for i, row in enumerate(top_rows):
        eng_l = row.get("engine_l", np.nan)
        cc    = row.get("engine_cc", np.nan)
        hp_raw = row.get("horsepower_hp", np.nan)
        fuel_raw = row.get("fuel_type", "")
        gears_raw = row.get("gears", "")

        engine_text = f"{float(eng_l):.1f} L" if pd.notna(eng_l) else "---"
        if pd.notna(cc):
            try:
                engine_text = (engine_text if engine_text != "---" else "") + f" ({int(cc)} cc)"
                engine_text = engine_text.strip()
            except Exception:
                pass

        hp_val = pd.to_numeric(hp_raw, errors="coerce")
        if pd.isna(hp_val):
            m = re.search(r"(\d+)", str(hp_raw))
            hp_val = int(m.group(1)) if m else np.nan
        hp_text = f"{int(hp_val)} แรงม้า" if pd.notna(hp_val) else "---"

        fuel_text = fuel_to_thai(fuel_raw) or (str(fuel_raw).strip() or "---")

        if pd.isna(gears_raw) or str(gears_raw).strip() == "":
            gears_text = "---"
        else:
            gears_num = pd.to_numeric(gears_raw, errors="coerce")
            gears_text = f"{int(gears_num)} สปีด" if pd.notna(gears_num) else str(gears_raw).strip()

        rendered.append(
            {
                "rank": i + 1,
                "name": row["full_name"],
                "price": int(row["price_thb"]),
                "description": row.get("description", ""),
                "engine_text": engine_text,                 

                "hp_text": hp_text,
                "fuel_text": fuel_text,
                "gears_text": gears_text,
                "brakes_text": str(row.get("brakes", "")).strip() or "---",
                "drive_text": (str(row.get("drive", "")).strip().upper() or "---"),
                "year": safe_int(row.get("year")),         
                "series": (str(row.get("series")).strip() if row.get("series") is not None else None),   
                "ai_explanation": (gpt_explanations[i] if i < len(gpt_explanations) else ""),
            }
        )

        serializable.append(
            {
                "full_name": (str(row.get("full_name")) if row.get("full_name") is not None else None),
                "price_thb": safe_int(row.get("price_thb")),
                "engine_l":  safe_float(row.get("engine_l")),
                "engine_cc": safe_int(row.get("engine_cc")),    
                "horsepower_hp": safe_int(row.get("horsepower_hp")),
                "fuel_type": (str(row.get("fuel_type")) if row.get("fuel_type") is not None else None),
                "gears": parse_gears(row.get("gears")),
                "drive": (str(row.get("drive")) if row.get("drive") is not None else None),
                "brakes": (str(row.get("brakes")) if row.get("brakes") is not None else None),
                "year": safe_int(row.get("year")),               
                "series": (str(row.get("series")) if row.get("series") is not None else None),  
                "description": (str(row.get("description")) if row.get("description") is not None else None),
            }
        )

    try:
        session["recent_ids"] = [int(r.get("row_id")) for r in top_rows[:5]]
    except Exception:
        session["recent_ids"] = list(range(len(top_rows[:5])))

    try:
        session["last_results"] = [ int(r.get("row_id")) for r in top_rows[:5] ]
    except Exception:
       
        session["last_results"] = list(range(len(top_rows[:5])))
    prefs["stage"] = "results"
    prefs["pending_key"] = None
    save_prefs(prefs)


    if rendered:
        reply = f"เราขอแนะนำรถทั้ง {len(rendered)} คันนี้ ลองดูรายละเอียดด้านล่างได้เลยครับ"
    else:
        reply = "ขออภัยด้วยครับ ไม่มีรถที่มีสเปคที่คุณต้องการ แต่เราสามารถหาคำแนะนำใหม่ให้ได้ครับ"

    print("=== DEBUG FAISS (exactly the same as UI) ===")
    _qemb = model.encode([combined_query]).astype("float32")
    faiss.normalize_L2(_qemb)

    for i, r in enumerate(top_rows[:5], 1):
        try:
            sc = r.get("_final_sc")
            if sc is None:
                rid = int(r.get("row_id"))
                x = np.asarray(index.reconstruct(rid), dtype="float32")
                sc = float(((_qemb[0] - x) ** 2).sum())
            else:
                sc = float(sc)

            l2sq = sc
            cos  = 1.0 - (l2sq / 2.0)
            legacy = 18.0 + 1.75 * l2sq
            name = str(r.get("full_name") or "")
            print(f"DEBUG >> PF {i}. {name} | L2^2*: {l2sq:.4f} | cos*: {cos:.3f} | FAISS*(18–25): {legacy:.2f}")
        except Exception:
            pass


    return jsonify(
        json_safe({"mode": "recommend", "reply": reply, "results": rendered})
    )
    
    
@app.route("/followup", methods=["POST"])
def followup():
    data = request.get_json(force=True) or {}
    user_input = (data.get("message") or "").strip()
    last_results = session.get("last_results") or []
    if last_results and isinstance(last_results[0], (int, np.integer)):
        cols = [
            "full_name","price_thb","engine_l","engine_cc","horsepower_hp",
            "fuel_type","gears","drive","brakes","series","description"
        ]
        try:
            last_results = df.loc[last_results, cols].to_dict(orient="records")
        except Exception:
            last_results = df.iloc[last_results][cols].to_dict(orient="records")

    if not last_results:
        return jsonify(
            {"mode": "ask", "reply": "ยังไม่มีผลลัพธ์ก่อนหน้า พิมพ์ “ช่วยแนะนำรถ” เพื่อเริ่มใหม่ครับ"}
        )

    m = re.search(r"คันที่\s*(\d+)", user_input)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(last_results):
            car = last_results[idx]
            eng_line = f"{car.get('engine_l','-')}L"
            if car.get('engine_cc') not in (None, "", "nan"):
                try:
                    eng_line = (eng_line if eng_line != "-" else "") + f" ({int(car.get('engine_cc'))} cc)"
                except Exception:
                    pass
            text = (
                f"รายละเอียด “{car.get('full_name')}” ปี {car.get('year','-')} / ซีรีส์ {car.get('series','-')} | "
                f"เครื่อง {eng_line} | "
                f"แรงม้า {car.get('horsepower_hp','-')} | "
                f"เชื้อเพลิง {car.get('fuel_type','-')} | เกียร์ {car.get('gears','-')}"
            )
            return jsonify({"mode": "followup", "reply": text})


    if is_efficiency_question(user_input):

        def score(r):
            try:
                eng = float(r.get("engine_l") or 9)
            except:
                eng = 9
            fuel = (r.get("fuel_type") or "").lower()
            bonus = -1 if any(k in fuel for k in ["hev", "hybrid", "ev", "phev"]) else 0
            return eng + bonus

        best = sorted(last_results, key=score)[0]
        return jsonify(
            {
                "mode": "followup",
                "reply": f"คันที่ดูประหยัดสุดจากชุดล่าสุดคือ “{best.get('full_name')}” ครับ",
            }
        )

    return jsonify(
        {
            "mode": "followup",
            "reply": "พิมพ์เช่น “คันที่ 1” หรือ “คันไหนประหยัดสุด” เพื่อถามต่อจาก 5 คันล่าสุดครับ",
        }
    )

if __name__ == "__main__":
    app.run(debug=True)
