from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import csv
import random
import re  # ← เพิ่มเพื่อช่วยทำความสะอาดข้อความเล็กน้อย


# ---------- CONFIG ----------

BASE_URL = "https://www.checkraka.com/car/mitsubishi/attrage/"
OUTPUT_FILE = "mitsubishi_attrage.csv"

# เดิมเป็น ["Model Name", "Price", "Details"] → เปลี่ยนเป็นคอลัมน์สเปกตามที่ต้องการ
HEADERS = ["Model Name", "engine_cc", "horsepower_hp", "gears", "fuel_type", "engine", "brakes", "drive"]

MIN_DELAY = 2.5
MAX_DELAY = 4.5
# ----------------------------

# ---------- SETUP ----------
options = Options()
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--disable-extensions")
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
wait = WebDriverWait(driver, 15)

def human_delay():
    time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

def get_model_links_and_info():
    print("เข้าหน้าเว็บหลักเพื่อดึงชื่อรุ่น + ราคา")
    driver.get(BASE_URL)
    human_delay()

    #ดูรุ่นย่อยทั้งหมด
    try:
        load_more_btn = wait.until(EC.element_to_be_clickable((By.ID, "btn-load-more")))
        driver.execute_script("arguments[0].scrollIntoView(true);", load_more_btn)
        time.sleep(1.5)  
        load_more_btn.click()
        print("ดูรุ่นย่อยทั้งหมด")
        time.sleep(3.5)
    except Exception as e:
        print("ℹไม่พบปุ่ม:", e)

    links_data = []
    try:
        product_cards = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.card.card-product")))
        print(f" พบ {len(product_cards)} รุ่นย่อย")
        for card in product_cards:
            try:
                name = card.find_element(By.CSS_SELECTOR, "a").get_attribute("title").strip()
                price = card.find_element(By.CSS_SELECTOR, "div.price").text.strip()  # (ยังคงไว้ ไม่ใช้งาน)
                link = card.find_element(By.CSS_SELECTOR, "a").get_attribute("href")
                links_data.append({"Model Name": name, "Price": price, "URL": link})
            except Exception:
                continue
    except Exception as e:
        print(f"ดึงข้อมูลรุ่นไม่ได้: {e}")
    return links_data


def scrape_details(url):
    try:
        driver.get(url)
        human_delay()
        try:
            spec_tab = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'สเปคละเอียด')]"))
            )
            driver.execute_script("arguments[0].click();", spec_tab)
            time.sleep(2.0)
        except Exception:
            print("⚠ ไม่เจอปุ่มสเปคละเอียด (ข้าม)")

        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div#tab2 div.tab-item")))

        spec_container = driver.find_element(By.CSS_SELECTOR, "div#tab2")

        def _clean(s: str) -> str:
            return re.sub(r"\s+", " ", s).strip()

        pairs = []
        rows = spec_container.find_elements(By.CSS_SELECTOR, "div.tab-item")
        for r in rows:
            try:
                k = r.find_element(By.CSS_SELECTOR, ".tab-title-item").text
                v = r.find_element(By.CSS_SELECTOR, ".tab-detail-item").text
                pairs.append((_clean(k), _clean(v)))
            except Exception:
                continue

        def pick(keys):
            for k, v in pairs:
                low = k.lower()
                if any(key in low for key in keys):
                    return v
            return ""
        engine       = pick(["เครื่องยนต์"])
        engine_cc    = pick(["cc", "ความจุ", "ขนาดเครื่องยนต์"])
        horsepower   = pick(["แรงม้า", "กำลังเครื่องยนต์", "กำลังสูงสุด", "แรงม้าสูงสุด"])
        gears        = pick(["ระบบเกียร์", "เกียร์"])
        fuel_type    = pick(["เชื้อเพลิง", "ประเภทน้ำมัน", "ประเภทเชื้อเพลิง"])
        brakes       = pick(["abs", "เบรก", "ระบบเบรก"])
        drive        = pick(["ขับเคลื่อน", "ระบบขับเคลื่อน"])
        return {
            "engine": engine,
            "engine_cc": engine_cc,
            "horsepower_hp": horsepower,
            "gears": gears,
            "fuel_type": fuel_type,
            "brakes": brakes,
            "drive": drive
            
        }

    except Exception as e:
        print(f"ดึงสเปคล้มเหลว {url}: {e}")
        return {"engine_cc": "", "horsepower_hp": "", "gears": "", "fuel_type": ""}


# MAIN 
try:
    model_data = get_model_links_and_info()
    for i, item in enumerate(model_data, 1):
        print(f"({i}/{len(model_data)}) กำลังดึงสเปค: {item['Model Name']}")
        specs = scrape_details(item["URL"])  
        item["engine"] = specs.get("engine", "")
        item["engine_cc"] = specs.get("engine_cc", "")
        item["horsepower_hp"] = specs.get("horsepower_hp", "")
        item["gears"] = specs.get("gears", "")
        item["fuel_type"] = specs.get("fuel_type", "")
        item["brakes"] = specs.get("brakes", "")
        item["drive"] = specs.get("drive", "")
        human_delay()

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writeheader()
        for row in model_data:
            writer.writerow({
                "Model Name": row["Model Name"],
                "engine": row.get("engine", ""),
                "engine_cc": row.get("engine_cc", ""),
                "horsepower_hp": row.get("horsepower_hp", ""),
                "gears": row.get("gears", ""),
                "fuel_type": row.get("fuel_type", ""),
                "brakes": row.get("brakes", ""),
                "drive": row.get("drive", "")
            })

    print(f"\n บันทึก {len(model_data)} รุ่นย่อย -> {OUTPUT_FILE}")

except Exception as e:
    print(f"เกิดข้อผิดพลาด: {e}")

finally:
    driver.quit()
