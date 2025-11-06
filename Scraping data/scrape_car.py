from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import csv
import random


CHROMEDRIVER_PATH = r"D:\scrap data\chromedriver.exe"
BASE_URL = "https://www.checkraka.com/car/suzuki/swift/"
OUTPUT_FILE = "Suzuki_Swift.csv"
HEADERS = ["Model Name", "Price", "Details"]
MIN_DELAY = 2.5
MAX_DELAY = 4.5

options = Options()
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--disable-extensions")
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36")

driver = webdriver.Chrome(service=Service(CHROMEDRIVER_PATH), options=options)
wait = WebDriverWait(driver, 15)

def human_delay():
    time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

def get_model_links_and_info():
    print("เข้าหน้าเว็บหลักเพื่อดึงชื่อรุ่น + ราคา")
    driver.get(BASE_URL)
    human_delay()

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
                price = card.find_element(By.CSS_SELECTOR, "div.price").text.strip()
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
            readmore = driver.find_element(By.CSS_SELECTOR, "a.read-more")
            driver.execute_script("arguments[0].click();", readmore)
            time.sleep(1.2)
        except:
            pass

        detail_box = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.model-detail")))
        paragraphs = detail_box.find_elements(By.TAG_NAME, "p")
        return " ".join(p.text.strip() for p in paragraphs if p.text.strip())
    except Exception as e:
        print(f"ดึงรายละเอียดล้มเหลว {url}: {e}")
        return ""

try:
    model_data = get_model_links_and_info()
    for i, item in enumerate(model_data, 1):
        print(f"({i}/{len(model_data)}) กำลังดึงข้อมูล: {item['Model Name']}")
        item["Details"] = scrape_details(item["URL"])
        human_delay()

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writeheader()
        for row in model_data:
            writer.writerow({
                "Model Name": row["Model Name"],
                "Price": row["Price"],
                "Details": row["Details"]
            })

    print(f"\n บันทึก {len(model_data)} คัน  {OUTPUT_FILE}")

except Exception as e:
    print(f"เกิดข้อผิดพลาด: {e}")

finally:
    driver.quit()
