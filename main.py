import cv2
import easyocr
import sqlite3
import re
from datetime import datetime
import os

class LicensePlateRecognizer:
    def __init__(self, db_name="license_plates.db"):
        self.db_name = db_name
        self.reader = easyocr.Reader(['en'], gpu=False)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detected_plates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate_text TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp DATETIME NOT NULL
                )
            ''')
            conn.commit()

    def preprocess_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image '{image_path}' not found.")
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        return img, filtered

    def extract_and_save(self, image_path):
        _, processed_img = self.preprocess_image(image_path)
        results = self.reader.readtext(processed_img)
        detected_entries = []

        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            for (_, text, prob) in results:
                clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                if len(clean_text) >= 4 and prob > 0.30:
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    cursor.execute('''
                        INSERT INTO detected_plates (plate_text, confidence, timestamp)
                        VALUES (?, ?, ?)
                    ''', (clean_text, round(prob, 4), current_time))
                    detected_entries.append((clean_text, prob, current_time))
            conn.commit()

        return detected_entries

if __name__ == "__main__":
    app = LicensePlateRecognizer()
    sample_image = "car.jpg"

    if os.path.exists(sample_image):
        print("Processing license plate recognition...")
        detections = app.extract_and_save(sample_image)
        for text, conf, ts in detections:
            print(f"Detected: {text} | Confidence: {conf*100:.1f}% | Time: {ts}")
    else:
        print(f"Please place a sample image named '{sample_image}' in the project directory.")
