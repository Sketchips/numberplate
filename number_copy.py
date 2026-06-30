import cv2
import time
import argparse
from collections import Counter, deque
import numpy as np
import threading
import re

try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False

# =====================================================================
# KAMUS DATA KODE WILAYAH INDONESIA
# =====================================================================
KODE_WILAYAH = {
    "A": "Banten / Serang", "B": "Jakarta / Depok / Tangerang / Bekasi",
    "D": "Bandung / Cimahi", "E": "Cirebon / Majalengka",
    "F": "Bogor / Cianjur / Sukabumi", "G": "Pekalongan / Tegal",
    "H": "Semarang / Salatiga", "K": "Pati / Kudus / Jepara",
    "L": "Surabaya", "M": "Madura", "N": "Malang / Probolinggo",
    "P": "Besuki / Banyuwangi", "R": "Banyumas / Purwokerto",
    "S": "Bojonegoro / Lamongan", "T": "Purwakarta / Karawang",
    "W": "Sidoarjo / Gresik", "X": "Kendaraan Dinas",
    "Z": "Garut / Tasikmalaya / Sumedang",
    "AA": "Kedu / Magelang / Kebumen", "AB": "Yogyakarta",
    "AD": "Surakarta / Solo", "AE": "Madiun / Ngawi",
    "AG": "Kediri / Blitar", "BA": "Sumatera Barat",
    "BB": "Sumatera Utara (Tapanuli)", "BD": "Bengkulu",
    "BE": "Lampung", "BG": "Sumatera Selatan / Palembang",
    "BH": "Jambi", "BK": "Sumatera Utara / Medan",
    "BL": "Aceh", "BM": "Riau", "BN": "Bangka Belitung",
    "BP": "Kepulauan Riau", "DK": "Bali", "EA": "Nusa Tenggara Barat / Sumbawa",
    "EB": "Nusa Tenggara Timur / Flores", "ED": "Sumba",
    "KB": "Kalimantan Barat", "KH": "Kalimantan Tengah",
    "KT": "Kalimantan Timur", "KU": "Kalimantan Utara",
    "DA": "Kalimantan Selatan", "DL": "Sangihe / Talaud",
    "DM": "Gorontalo", "DN": "Sulawesi Tengah",
    "DB": "Manado / Minahasa", "DD": "Sulawesi Selatan / Makassar",
    "DC": "Sulawesi Barat", "DT": "Sulawesi Tenggara",
    "DE": "Maluku Maluku", "DG": "Maluku Utara",
    "PA": "Papua", "PB": "Papua Barat"
}

def dapatkan_lokasi(prefiks):
    return KODE_WILAYAH.get(prefiks.upper(), "Lokasi Tidak Diketahui")

# =====================================================================
# PART 1: CORE OCR & REGEX VALIDATION
# =====================================================================
reader = None

def ensure_reader():
    global reader
    if HAS_EASYOCR and reader is None:
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    return reader

def all_plates_ocr(crop_img):
    """Membaca seluruh baris teks yang ada di plat, bukan cuma satu baris tertinggi."""
    ensure_reader()
    if reader is None:
        return []
    
    results = reader.readtext(crop_img)
    return results # Mengembalikan seluruh list teks terdeteksi

def validate_plate_indonesia(text):
    clean_text = text.replace(" ", "").upper()
    pattern = r"^([A-Z]{1,2})(\d{1,4})([A-Z]{1,3})$"
    match = re.match(pattern, clean_text)
    
    if match:
        return {
            "valid": True,
            "prefiks": match.group(1),
            "normalized": f"{match.group(1)} {match.group(2)} {match.group(3)}"
        }
    return {"valid": False, "prefiks": "", "normalized": text}

# =====================================================================
# PART 2: SMART TRACKING WITH MULTI-PREFIX AGGREGATION
# =====================================================================
def iou(a, b):
    x1, y1, w1, h1 = a
    x2, y2, w2, h2 = b
    xa, ya = max(x1, x2), max(y1, y2)
    xb, yb = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter_w, inter_h = max(0, xb - xa), max(0, yb - ya)
    inter = inter_w * inter_h
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0

class Track:
    def __init__(self, tid, bbox, frame_idx):
        self.id = tid
        self.bbox = bbox
        self.last_seen = frame_idx
        self.stable = False
        self.is_processing_ocr = False
        self.last_result = "MENUNGGU..."
        self.ocr_attempt_count = 0 
        
        self.prefiks = ""   
        self.nomor = ""     
        self.sufiks = ""    

    def push_vote(self, text):
        if not text or len(text) < 1:
            return
        
        # Abaikan text jika mengandung format titik/strip pajak (contoh: 09.27 atau 09-27)
        if re.search(r"\d+[\.\-]\d+", text):
            return

        digits = "".join(re.findall(r"\d+", text))
        # Hanya ambil angka utama (1-4 digit) yang tidak diawali angka 0 jika panjangnya 4 digit (menghindari angka pajak)
        if len(digits) >= 1 and len(digits) <= 4:
            if not (len(digits) == 4 and digits.startswith('0')):
                self.nomor = digits

        letters = "".join(re.findall(r"[A-Z]+", text))
        
        if len(letters) > 0:
            if text[0].isalpha() and any(c.isdigit() for c in text):
                match_front = re.match(r"^([A-Z]{1,2})", text)
                if match_front: self.prefiks = match_front.group(1)
            
            elif text[-1].isalpha() and any(c.isdigit() for c in text):
                match_back = re.search(r"([A-Z]{1,3})$", text)
                if match_back: self.sufiks = match_back.group(1)
            
            elif text.isalpha():
                if len(text) <= 2:
                    self.prefiks = text
                elif len(text) >= 2 and not self.sufiks:
                    self.sufiks = text

        p = self.prefiks
        n = self.nomor
        s = self.sufiks
        
        if n:
            self.last_result = f"{p}{n}{s}".strip()
            if p and n and s: # Kunci stabil [OK] jika ketiga komponen lengkap
                self.stable = True
        else:
            self.last_result = p if p else "MENUNGGU..."

    def best(self):
        return self.last_result

# =====================================================================
# PART 3: MULTI-ANGLE ROTATION HAAR CASCADE
# =====================================================================
def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h))

def detect_with_haar(img, cascade_path="model/haarcascade_russian_plate_number.xml"):
    cascade = cv2.CascadeClassifier(cascade_path)
    all_rects = []
    
    # Pra-pemrosesan untuk mempertegas garis tepi plat yang miring
    smoothed = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    kernel_sharpening = np.array([[-1,-1,-1], 
                                  [-1, 9,-1], 
                                  [-1,-1,-1]])
    sharpened = cv2.filter2D(smoothed, -1, kernel_sharpening)
    
    # SOLUSI 1: Perlebar rentang sudut rotasi dari ekstrem kiri hingga kanan
    # Ini mendeteksi plat yang miring ke kiri/kanan (Roll) hingga 20 derajat
    angles = [0, -10, 10, -20, 20, -5, 5] 
    
    for angle in angles:
        rotated = sharpened.copy() if angle == 0 else rotate_image(sharpened, angle)
        gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Jendela deteksi diperketat (scaleFactor=1.03) agar pencarian plat miring lebih presisi
        rects = cascade.detectMultiScale(gray, scaleFactor=1.03, minNeighbors=2, minSize=(60, 18))
        
        for (x, y, w, h) in rects:
            # Jika gambar diputar, kembalikan estimasi koordinat kotak ke posisi semula (0 derajat)
            all_rects.append([x, y, w, h])
            
    # SOLUSI 2: JALUR CADANGAN (FALLBACK PERSPEKTIF)
    # Jika Haar Cascade sama sekali gagal karena miringnya sudut kamera (Pitch/Yaw)
    if len(all_rects) == 0:
        gray_fallback = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_fallback, (5, 5), 0)
        edged = cv2.Canny(blurred, 30, 200) # Temukan semua garis tepi
        
        # Cari kontur berbentuk kotak di area tengah bawah (rasio plat nomor)
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            # Jika kontur memiliki 4 sudut (segi empat), kemungkinan besar itu plat nomor
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                # Rasio plat nomor Indonesia berkisar antara 2.5 hingga 4.5
                if aspect_ratio >= 2.5 and aspect_ratio <= 4.5 and w > 60:
                    all_rects.append([x, y, w, h])
                    break # Ambil yang terbaik saja
                    
    if len(all_rects) > 0:
        rects_grouped, _ = cv2.groupRectangles(all_rects, groupThreshold=1, eps=0.2)
        return rects_grouped
        
    return all_rects

def async_ocr_worker(track, crop_img, on_finish_callback):
    t_start = time.time()
    try:
        if HAS_EASYOCR:
            gray_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            
            target_h = 90 # Naikkan resolusi crop sedikit agar text utama kontras
            h, w = gray_crop.shape[:2]
            scale = target_h / h
            target_w = int(w * scale)
            resized_gray = cv2.resize(gray_crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
            enhanced = clahe.apply(resized_gray)
            
            # Pindai menggunakan deteksi multi-baris
            ocr_results = all_plates_ocr(enhanced)
            
            for res in ocr_results:
                raw_text = res[1].strip().upper()
                # Bersihkan karakter aneh kecuali strip dan titik untuk deteksi filter pajak
                clean_text = "".join([c for c in raw_text if c.isalnum() or c in ['.', '-']])
                track.push_vote(clean_text)
                
    except Exception as e:
        print(f"[ERROR OCR]: {e}")
    
    track.ocr_attempt_count += 1
    track.is_processing_ocr = False
    on_finish_callback(track.id, track.stable)

def run_realtime(source=0, process_every=2, max_age=45):
    if not HAS_EASYOCR:
        print("[ERROR] Library 'easyocr' tidak ditemukan.")
        return

    ensure_reader()
    print("==> Model Deteksi Plat Nomor Aktif!")

    cap = cv2.VideoCapture(source if isinstance(source, int) or source.isdigit() else source)
    tracks = {}
    next_id = 1
    frame_idx = 0
    active_track_id = None 

    def ocr_callback(tid, is_stable):
        nonlocal active_track_id
        if is_stable or (tid in tracks and tracks[tid].ocr_attempt_count >= 15):
            active_track_id = None # Lepas kunci antrean

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1

        if frame_idx % process_every == 0:
            rects = detect_with_haar(frame)
            best_tid, best_iou, matched_rect = None, 0, None

            for (x, y, rw, rh) in rects:
                if active_track_id in tracks:
                    i = iou((x, y, rw, rh), tracks[active_track_id].bbox)
                    if i > best_iou:
                        best_iou, best_tid, matched_rect = i, active_track_id, (x, y, rw, rh)
                else:
                    for tid, tr in tracks.items():
                        i = iou((x, y, rw, rh), tr.bbox)
                        if i > best_iou:
                            best_iou, best_tid, matched_rect = i, tid, (x, y, rw, rh)

            if best_iou > 0.05 and best_tid is not None:
                tr = tracks[best_tid]
                x, y, rw, rh = matched_rect
                ox, oy, ow, oh = tr.bbox
                tr.bbox = (int(ox*0.7 + x*0.3), int(oy*0.7 + y*0.3), int(ow*0.7 + rw*0.3), int(oh*0.7 + rh*0.3))
                tr.last_seen = frame_idx
            
            elif active_track_id is None and len(rects) > 0:
                x, y, rw, rh = rects[0]
                best_tid = next_id
                next_id += 1
                tracks[best_tid] = Track(best_tid, (x, y, rw, rh), frame_idx)
                active_track_id = best_tid

            if active_track_id is not None and active_track_id in tracks:
                current_track = tracks[active_track_id]
                if not current_track.is_processing_ocr and not current_track.stable:
                    tx, ty, tw, th = current_track.bbox
                    pad_w, pad_h = int(tw * 0.25), int(th * 0.10)
                    x0, y0 = max(0, tx - pad_w), max(0, ty - pad_h)
                    w0, h0 = min(frame.shape[1] - x0, tw + (pad_w * 2)), min(frame.shape[0] - y0, th + (pad_h * 2))
                    
                    crop = frame[y0:y0 + h0, x0:x0 + w0].copy()
                    if crop.size > 0:
                        current_track.is_processing_ocr = True
                        current_track.last_result = "MEMPROSES..."
                        t = threading.Thread(target=async_ocr_worker, args=(current_track, crop, ocr_callback))
                        t.daemon = True
                        t.start()

            to_del = []
            for tid, tr in list(tracks.items()):
                if frame_idx - tr.last_seen > max_age:
                    to_del.append(tid)
                    if tid == active_track_id: active_track_id = None
            for tid in to_del: del tracks[tid]

        for tid, tr in tracks.items():
            x, y, rw, rh = tr.bbox
            box_color = (0, 0, 255) if tid == active_track_id else (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + rw, y + rh), box_color, 2)
            
            label = tr.best()
            lokasi = ""
            
            if label not in ["MENUNGGU...", "TIDAK TERBACA"] and "MEMPROSES" not in label:
                valid = validate_plate_indonesia(label)
                label = f"{valid.get('normalized')}"
                if valid.get('prefiks'):
                    lokasi = f" ({dapatkan_lokasi(valid.get('prefiks'))})"

            status_str = "[OK]" if tr.stable else ("[PROSES]" if tr.is_processing_ocr else "[ANTRE]")
            display_text = f"ID:{tid} - {label} {status_str}{lokasi}"
            
            # Gambar teks overlay dua lapis (Drop shadow hitam tebal agar terbaca jelas)
            cv2.putText(frame, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("ANPR Indonesia - Gate Edition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_realtime(source=0)