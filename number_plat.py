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
# PART 1: CORE OCR & REGEX VALIDATION
# =====================================================================
reader = None

def ensure_reader():
    global reader
    if HAS_EASYOCR and reader is None:
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    return reader

def best_plate_ocr(crop_img):
    ensure_reader()
    if reader is None:
        return "", 0.0, []
    
    results = reader.readtext(crop_img)
    if not results:
        return "", 0.0, []
    
    best_match = max(results, key=lambda x: x[2])
    return best_match[1], best_match[2], best_match[0]

def validate_plate_indonesia(text):
    clean_text = text.replace(" ", "").upper()
    pattern = r"^([A-Z]{1,2})(\d{1,4})([A-Z]{1,3})$"
    match = re.match(pattern, clean_text)
    
    if match:
        return {
            "valid": True,
            "normalized": f"{match.group(1)} {match.group(2)} {match.group(3)}"
        }
    return {"valid": False, "normalized": text}

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
        
        # Abaikan angka tahun pajak kecil (2 digit) jika terdeteksi sendirian di awal proses
        if len(text) <= 2 and text.isdigit():
            return

        # 1. Ekstrak Angka (Nomor Tengah Plat)
        digits = "".join(re.findall(r"\d+", text))
        if len(digits) >= 3 and len(digits) <= 4:
            self.nomor = digits

        # 2. Ekstrak Huruf (Kemungkinan Prefiks atau Sufiks)
        letters = "".join(re.findall(r"[A-Z]+", text))
        
        # Analisis posisi huruf berdasarkan teks mentah yang masuk
        if len(letters) > 0:
            # Jika teks diawali huruf lalu diikuti angka (misal: F6797)
            if text[0].isalpha() and any(c.isdigit() for c in text):
                match_front = re.match(r"^([A-Z]{1,2})", text)
                if match_front: self.prefiks = match_front.group(1)
            
            # Jika teks diawali angka dulu baru huruf di akhir (misal: 6797BB atau 1001ZZZ)
            elif text[-1].isalpha() and any(c.isdigit() for c in text):
                match_back = re.search(r"([A-Z]{1,3})$", text)
                if match_back: self.sufiks = match_back.group(1)
            
            # Jika murni huruf saja yang tertangkap kamera
            elif text.isalpha():
                if len(text) <= 2:
                    # Huruf pendek kemungkinan besar kode wilayah depan (B, F, D)
                    self.prefiks = text
                elif len(text) >= 2 and not self.sufiks:
                    # Huruf seri belakang
                    self.sufiks = text

        # Jembatan penggabung cerdas
        p = self.prefiks
        n = self.nomor
        s = self.sufiks
        
        if n:
            self.last_result = f"{p}{n}{s}".strip()
            # Kunci sukses [OK] jika minimal Angka dan salah satu Huruf sudah lengkap terkumpul
            if (p and n) or (n and s):
                if len(self.last_result) >= 5:
                    self.stable = True
        else:
            self.last_result = p if p else "MENUNGGU..."

    def best(self):
        return self.last_result

# =====================================================================
# PART 3: MULTI-ANGLE ROTATION HAAR CASCADE (SOLUSI PLAT MIRING)
# =====================================================================
def rotate_image(image, angle):
    """Memutar frame video untuk mencari plat nomor tersembunyi akibat miring."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h))

def detect_with_haar(img, cascade_path="model/haarcascade_russian_plate_number.xml"):
    cascade = cv2.CascadeClassifier(cascade_path)
    all_rects = []
    
    # Periksa dalam 3 variasi sudut pandang kamera (Normal, Miring Kiri, Miring Kanan)
    angles = [0, -12, 12] 
    
    for angle in angles:
        if angle == 0:
            rotated = img.copy()
        else:
            rotated = rotate_image(img, angle)
            
        gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # minNeighbors diatur ke 4 agar deteksi sudut miring lebih sensitif menyambar target
        rects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(80, 25))
        
        for (x, y, w, h) in rects:
            all_rects.append([x, y, w, h])
            
    # Satukan koordinat tumpang tindih dari berbagai rotasi sudut agar tidak double-box
    if len(all_rects) > 0:
        rects_grouped, _ = cv2.groupRectangles(all_rects, groupThreshold=1, eps=0.3)
        return rects_grouped
        
    return all_rects

def async_ocr_worker(track, crop_img, on_finish_callback):
    t_start = time.time()
    txt = ""
    try:
        if HAS_EASYOCR:
            # 1. Ubah ke Grayscale
            gray_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            
            # 2. Normalisasi Ukuran Tinggi ke 70px agar karakter proporsional
            target_h = 70
            h, w = gray_crop.shape[:2]
            scale = target_h / h
            target_w = int(w * scale)
            resized_gray = cv2.resize(gray_crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            
            # 3. PENGOLAHAN GAMBAR ADAPTIF (Penyelamat Huruf Pudar & Anti-Silau):
            # Tingkatkan kontras lokal terlebih dahulu agar karakter abu-abu menjadi lebih tegas
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            enhanced = clahe.apply(resized_gray)
            
            # Gunakan ADAPTIVE_THRESH_GAUSSIAN_C agar background silau diredam dan teks tipis dipertahankan
            # blockSize=11 dan C=5 adalah kombinasi paling aman untuk ukuran font plat nomor
            threshed = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 5
            )
            
            # Pastikan teks berwarna HITAM di atas dasar PUTIH sebelum diserahkan ke EasyOCR
            black_pixels = np.sum(threshed == 0)
            white_pixels = np.sum(threshed == 255)
            if black_pixels > white_pixels:
                threshed = cv2.bitwise_not(threshed)
            
            # Balikkan ke format 3-channel (BGR) untuk EasyOCR
            final_bgr = cv2.cvtColor(threshed, cv2.COLOR_GRAY2BGR)
            
            # 4. Jalankan Proses Pembacaan Teks
            ocr_results = best_plate_ocr(final_bgr)
            if isinstance(ocr_results, (tuple, list)) and len(ocr_results) >= 2:
                txt = ocr_results[0]
            else:
                txt = str(ocr_results)
                
            # Filter karakter non-alfanumerik
            txt = "".join([c for c in txt if c.isalnum()]).strip().upper()
            
            # Sistem Autokoreksi Typo Angka/Huruf Mirip Standar Indonesia
            if 'TOO' in txt: txt = txt.replace('TOO', '100')
            if 'IOO' in txt: txt = txt.replace('IOO', '100')
            
    except Exception as e:
        print(f"[ERROR OCR]: {e}")
        txt = ""
    
    elapsed = time.time() - t_start
    print(f"[THREAD OCR] Track #{track.id} Selesai dalam {elapsed:.2f} detik. Hasil Mentah: '{txt}'")
    
    track.push_vote(txt)
    track.ocr_attempt_count += 1
    track.is_processing_ocr = False
    
    on_finish_callback(track.id, track.stable)

def run_realtime(source=0, process_every=2, max_age=120):
    if not HAS_EASYOCR:
        print("[ERROR] Library 'easyocr' tidak ditemukan.")
        return

    ensure_reader()
    print("==> Model ANPR Siap Diluncurkan!")

    cap = cv2.VideoCapture(source if isinstance(source, int) or source.isdigit() else source)
    tracks = {}
    next_id = 1
    frame_idx = 0
    active_track_id = None 

    def ocr_callback(tid, is_stable):
        nonlocal active_track_id
        if is_stable or (tid in tracks and tracks[tid].ocr_attempt_count >= 20):
            print(f"[SYSTEM] Penguncian ID #{tid} selesai dibukukan.")
            active_track_id = None

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
                tr.bbox = (int(ox*0.8 + x*0.2), int(oy*0.8 + y*0.2), int(ow*0.8 + rw*0.2), int(oh*0.8 + rh*0.2))
                tr.last_seen = frame_idx
            
            elif active_track_id is None and len(rects) > 0:
                x, y, rw, rh = rects[0]
                best_tid = next_id
                next_id += 1
                tracks[best_tid] = Track(best_tid, (x, y, rw, rh), frame_idx)
                active_track_id = best_tid
                print(f"[SYSTEM] Mengunci ID baru: #{active_track_id}")

            if active_track_id is not None and active_track_id in tracks:
                current_track = tracks[active_track_id]
                if not current_track.is_processing_ocr and not current_track.stable:
                    tx, ty, tw, th = current_track.bbox
                    pad_w, pad_h = int(tw * 0.30), int(th * 0.12)
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
                    if tid == active_track_id:
                        print(f"[SYSTEM] Track #{tid} Lepas. Membuka Kunci Antrean.")
                        active_track_id = None
            for tid in to_del: del tracks[tid]

        for tid, tr in tracks.items():
            x, y, rw, rh = tr.bbox
            box_color = (0, 0, 255) if tid == active_track_id else (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + rw, y + rh), box_color, 2)
            
            label = tr.best()
            if label not in ["MENUNGGU...", "TIDAK TERBACA"] and "MEMPROSES" not in label:
                valid = validate_plate_indonesia(label)
                label = f"{valid.get('normalized')}"

            status_str = "[OK]" if tr.stable else ("[PROSES]" if tr.is_processing_ocr else "[ANTRE]")
            display_text = f"ID:{tid} - {label} {status_str}"
            
            cv2.putText(frame, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("ANPR Realtime - Multi Prefix Edition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_realtime(source=0)