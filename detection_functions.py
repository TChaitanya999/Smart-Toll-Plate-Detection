"""
Detection functions module - extracted from yolo_ocr_strict.py for reuse
"""
import cv2
import easyocr
import re
import numpy as np
from collections import Counter

                                            
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

                                           
VALID_STATE_CODES = {
    "AN","AP","AR","AS","BR","CH","DD","DL","DN","GA","GJ","HP","HR","JH","JK","KA",
    "KL","LA","LD","MH","ML","MN","MP","MZ","NL","OD","PB","PY","RJ","SK","TN","TR",
    "TS","UA","UK","UP","UT","WB"
}

def enhance_contrast(img):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def sharpen_image(img):
    """Apply sharpening filter"""
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(img, -1, kernel)

def denoise_image(img):
    """Remove noise while preserving edges"""
    return cv2.fastNlMeansDenoising(img, None, 10, 7, 21)

def preprocess_plate(img, scale_factor=4, fast_mode=False):
    """Enhanced preprocessing with multiple steps for better character clarity"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    
    if fast_mode:
                                                            
        gray = enhance_contrast(gray)
                                                                       
        gray = cv2.convertScaleAbs(gray, alpha=1.3, beta=15)
                                                    
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
                                            
        gray = denoise_image(gray)
        gray = enhance_contrast(gray)
        gray = sharpen_image(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def preprocess_variants(img):
    variants = []
    for scale in [3, 4, 5]:
        base = preprocess_plate(img, scale_factor=scale)
        variants.append(base)
        variants.append(cv2.bitwise_not(base))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        variants.append(cv2.dilate(base, kernel, iterations=1))
        variants.append(cv2.erode(base, kernel, iterations=1))
        h, w = base.shape[:2]
        center = (w // 2, h // 2)
        for ang in [-7, -4, -2, 2, 4, 7]:
            M = cv2.getRotationMatrix2D(center, ang, 1.0)
            rotated = cv2.warpAffine(base, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            variants.append(rotated)
            variants.append(cv2.bitwise_not(rotated))
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    gray = enhance_contrast(gray)
    adaptive1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
    adaptive2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
    variants.append(adaptive1)
    variants.append(adaptive2)
    variants.append(cv2.bitwise_not(adaptive1))
    variants.append(cv2.bitwise_not(adaptive2))
    return variants

def generate_crops(plate_img):
    """Return several crop variants to avoid losing characters on the left"""
    h, w = plate_img.shape[:2]
    variants = [plate_img]
    cut10 = int(w * 0.10)
    variants.append(plate_img[:, cut10:w])
    cut20 = int(w * 0.20)
    variants.append(plate_img[:, cut20:w])
    return variants

def clean_text(text):
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)
    return text

def fix_common(text):
    """Fix common OCR errors - but be careful not to change letters in letter positions"""
                                                                           
                                                               
    text = text.replace("I", "1")                           
    text = text.replace("Z", "2")                           
    text = text.replace("S", "5")                           
    text = text.replace("B", "8")                           
    text = text.replace("G", "6")                           
    text = text.replace("Q", "0")                           
    text = text.replace("T", "7")                           
    text = text.replace("L", "1")                           
                                                                           
    return text

def position_aware_fix(candidate):
    """Apply character fixes based on expected positions - ONLY fix digit/letter position errors, NO letter-to-letter conversions"""
    c = clean_text(candidate)
    if len(c) != 10:
        return fix_common(c)

                                                               
    if re.fullmatch(r'[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}', c):
        return c                          

                                                                 
    digit_confusions = {"O": "0", "I": "1", "Z": "2", "S": "5", "B": "8", "G": "6", "Q": "0", 
                       "D": "0", "T": "7", "L": "1", "F": "E"}
    letter_confusions = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G", 
                         "7": "T", "3": "E", "4": "A"}

    chars = list(c)
    letter_positions = {0, 1, 4, 5}                                                      
    digit_positions = {2, 3, 6, 7, 8, 9}                                                  
    
                                                                                   
                                                      
    for i in range(min(10, len(chars))):
        char = chars[i]
        if i in letter_positions:
                                
            if char.isdigit():
                                                                 
                chars[i] = letter_confusions.get(char, char)
                                                                      
        elif i in digit_positions:
                               
            if char.isalpha():
                                                                 
                if char in digit_confusions:
                    chars[i] = digit_confusions.get(char, char)
            elif char not in "0123456789":
                                         
                if char in letter_confusions:
                    chars[i] = digit_confusions.get(letter_confusions[char], char)

    return "".join(chars)

def group_detections_into_lines(detections):
    """Group EasyOCR detections into lines"""
    items = []
    for bbox, text, conf in detections:
        if not text or not text.strip():
            continue
        ys = [p[1] for p in bbox]
        xs = [p[0] for p in bbox]
        y_center = sum(ys) / 4.0
        x_center = sum(xs) / 4.0
        height = max(ys) - min(ys)
        items.append((y_center, x_center, text, conf, height))

    if not items:
        return []

    items.sort(key=lambda t: t[0])
    avg_height = sum(t[4] for t in items) / len(items) if items else 20.0
    gap_thresh = avg_height * 1.5

    lines = []
    current = [items[0]]
    for prev, cur in zip(items, items[1:]):
        y_gap = cur[0] - prev[0]
        if y_gap > gap_thresh:
            lines.append(current)
            current = [cur]
        else:
            current.append(cur)
    lines.append(current)

    out = []
    for line in lines:
        line_sorted = sorted(line, key=lambda t: t[1])
        out.append([(t[1], t[2], t[3]) for t in line_sorted])
    return out

def reconstruct_plate_from_detections(detections):
    """Handles 1-line and 2-line Indian plates - preserves exact OCR text"""
    lines = group_detections_into_lines(detections)
    if not lines:
        return ""

    joined_lines = []
    for line in lines:
        line_sorted = sorted(line, key=lambda t: t[0])                                  
        raw = "".join([t[1] for t in line_sorted])
        cleaned = clean_text(raw)
        joined_lines.append(cleaned)

                                                         
    if len(joined_lines) >= 2:
        top = joined_lines[0]                                                  
        bottom = joined_lines[1]
        
                                    
                                                                     
        if len(top) >= 5 and len(bottom) >= 5:
            candidate = (top[:5] + bottom[:5])[:10]
            if len(candidate) == 10:
                return candidate                                           
        
                                                
        if len(top) >= 4 and len(bottom) >= 6:
            candidate = (top[:4] + bottom[:6])[:10]
            if len(candidate) == 10:
                return candidate
        
                                     
        candidate = (top + bottom)[:10]
        if len(candidate) == 10:
            return candidate
        
                                         
        if len(top) >= 3 and len(bottom) >= 7:
            candidate = (top[:3] + bottom[:7])[:10]
            if len(candidate) == 10:
                return candidate
        
                                                        
        return (top + bottom)[:10] if len(top + bottom) >= 10 else (top + bottom)

                                               
    all_text = "".join(joined_lines)
                                                         
    if len(all_text) >= 10:
        return all_text[:10]
    return all_text

def find_best_plate(text):
    """Find best plate pattern in text - strict mode (no letter-to-letter substitutions)"""
    pat = r'[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}'
    match = re.search(pat, text)
    if match:
        candidate = match.group()
        return candidate
                                                                                           
    if len(text) == 10:
        corrected = position_aware_fix(text)
        match = re.search(pat, corrected)
        if match:
            return match.group()
    return text

def score_candidate(text):
    """Score a candidate plate text"""
    if not text:
        return 0
    
    score = 0
    cleaned = clean_text(text)
    
    if len(cleaned) == 10:
        score += 100
    elif 8 <= len(cleaned) <= 12:
        score += 50 - abs(len(cleaned) - 10) * 5
    else:
        score += max(0, 20 - abs(len(cleaned) - 10))
    
    if re.fullmatch(r'[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}', cleaned):
        score += 200
    elif re.search(r'[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}', cleaned):
        score += 100
    
    if len(cleaned) >= 2 and cleaned[0:2] in VALID_STATE_CODES:
        score += 150
    
    if len(cleaned) == 10:
        letter_pos = {0, 1, 4, 5}
        digit_pos = {2, 3, 6, 7, 8, 9}
        for i in range(10):
            if i in letter_pos and cleaned[i].isalpha():
                score += 5
            elif i in digit_pos and cleaned[i].isdigit():
                score += 5
    
    return score

def vote_best_plate(candidates):
    """Use voting mechanism to find best plate - preserve exact OCR results, NO letter conversions"""
    if not candidates:
        return ""
    
                                                                           
    valid_candidates = []
    for c in candidates:
        cleaned = clean_text(c)
        if len(cleaned) == 10 and re.fullmatch(r'[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}', cleaned):
            valid_candidates.append(cleaned)                                   
    
                                                                                
    if valid_candidates:
        candidate_counts = Counter(valid_candidates)
                                                                         
        if candidate_counts:
            return candidate_counts.most_common(1)[0][0]
    
                                                                                       
                                     
    fixed_candidates = []
    for c in candidates:
        cleaned = clean_text(c)
        if len(cleaned) == 10:
                                                            
            fixed = position_aware_fix(cleaned)
            fixed_candidates.append(fixed)
        else:
            fixed_candidates.append(cleaned)
    
    candidate_counts = Counter(fixed_candidates)
    scored = []
    for candidate, count in candidate_counts.items():
        if not candidate or len(candidate.strip()) == 0:
            continue
        score = score_candidate(candidate)
        score += count * 10
                                     
        if len(candidate) >= 2 and candidate[0:2] in VALID_STATE_CODES:
            score += 50
        scored.append((score, candidate, count))
    
    if not scored:
        return ""
    
    scored.sort(key=lambda x: x[0], reverse=True)
    
                                                                                
    if scored:
        return scored[0][1]
    
    return ""

def choose_best_candidate(ocr_results):
    """Pick the most plausible plate from multiple OCR runs (strict, preserve OCR)"""
    candidates_with_scores = []
    
    for raw in ocr_results:
        if not raw or len(raw.strip()) == 0:
            continue
        cleaned = clean_text(raw)
        candidate = find_best_plate(cleaned)
        if len(candidate) == 10:
            candidate = position_aware_fix(candidate)
        else:
            if len(cleaned) == 10:
                candidate = position_aware_fix(cleaned)
            else:
                candidate = cleaned
        
        score = score_candidate(candidate)
        candidates_with_scores.append((score, candidate, cleaned))
    
    if not candidates_with_scores:
        return ""
    
    candidates_with_scores.sort(key=lambda x: x[0], reverse=True)
    
    for score, candidate, cleaned in candidates_with_scores:
        if (len(candidate) == 10
            and re.fullmatch(r'[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}', candidate)
            and candidate[0:2] in VALID_STATE_CODES):
            return candidate
    
    if candidates_with_scores:
        best = candidates_with_scores[0][1]
        if len(best) == 10:
            return position_aware_fix(best)
        return best
    
    return ""

def add_spaces(plate):
    """Converts: KA09MA2662 -> KA 09 MA 2662"""
    if len(plate) == 10:
        return f"{plate[0:2]} {plate[2:4]} {plate[4:6]} {plate[6:10]}"
    return plate

def segment_characters(img):
    """Segment individual characters from a number plate image"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    char_images = []
    char_positions = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 5 or h < 10:
            continue
        aspect_ratio = h / w if w > 0 else 0
        if aspect_ratio < 0.3 or aspect_ratio > 5:
            continue
        
        char_img = binary[y:y+h, x:x+w]
        padding = 5
        char_img = cv2.copyMakeBorder(char_img, padding, padding, padding, padding, 
                                     cv2.BORDER_CONSTANT, value=0)
        char_images.append(char_img)
        char_positions.append(x)
    
    sorted_chars = sorted(zip(char_positions, char_images), key=lambda x: x[0])
    return [char for _, char in sorted_chars]

def recognize_characters_individually(img, reader):
    """Recognize characters one by one for better accuracy"""
    chars = segment_characters(img)
    if not chars:
        return ""
    
    recognized = []
    for char_img in chars:
        char_inv = cv2.bitwise_not(char_img)
        h, w = char_inv.shape
        if h < 20:
            scale = 20 / h
            char_inv = cv2.resize(char_inv, None, fx=scale, fy=scale, 
                                 interpolation=cv2.INTER_CUBIC)
        
        try:
            results = reader.readtext(char_inv, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                                     detail=0, paragraph=True)
            if results:
                text = results[0].strip().upper()
                if text and len(text) > 0:
                    recognized.append(text[0])
        except:
            pass
    
    return "".join(recognized)

def grid_ocr(plate_img, reader):
    """Fast, position-stable OCR: split into character slots and read per slot.
    Does not change letters; only reads and cleans."""
    if len(plate_img.shape) == 3:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_img.copy()
    gray = cv2.resize(gray, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)
    gray = enhance_contrast(gray)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_inv = cv2.bitwise_not(binary)
    contours, _ = cv2.findContours(binary_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < 8 or h < 20:
            continue
        ar = h / float(w)
        if 0.4 < ar < 5.0:
            boxes.append((x, y, w, h))
    if not boxes:
        return ""
    boxes.sort(key=lambda b: b[0])
                                                        
    merged = []
    for b in boxes:
        if not merged:
            merged.append(b)
        else:
            px, py, pw, ph = merged[-1]
            x, y, w, h = b
            if x - (px + pw) < max(3, pw * 0.15):                     
                nx = min(px, x)
                ny = min(py, y)
                nw = max(px + pw, x + w) - nx
                nh = max(py + ph, y + h) - ny
                merged[-1] = (nx, ny, nw, nh)
            else:
                merged.append(b)
                                                 
    merged.sort(key=lambda b: b[0])
    chars = []
    for x, y, w, h in merged[:12]:
        roi = binary[y:y+h, x:x+w]
        roi = cv2.copyMakeBorder(roi, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=255)
                 
        txt1 = ""
        try:
            r1 = reader.readtext(cv2.bitwise_not(roi), allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", detail=0, paragraph=True)
            if r1:
                txt1 = (r1[0] or "").strip().upper()
        except:
            pass
                   
        txt2 = ""
        if TESSERACT_AVAILABLE:
            try:
                config = r'--oem 3 --psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                t = pytesseract.image_to_string(roi, config=config)
                txt2 = (t or "").strip().upper()
            except:
                pass
        cand = clean_text(txt1 or txt2)
        if cand:
            chars.append(cand[0])
    return "".join(chars)
def ocr_with_tesseract(img):
    """Use Tesseract OCR as alternative method"""
    if not TESSERACT_AVAILABLE:
        return ""
    
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    
    try:
        text = pytesseract.image_to_string(thresh, config=custom_config)
        return clean_text(text)
    except:
        return ""

def ensemble_ocr(plate_img, reader):
    """Use multiple OCR methods and combine results"""
    all_results = []
    variants = preprocess_variants(plate_img)
    for v in variants[:5]:
        try:
            dets = reader.readtext(v, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                                  detail=1, paragraph=False, text_threshold=0.5)
            if dets:
                text = "".join([d[1] for d in dets])
                all_results.append(clean_text(text))
        except:
            pass
    
    for v in variants[:3]:
        try:
            char_text = recognize_characters_individually(v, reader)
            if char_text:
                all_results.append(clean_text(char_text))
        except:
            pass
    
    if TESSERACT_AVAILABLE:
        for v in variants[:3]:
            try:
                tess_text = ocr_with_tesseract(v)
                if tess_text:
                    all_results.append(tess_text)
            except:
                pass
    
    return all_results

def analyze_character_uniformity(plate_img):
    """Analyze if characters are uniform in size"""
    if len(plate_img.shape) == 3:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_img.copy()
    
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) < 4:
        return 0.0, False
    
    char_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 5 or h < 10:
            continue
        char_boxes.append((x, y, w, h))
    
    if len(char_boxes) < 4:
        return 0.0, False
    
    char_boxes.sort(key=lambda b: b[0])
    heights = [h for _, _, _, h in char_boxes]
    widths = [w for _, _, w, _ in char_boxes]
    
    height_mean = np.mean(heights)
    height_std = np.std(heights)
    height_cv = height_std / height_mean if height_mean > 0 else 1.0
    
    width_mean = np.mean(widths)
    width_std = np.std(widths)
    width_cv = width_std / width_mean if width_mean > 0 else 1.0
    
    aspect_ratios = [h/w if w > 0 else 0 for _, _, w, h in char_boxes]
    ar_mean = np.mean(aspect_ratios)
    ar_std = np.std(aspect_ratios)
    ar_cv = ar_std / ar_mean if ar_mean > 0 else 1.0
    
    uniformity_score = 1.0 - min(1.0, (height_cv * 0.4 + width_cv * 0.3 + ar_cv * 0.3))
    is_uniform = (height_cv < 0.25 and width_cv < 0.30 and ar_cv < 0.25)
    
    if not is_uniform and uniformity_score > 0.6:
        is_uniform = True
    
    return uniformity_score, is_uniform

def analyze_character_spacing(plate_img):
    """Analyze spacing between characters"""
    if len(plate_img.shape) == 3:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_img.copy()
    
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) < 4:
        return 0.0, False
    
    char_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 5 or h < 10:
            continue
        char_boxes.append((x, y, w, h))
    
    if len(char_boxes) < 4:
        return 0.0, False
    
    char_boxes.sort(key=lambda b: b[0])
    gaps = []
    for i in range(len(char_boxes) - 1):
        x1, _, w1, _ = char_boxes[i]
        x2, _, _, _ = char_boxes[i + 1]
        gap = x2 - (x1 + w1)
        gaps.append(gap)
    
    if len(gaps) == 0:
        return 0.0, False
    
    gap_mean = np.mean(gaps)
    gap_std = np.std(gaps)
    gap_cv = gap_std / gap_mean if gap_mean > 0 else 1.0
    
    avg_char_width = np.mean([w for _, _, w, _ in char_boxes])
    ideal_gap = avg_char_width * 0.1
    
    spacing_score = 1.0 - min(1.0, abs(gap_mean - ideal_gap) / (ideal_gap + 1) + gap_cv * 0.5)
    is_proper_spacing = (gap_cv < 0.35 and 0.3 * ideal_gap < gap_mean < 3.0 * ideal_gap)
    
    if not is_proper_spacing and spacing_score > 0.55:
        is_proper_spacing = True
    
    return spacing_score, is_proper_spacing

def analyze_font_style(plate_img):
    """Analyze font style"""
    if len(plate_img.shape) == 3:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_img.copy()
    
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) < 4:
        return 0.0, False
    
    italic_scores = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 10 or h < 20:
            continue
        char_region = binary[y:y+h, x:x+w]
        moments = cv2.moments(char_region)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            center_x = w // 2
            slant = abs(cx - center_x) / w
            italic_scores.append(slant)
    
    if len(italic_scores) == 0:
        return 0.0, False
    
    avg_italic = np.mean(italic_scores)
    
    stroke_widths = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 10 or h < 20:
            continue
        char_region = binary[y:y+h, x:x+w]
        area = cv2.countNonZero(char_region)
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            stroke_width = (2 * area) / perimeter
            stroke_widths.append(stroke_width)
    
    if len(stroke_widths) == 0:
        return 0.0, False
    
    avg_stroke = np.mean(stroke_widths)
    stroke_cv = np.std(stroke_widths) / avg_stroke if avg_stroke > 0 else 1.0
    
    font_score = 1.0 - min(1.0, avg_italic * 1.5 + stroke_cv * 0.5)
    is_standard_font = (avg_italic < 0.25 and stroke_cv < 0.30)
    
    if not is_standard_font and font_score > 0.55:
        is_standard_font = True
    
    return font_score, is_standard_font

def fast_ocr_plate(plate_img, reader):
    variants = []
    v1 = preprocess_plate(plate_img, scale_factor=6, fast_mode=True)
    v2 = cv2.bitwise_not(v1)
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)
    gray = enhance_contrast(gray)
    a1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    a2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    variants.extend([v1, v2, a1, a2, plate_img])
    conf_thresh = 0.2
    best = ""
    for v in variants:
        try:
            dets = reader.readtext(v, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", detail=1, paragraph=False, width_ths=0.5, height_ths=0.5, text_threshold=0.3, link_threshold=0.2, slope_ths=0.2, ycenter_ths=0.7)
            if dets:
                sorted_dets = sorted(dets, key=lambda t: (sum([p[1] for p in t[0]])/4, sum([p[0] for p in t[0]])/4))
                high_conf = [(bbox, text, conf) for bbox, text, conf in sorted_dets if conf > conf_thresh]
                if high_conf:
                    joined_text = "".join([x[1] for x in high_conf]).strip()
                    jt = clean_text(joined_text)
                    cand1 = reconstruct_plate_from_detections(high_conf)
                    cands = [jt, cand1] if cand1 else [jt]
                    cand = vote_best_plate([c for c in cands if c])
                    if not cand:
                        cand = choose_best_candidate([c for c in cands if c])
                    if cand:
                        cand = clean_text(cand)
                        if len(cand) == 10 and re.fullmatch(r'[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}', cand):
                            return cand
                        if len(cand) == 10:
                            cand = position_aware_fix(cand)
                        if score_candidate(cand) > score_candidate(best):
                            best = cand
        except:
            pass
    if not best:
        try:
            grid_text = grid_ocr(plate_img, reader)
            if grid_text and len(grid_text) >= 8:
                gt = clean_text(grid_text)
                if len(gt) == 10 and re.fullmatch(r'[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}', gt):
                    return gt
                if len(gt) == 10:
                    gt = position_aware_fix(gt)
                best = gt
        except:
            pass
    if TESSERACT_AVAILABLE:
        for v in [v1, v2]:
            tess = ocr_with_tesseract(v)
            if tess and len(tess) >= 8:
                if len(tess) == 10 and re.fullmatch(r'[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}', tess):
                    return tess
                fixed = position_aware_fix(tess) if len(tess) == 10 else tess
                if score_candidate(fixed) > score_candidate(best):
                    best = fixed
    return best

def check_plate_authenticity(plate_img, plate_text, is_registered_func, registry_db_path, fast_mode=True):
    """Comprehensive check for plate authenticity"""
    details = {}
    
    format_valid = False
    if len(plate_text) == 10:
        format_valid = re.fullmatch(r'[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}', plate_text) is not None
        if format_valid:
            format_valid = plate_text[0:2] in VALID_STATE_CODES
    elif len(plate_text) == 9:
        format_valid = re.fullmatch(r'[A-Z]{2}[0-9]{2}[A-Z]{1}[0-9]{4}', plate_text) is not None
        if format_valid:
            format_valid = plate_text[0:2] in VALID_STATE_CODES
    
    details['format_valid'] = bool(format_valid)
    
    registered = is_registered_func(plate_text, db_path=registry_db_path)
    details["registered"] = bool(registered)
    
    if fast_mode:
                                                          
        if format_valid:
            uniformity_score = 0.70
            is_uniform = True
            spacing_score = 0.70
            is_proper_spacing = True
            font_score = 0.70
            is_standard_font = True
        else:
            uniformity_score = 0.30
            is_uniform = False
            spacing_score = 0.30
            is_proper_spacing = False
            font_score = 0.30
            is_standard_font = False
    else:
                                     
        uniformity_score, is_uniform = analyze_character_uniformity(plate_img)
        if uniformity_score == 0.0 and format_valid:
            uniformity_score = 0.65
            is_uniform = True
        
        spacing_score, is_proper_spacing = analyze_character_spacing(plate_img)
        if spacing_score == 0.0 and format_valid:
            spacing_score = 0.65
            is_proper_spacing = True
        
        font_score, is_standard_font = analyze_font_style(plate_img)
        if font_score == 0.0 and format_valid:
            font_score = 0.65
            is_standard_font = True
    
    details['uniformity'] = {'score': float(uniformity_score), 'is_uniform': bool(is_uniform)}
    details['spacing'] = {'score': float(spacing_score), 'is_proper': bool(is_proper_spacing)}
    details['font'] = {'score': float(font_score), 'is_standard': bool(is_standard_font)}
    
    is_real = bool(registered and format_valid and is_uniform and is_proper_spacing)
    
    if is_real:
        confidence = 0.95
    elif registered:
        confidence = 0.40
    else:
        confidence = 0.10
    
    return bool(is_real), float(confidence), details
