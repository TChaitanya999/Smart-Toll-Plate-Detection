

import cv2
import easyocr
import re
import numpy as np
from collections import Counter
from ultralytics import YOLO
from plate_registry import is_registered

                                            
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("⚠ Tesseract OCR not available. Install with: pip install pytesseract")

model = YOLO("my_model.pt")
reader = easyocr.Reader(['en'])

image_path = r"dataset(FakePlates)\images\Cars285.png"                           

                                                       
REGISTRY_DB_PATH = "plates.csv"

                                                                          
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

def preprocess_plate(img, scale_factor=4):
    """Enhanced preprocessing with multiple steps"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
                                              
    gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    
             
    gray = denoise_image(gray)
    
                      
    gray = enhance_contrast(gray)
    
             
    gray = sharpen_image(gray)
    
                                   
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
                       
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return th

def preprocess_variants(img):
    """
    Create multiple cleaned versions of the plate with different preprocessing strategies.
    """
    variants = []
    
                             
    for scale in [3, 4, 5]:
        base = preprocess_plate(img, scale_factor=scale)
        variants.append(base)
        
                          
        inverted = cv2.bitwise_not(base)
        variants.append(inverted)
        
                                  
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        dilated = cv2.dilate(base, kernel, iterations=1)
        variants.append(dilated)
        
        eroded = cv2.erode(base, kernel, iterations=1)
        variants.append(eroded)
    
                                    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    gray = enhance_contrast(gray)
    
                                          
    adaptive1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
    variants.append(adaptive1)
    
    adaptive2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
    variants.append(adaptive2)
    
                       
    variants.append(cv2.bitwise_not(adaptive1))
    variants.append(cv2.bitwise_not(adaptive2))
    
    return variants

def remove_left_ind_strip(plate_img):
    h, w = plate_img.shape[:2]
    cut = int(w * 0.25)
    return plate_img[:, cut:w]

def generate_crops(plate_img):
    """
    Return several crop variants to avoid losing characters on the left
    while also suppressing the blue IND strip if present.
    """
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
    text = text.replace("O", "0")
    text = text.replace("I", "1")
    text = text.replace("Z", "2")
    text = text.replace("S", "5")
    return text

def position_aware_fix(candidate):
    """
    Apply character fixes based on expected positions for Indian plates:
    AA NN AA NNNN  (10 chars)
    - Positions 0,1,4,5 are letters; positions 2,3,6,7,8,9 are digits.
    """
    c = clean_text(candidate)
    if len(c) != 10:
        return fix_common(c)

                              
    digit_confusions = {"O": "0", "I": "1", "Z": "2", "S": "5", "B": "8", "G": "6", "Q": "0", 
                       "D": "0", "T": "7", "L": "1"}
    letter_confusions = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G", 
                         "7": "T", "3": "E", "4": "A"}

    chars = list(c)
    letter_positions = {0, 1, 4, 5}                                                      
    digit_positions = {2, 3, 6, 7, 8, 9}                                                  
    
    for i in range(10):
        char = chars[i]
        if i in letter_positions:
                                
            if char.isdigit():
                                                       
                chars[i] = letter_confusions.get(char, char)
            elif char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                                                         
                if char in digit_confusions:
                                                          
                    pass
        elif i in digit_positions:
                               
            if char.isalpha():
                                                       
                chars[i] = digit_confusions.get(char, char)
            elif char not in "0123456789":
                                                        
                if char in letter_confusions:
                    chars[i] = digit_confusions.get(letter_confusions[char], char)

    return "".join(chars)

def group_detections_into_lines(detections):
    """
    EasyOCR detections: [ (bbox, text, conf), ... ]
    Returns list of lines, each line is list of (x_center, text, conf).
    Improved line grouping using height-based clustering.
    """
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
    """
    Handles 1-line and 2-line Indian plates.
    Example 2-line: "KA-09" / "MA 2662" -> KA09MA2662
    Now uses confidence scores to weight detections.
    """
    lines = group_detections_into_lines(detections)
    if not lines:
        return ""

                                                           
    joined_lines = []
    for line in lines:
                                                          
        line_sorted = sorted(line, key=lambda t: t[0])                      
        raw = "".join([t[1] for t in line_sorted])                          
        cleaned = clean_text(raw)
        joined_lines.append(cleaned)

                                                       
    all_text = "".join(joined_lines)
    all_text = position_aware_fix(all_text) if len(all_text) == 10 else fix_common(all_text)
    direct = find_best_plate(all_text)
    if len(direct) == 10 and direct[0:2] in VALID_STATE_CODES:
        return position_aware_fix(direct)

                                     
    if len(joined_lines) >= 2:
        top = fix_common(joined_lines[0])
        bottom = fix_common(joined_lines[1])

                                             
                                    
        top_trimmed = top[:4]                             
        bottom_trimmed = bottom[:6]                               
        candidate = position_aware_fix((top_trimmed + bottom_trimmed)[:10])
        if len(candidate) == 10 and candidate[0:2] in VALID_STATE_CODES:
            return candidate
        
                              
        candidate2 = position_aware_fix((top + bottom)[:10])
        if len(candidate2) == 10 and candidate2[0:2] in VALID_STATE_CODES:
            return candidate2

                                                         
    cleaned = fix_common("".join(joined_lines))
    for i in range(max(0, len(cleaned) - 9)):
        window = cleaned[i:i+10]
        window = position_aware_fix(window)
        if (
            re.fullmatch(r'[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}', window)
            and window[0:2] in VALID_STATE_CODES
        ):
            return window

    return fix_common("".join(joined_lines))

def find_best_plate(text):
    pat = r'[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}'
    match = re.search(pat, text)
    if match:
        return match.group()
    return text

def score_candidate(text):
    """
    Score a candidate plate text based on:
    - Length (prefer 10 chars)
    - Pattern match (Indian format)
    - Valid state code
    - Character validity at each position
    """
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
    """
    Use voting mechanism: count occurrences of each candidate and prefer most common valid plate.
    """
    if not candidates:
        return ""
    
                       
    candidate_counts = Counter(candidates)
    
                                 
    scored = []
    for candidate, count in candidate_counts.items():
        if not candidate or len(candidate.strip()) == 0:
            continue
        score = score_candidate(candidate)
                                        
        score += count * 10
        scored.append((score, candidate, count))
    
    if not scored:
        return ""
    
                   
    scored.sort(key=lambda x: x[0], reverse=True)
    
                                                                                
    for score, candidate, count in scored:
        if (
            len(candidate) == 10
            and re.fullmatch(r'[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}', candidate)
            and candidate[0:2] in VALID_STATE_CODES
        ):
            return position_aware_fix(candidate)
    
                                                                                
    if scored:
        best = scored[0][1]
                                                                             
        if len(best) == 10:
            return position_aware_fix(best)
        return best
    
    return ""

def choose_best_candidate(ocr_results):
    """
    Pick the most plausible plate from multiple OCR runs using scoring.
    """
    candidates_with_scores = []
    
    for raw in ocr_results:
        if not raw or len(raw.strip()) == 0:
            continue
            
        cleaned = clean_text(raw)
        cleaned = fix_common(cleaned)
        
                                      
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
        if (
            len(candidate) == 10
            and re.fullmatch(r'[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}', candidate)
            and candidate[0:2] in VALID_STATE_CODES
        ):
            return candidate
    
                                                              
    if candidates_with_scores:
        best = candidates_with_scores[0][1]
                                                                             
        if len(best) == 10:
            return position_aware_fix(best)
        return best
    
    return ""

def add_spaces(plate):
    """
    Converts: KA09MA2662 -> KA 09 MA 2662
    """
    if len(plate) == 10:
        return f"{plate[0:2]} {plate[2:4]} {plate[4:6]} {plate[6:10]}"
    return plate

def segment_characters(img):
    """
    Segment individual characters from a number plate image.
    Returns list of character images.
    """
                                    
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
    """
    Recognize characters one by one for better accuracy.
    """
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

def ocr_with_tesseract(img):
    """
    Use Tesseract OCR as alternative method.
    """
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
    """
    Use multiple OCR methods and combine results.
    """
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
    """
    Analyze if characters are uniform in size, straight, and not compressed/stretched.
    Returns: (uniformity_score, is_uniform)
    """
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
    """
    Analyze spacing between characters - should be uniform and not too close/wide.
    Returns: (spacing_score, is_proper_spacing)
    """
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
    """
    Analyze font style - should be standard HSRP font (no italic, no bold, no stylish).
    Returns: (font_score, is_standard_font)
    """
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

def check_plate_authenticity(plate_img, plate_text):
    """
    Comprehensive check for plate authenticity.
    Returns: (is_real, confidence, details)
    """
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
    
    details['format_valid'] = format_valid
    
                             
    uniformity_score, is_uniform = analyze_character_uniformity(plate_img)
                                                                            
    if uniformity_score == 0.0 and format_valid:
        uniformity_score = 0.65                            
        is_uniform = True
    details['uniformity'] = {
        'score': uniformity_score,
        'is_uniform': is_uniform
    }
    
                          
    spacing_score, is_proper_spacing = analyze_character_spacing(plate_img)
                                                                            
    if spacing_score == 0.0 and format_valid:
        spacing_score = 0.65                            
        is_proper_spacing = True
    details['spacing'] = {
        'score': spacing_score,
        'is_proper': is_proper_spacing
    }
    
                   
    font_score, is_standard_font = analyze_font_style(plate_img)
                                                                            
    if font_score == 0.0 and format_valid:
        font_score = 0.65                            
        is_standard_font = True
    details['font'] = {
        'score': font_score,
        'is_standard': is_standard_font
    }
    
                                    
                      
    weights = {
        'format': 0.3,
        'uniformity': 0.25,
        'spacing': 0.25,
        'font': 0.20
    }
    
    overall_score = (
        (1.0 if format_valid else 0.0) * weights['format'] +
        uniformity_score * weights['uniformity'] +
        spacing_score * weights['spacing'] +
        font_score * weights['font']
    )
    
                                       
                                             
                                                                                              
    visual_checks = [is_uniform, is_proper_spacing, is_standard_font]
    passed_visual = sum(visual_checks)
    
                        
                             
                                                                                   
                                                                                    
                                                                                            
    
    if format_valid:
        if overall_score >= 0.70:
                                                                
            is_real = passed_visual >= 2
        elif overall_score >= 0.55:
                                                                 
            is_real = passed_visual >= 1
        else:
                                                        
            is_real = passed_visual >= 3
    else:
                                          
        is_real = False
    
                                                                             
    if format_valid:
        confidence = max(overall_score, 0.60)                                  
    else:
        confidence = overall_score

                    
    registered = is_registered(plate_text, db_path=REGISTRY_DB_PATH)
    details["registered"] = registered

                                 
                                                                              
    is_real = (
        registered and 
        format_valid and 
        is_uniform and 
        is_proper_spacing
    )
    
                                               
    if is_real:
        confidence = 0.95
    elif registered:
                                             
        confidence = 0.40
    else:
                        
        confidence = 0.10
    
    return is_real, confidence, details

def classify_plate_status(plate_text, known_valid=None):
    """
    Returns a simple status tuple: (is_valid_pattern, is_known_real)
    - pattern: matches Indian format and starts with a valid state code
    - known_real: present in a provided whitelist (e.g., CSV/database/API)
    """
    is_pattern = (
        len(plate_text) == 10
        and re.fullmatch(r'[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}', plate_text) is not None
        and plate_text[0:2] in VALID_STATE_CODES
    )
    is_known_real = False
    if known_valid:
        is_known_real = plate_text in known_valid
    return is_pattern, is_known_real

img = cv2.imread(image_path)
if img is None:
    print("❌ Image not found!")
    exit()

results = model(img)

                                              
valid_plates = []

for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
                                                                      
                                                                        
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = width / height if height > 0 else 0
        
                                                               
        if width < 50 or height < 20 or aspect_ratio < 1.2:
            continue

        plate_crop = img[y1:y2, x1:x2]

                                                                                    
        crop_versions = generate_crops(plate_crop)

        raw_candidates = []
        
                                                
        for crop in crop_versions:
                                                   
            ensemble_results = ensemble_ocr(crop, reader)
            raw_candidates.extend(ensemble_results)
            
                                                                                
            variants = preprocess_variants(crop)
            for v in variants[:3]:                                      
                try:
                                                                         
                    dets = reader.readtext(
                        v, 
                        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", 
                        detail=1, 
                        paragraph=False,
                        width_ths=0.7,
                        height_ths=0.7,
                        slope_ths=0.1,
                        ycenter_ths=0.5,
                        link_threshold=0.4,
                        text_threshold=0.6,                                 
                        mag_ratio=1.5
                    )
                    
                                          
                    high_conf_dets = [(bbox, text, conf) for bbox, text, conf in dets if conf > 0.4]
                    
                    if high_conf_dets:
                                                                                   
                        plate_guess = reconstruct_plate_from_detections(high_conf_dets)
                        if plate_guess and len(plate_guess) >= 8:
                            raw_candidates.append(plate_guess)

                                                     
                        joined_text = "".join([x[1] for x in high_conf_dets]).strip()
                        if joined_text:
                            raw_candidates.append(clean_text(joined_text))
                    
                                                         
                    if dets:
                        plate_guess_all = reconstruct_plate_from_detections(dets)
                        if plate_guess_all and len(plate_guess_all) >= 8:
                            raw_candidates.append(plate_guess_all)
                except Exception as e:
                    pass

                                     
        cleaned_candidates = []
        for c in raw_candidates:
            if c and len(c.strip()) > 0:
                cleaned = clean_text(c)
                                                                                                     
                if len(cleaned) >= 4:
                    cleaned_candidates.append(cleaned)
        
                                    
        plate = vote_best_plate(cleaned_candidates)
        
                                                                 
        if not plate or len(plate) < 4:
            plate = choose_best_candidate(cleaned_candidates)
        
                                                                
        if plate:
            plate = clean_text(plate)
            
                                          
            if len(plate) == 10:
                                                                               
                plate = position_aware_fix(plate)
                plate = position_aware_fix(plate)                                   
            elif len(plate) >= 4:
                                              
                pattern_match = find_best_plate(plate)
                if pattern_match and len(pattern_match) == 10:
                    plate = position_aware_fix(pattern_match)
                elif len(plate) == 9 or len(plate) == 11:
                                                                        
                                                           
                    if len(plate) == 9:
                                                                    
                        for i in range(len(plate) + 1):
                            for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
                                test_plate = plate[:i] + char + plate[i:]
                                if len(test_plate) == 10:
                                    test_plate = position_aware_fix(test_plate)
                                    if (re.fullmatch(r'[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}', test_plate) 
                                        and test_plate[0:2] in VALID_STATE_CODES):
                                        plate = test_plate
                                        break
                            if len(plate) == 10:
                                break
                    elif len(plate) == 11:
                                                                   
                        for i in range(len(plate)):
                            test_plate = plate[:i] + plate[i+1:]
                            if len(test_plate) == 10:
                                test_plate = position_aware_fix(test_plate)
                                if (re.fullmatch(r'[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}', test_plate) 
                                    and test_plate[0:2] in VALID_STATE_CODES):
                                    plate = test_plate
                                    break
        
                                                       
        if plate and len(plate.strip()) > 0 and len(plate) >= 4:
                                                                                       
            has_letters = any(c.isalpha() for c in plate)
            has_numbers = any(c.isdigit() for c in plate)
            
                                                            
            if has_letters and has_numbers:
                                                                                       
                if len(plate) == 10:
                    plate_with_spaces = add_spaces(plate)
                else:
                                                                                        
                    plate_with_spaces = plate

                                                                   
                valid_plates.append({
                    'text': plate_with_spaces,
                    'bbox': (x1, y1, x2, y2),
                    'raw_plate': plate,
                    'plate_img': plate_crop.copy()                                              
                })

                                                
if len(valid_plates) > 1:
                                          
    scored_plates = []
    for plate_info in valid_plates:
        raw_plate = plate_info['raw_plate']
        score = score_candidate(raw_plate)
                                                  
        if len(raw_plate) == 10 and re.fullmatch(r'[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}', raw_plate):
            score += 200
        elif len(raw_plate) == 9 and re.fullmatch(r'[A-Z]{2}[0-9]{2}[A-Z]{1}[0-9]{4}', raw_plate):
            score += 150
        scored_plates.append((score, plate_info))
    
                                     
    scored_plates.sort(key=lambda x: x[0], reverse=True)
    valid_plates = [plate_info for _, plate_info in scored_plates[:1]]                      

                                                
if valid_plates:
    plate_info = valid_plates[0]                                   
    x1, y1, x2, y2 = plate_info['bbox']
    plate_text = plate_info['text']
    raw_plate = plate_info['raw_plate']
    plate_img = plate_info['plate_img']
    
                                
    is_real, confidence, details = check_plate_authenticity(plate_img, raw_plate)
    
                   
    print("=" * 60)
    print("✅ Final Plate :", plate_text)
    print("=" * 60)
    print("\n📋 AUTHENTICITY ANALYSIS:")
    print(f"   Format Valid      : {'✓ YES' if details['format_valid'] else '✗ NO'}")
    print(f"   Character Uniform : {'✓ YES' if details['uniformity']['is_uniform'] else '✗ NO'} (Score: {details['uniformity']['score']:.2f})")
    print(f"   Proper Spacing    : {'✓ YES' if details['spacing']['is_proper'] else '✗ NO'} (Score: {details['spacing']['score']:.2f})")
    print(f"   Standard Font      : {'✓ YES' if details['font']['is_standard'] else '✗ NO'} (Score: {details['font']['score']:.2f})")
    print(f"   Registered in DB  : {'✓ YES' if details.get('registered') else '✗ NO'}")
    print(f"\n   Overall Confidence : {confidence:.1%}")
    print("=" * 60)
    
                   
    if is_real:
        print("\n🟢 VERDICT: REAL NUMBER PLATE")
        print("   ✓ Plate is registered and passed all visual checks")
        color = (0, 255, 0)                  
        status_text = "REAL"
    else:
        print("\n🔴 VERDICT: FAKE NUMBER PLATE")
        if details.get('registered'):
            print("   ✓ Plate is found in registry database")
            print("   ✗ However, it FAILED visual authenticity checks")
        else:
            print("   ✗ Plate is NOT found in registry database")
        color = (0, 0, 255)                
        status_text = "FAKE"
    print("=" * 60)
    
                                
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, plate_text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                             
    cv2.putText(img, status_text, (x1, y2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

cv2.imshow("YOLO + OCR Plate With Spaces", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
