from flask import Flask, render_template, request, jsonify
import cv2
import easyocr
import re
from ultralytics import YOLO
from plate_registry import is_registered, get_plate_record
import base64
import os
import traceback
from werkzeug.utils import secure_filename
from detection_functions import (
    clean_text, reconstruct_plate_from_detections,
    vote_best_plate, choose_best_candidate, add_spaces, score_candidate,
    check_plate_authenticity, position_aware_fix, find_best_plate, preprocess_plate,
    recognize_characters_individually, VALID_STATE_CODES, enhance_contrast, ensemble_ocr, fast_ocr_plate, ocr_with_tesseract
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024                      

                                              
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = YOLO("my_model.pt")
reader = easyocr.Reader(['en'], gpu=False)

                        
REGISTRY_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plates.csv")

def process_image(image_path):
    """
    Process an image and return detection results.
    Returns a dictionary with all the analysis results.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"success": False, "error": "Could not read image"}
        
        h, w = img.shape[:2]
        target_w = 640
        if w > target_w:
            scale = target_w / float(w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        results = model.predict(img, imgsz=640, conf=0.25, iou=0.5, verbose=False)
        valid_plates = []
        
        best_box = None
        best_conf = -1.0
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                width = x2 - x1
                height = y2 - y1
                aspect_ratio = width / height if height > 0 else 0
                conf_val = float(box.conf[0]) if hasattr(box, 'conf') else 0.0
                if width < 50 or height < 20 or aspect_ratio < 1.2:
                    continue
                if conf_val > best_conf:
                    best_conf = conf_val
                    best_box = (x1, y1, x2, y2)
        
        if best_box:
            x1, y1, x2, y2 = best_box
            plate_crop = img[y1:y2, x1:x2]
            raw_candidates = []
            
            try:
                h, w = plate_crop.shape[:2]
                
                if w > 50:
                    plate_crop_trimmed = plate_crop[:, int(w*0.15):]
                    crop_variants = [(plate_crop_trimmed, "trimmed"), (plate_crop, "original")]
                else:
                    crop_variants = [(plate_crop, "original")]
                
                for plate_variant, _ in crop_variants:
                    try:
                        fast_candidate = fast_ocr_plate(plate_variant, reader)
                        if fast_candidate and len(fast_candidate) >= 8:
                            raw_candidates.append(clean_text(fast_candidate))
                            if len(fast_candidate) == 10 and re.fullmatch(r'[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}', fast_candidate):
                                break
                    except:
                        pass
                    if not raw_candidates:
                        try:
                            pre = preprocess_plate(plate_variant, scale_factor=6, fast_mode=True)
                            tess_guess = ocr_with_tesseract(pre)
                            if tess_guess and len(tess_guess) >= 8:
                                raw_candidates.append(clean_text(tess_guess))
                        except:
                            pass
                    if raw_candidates:
                        break
                
                if not raw_candidates:
                    try:
                        fallback_crop = plate_crop[:, int(w*0.15):] if w > 50 else plate_crop
                        dets_para = reader.readtext(
                            fallback_crop,
                            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                            detail=0,
                            paragraph=True,
                            text_threshold=0.3
                        )
                        if dets_para:
                            cleaned_para = clean_text(dets_para)
                            if len(cleaned_para) >= 8:
                                raw_candidates.append(cleaned_para)
                    except:
                        pass
                        
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
                
                if len(plate) == 10 and re.fullmatch(r'[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}', plate):
                    pass
                elif len(plate) == 10:
                    plate = position_aware_fix(plate)
                elif len(plate) >= 8:
                    pattern_match = find_best_plate(plate)
                    if pattern_match and len(pattern_match) == 10:
                        if re.fullmatch(r'[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}', pattern_match):
                            plate = pattern_match
                        else:
                            plate = position_aware_fix(pattern_match)
            
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
        else:
            return {
                "success": False,
                "error": "No number plate detected in the image"
            }
        
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
            
                                                  
            is_real, confidence, details = check_plate_authenticity(
                plate_img, raw_plate, is_registered, REGISTRY_DB_PATH, fast_mode=True
            )

            registry_record = None
            if details.get("registered"):
                registry_record = get_plate_record(raw_plate, db_path=REGISTRY_DB_PATH)
            
                                        
            color = (0, 255, 0) if is_real else (0, 0, 255)
            status_text = "REAL" if is_real else "FAKE"
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, plate_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img, status_text, (x1, y2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
                                                
            _, buffer = cv2.imencode('.jpg', img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return {
                "success": True,
                "plate_text": plate_text,
                "raw_plate": raw_plate,
                "is_real": bool(is_real),
                "confidence": float(confidence),
                "registry": registry_record,
                "details": {
                    "format_valid": bool(details['format_valid']),
                    "uniformity": {
                        "is_uniform": bool(details['uniformity']['is_uniform']),
                        "score": float(details['uniformity']['score'])
                    },
                    "spacing": {
                        "is_proper": bool(details['spacing']['is_proper']),
                        "score": float(details['spacing']['score'])
                    },
                    "font": {
                        "is_standard": bool(details['font']['is_standard']),
                        "score": float(details['font']['score'])
                    },
                    "registered": bool(details.get('registered', False))
                },
                "image": img_base64
            }
        else:
            return {
                "success": False,
                "error": "No number plate detected in the image"
            }
    except Exception as e:
        traceback.print_exc()
        return {
            "success": False,
            "error": f"Error processing image: {str(e)}"
        }

@app.errorhandler(500)
def handle_500_error(e):
    return jsonify({"error": "Internal server error", "success": False}), 500

@app.errorhandler(400)
def handle_400_error(e):
    return jsonify({"error": "Bad request", "success": False}), 400

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    filepath = None
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not file:
            return jsonify({"error": "Invalid file"}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result = process_image(filepath)
        
                                
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass
        
        return jsonify(result)
        
    except Exception as e:
                                             
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass
        
                                              
        error_msg = str(e)
        traceback.print_exc()
        return jsonify({"error": error_msg, "success": False}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
