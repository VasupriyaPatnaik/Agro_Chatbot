from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import json, re, logging, sqlite3, os
from difflib import get_close_matches
from googletrans import Translator
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Image Processing Flags ----------------
IMAGE_PROCESSING_AVAILABLE = False
TENSORFLOW_AVAILABLE = False

try:
    import numpy as np
    from PIL import Image
    import cv2
    IMAGE_PROCESSING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Image processing libraries not available: {e}")
    class DummyNumpy:
        def expand_dims(self, *args, **kwargs): return None
        def argmax(self, *args, **kwargs): return 0
        def array(self, *args, **kwargs): return []
    np = DummyNumpy()

# Optional TensorFlow
ENABLE_ML = os.environ.get('ENABLE_ML', 'false').lower() == 'true'
if ENABLE_ML and IMAGE_PROCESSING_AVAILABLE:
    try:
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing import image
        TENSORFLOW_AVAILABLE = True
        logger.info("✅ TensorFlow libraries imported successfully")
    except ImportError as e:
        logger.warning(f"TensorFlow not available: {e}")

# ---------------- Flask App ----------------
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret")
translator = Translator()

# ---------------- Model Setup ----------------
MODEL_PATH = 'model.h5'
model = None
CLASS_NAMES = ['Healthy', 'Powdery', 'Rust']

if TENSORFLOW_AVAILABLE:
    try:
        if os.path.exists(MODEL_PATH):
            model = load_model(MODEL_PATH)
            logger.info("✅ TensorFlow model loaded successfully")
        else:
            logger.warning(f"⚠️ Model file {MODEL_PATH} not found")
    except Exception as e:
        logger.warning(f"⚠️ Could not load TensorFlow model: {e}")
        model = None
elif IMAGE_PROCESSING_AVAILABLE:
    logger.info("📷 Basic image processing available (no AI classification)")
else:
    logger.info("ℹ️ Image processing disabled")

# ---------------- Uploads ----------------
app.config['MAX_CONTENT_LENGTH'] = 12 * 1024 * 1024
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# ---------------- Helper: Translator ----------------
def safe_translate(text, dest="en"):
    if not text:
        return ""
    try:
        return translator.translate(text, dest=dest).text
    except Exception:
        try:
            new_translator = Translator()
            return new_translator.translate(text, dest=dest).text
        except Exception:
            return text

# ---------------- Image Preprocessing ----------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path, target_size=(225, 225)):
    if not IMAGE_PROCESSING_AVAILABLE:
        return None
    try:
        if TENSORFLOW_AVAILABLE:
            img = image.load_img(img_path, target_size=target_size)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            return img_array
        else:
            img = Image.open(img_path)
            img = img.resize(target_size)
            return np.array(img) / 255.0
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None

def classify_image(img_path):
    if model is not None and TENSORFLOW_AVAILABLE:
        try:
            input_image = preprocess_image(img_path)
            if input_image is None:
                return "Processing failed", [0.33, 0.33, 0.34]
            predictions = model.predict(input_image)
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            return CLASS_NAMES[predicted_class_index], predictions[0]
        except Exception as e:
            logger.error(f"TensorFlow classification error: {e}")
            return "Classification failed", [0.33, 0.33, 0.34]
    elif IMAGE_PROCESSING_AVAILABLE:
        try:
            img = Image.open(img_path)
            img_array = np.array(img)
            avg_color = np.mean(img_array, axis=(0, 1))
            green_ratio = avg_color[1] / (avg_color[0] + avg_color[2] + 1)
            if green_ratio > 0.7:
                return "Healthy", [0.8, 0.1, 0.1]
            elif avg_color[0] > avg_color[1]:
                return "Rust", [0.1, 0.1, 0.8]
            else:
                return "Powdery", [0.1, 0.8, 0.1]
        except Exception as e:
            logger.error(f"Basic image analysis error: {e}")
            return "Analysis failed", [0.33, 0.33, 0.34]
    else:
        return "Image processing not available", [0.33, 0.33, 0.34]

# ---------------- Database Setup ----------------
def init_db():
    conn = sqlite3.connect("agrobot.db")
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('admin','farmer'))
    )''')
    if not cur.execute("SELECT 1 FROM users WHERE username='admin'").fetchone():
        cur.execute("INSERT INTO users (username,password,role) VALUES (?,?,?)",
                    ("admin", generate_password_hash("admin123"), "admin"))
    if not cur.execute("SELECT 1 FROM users WHERE username='farmer'").fetchone():
        cur.execute("INSERT INTO users (username,password,role) VALUES (?,?,?)",
                    ("farmer", generate_password_hash("farmer123"), "farmer"))
    conn.commit()
    conn.close()

init_db()

# ---------------- Knowledge Base Management ----------------
def load_kb():
    try:
        with open("knowledge_base.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception as e:
        logger.error(f"Error loading KB: {e}")
        # Return default data if file doesn't exist
        default_data = [
            {
                "crop": "Tomato",
                "symptom": "Willed plants despite adequate water, yellowing leaves, brown vascular tissue",
                "possible_issue": "Fusarium wilt",
                "severity": "High",
                "season": "Warm soil temperatures (28-32°C)",
                "prevention": "Use resistant varieties (VFN), avoid continuous tomato cropping, solarize soil",
                "treatment": "Remove infected plants, treat soil with carbendazim (1g/liter)",
                "organic_treatment": "Apply Trichoderma viride, practice 4-year crop rotation"
            },
            {
                "crop": "Tomato",
                "symptom": "Concentric dark spots with yellow halos on lower leaves",
                "possible_issue": "Early blight",
                "severity": "Medium",
                "season": "Humid conditions, rainy season",
                "prevention": "Use disease-free seeds, proper spacing for air circulation",
                "treatment": "Apply fungicides like chlorothalonil or mancozeb",
                "organic_treatment": "Apply copper-based fungicides, neem oil"
            }
        ]
        save_kb(default_data)
        return default_data

def save_kb(data):
    try:
        with open("knowledge_base.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Error saving KB: {e}")
        return False

knowledge_base = load_kb()

# ---------------- Crop & Symptom Parsing ----------------
def parse_crop_and_symptom(text):
    text_lower = text.lower()
    crops = ["tomato", "wheat", "rice", "corn", "maize", "potato", "cotton", "soybean",
             "sugarcane", "onion", "pepper", "cucumber", "bean", "peas", "carrot", "cabbage", "lettuce"]
    detected_crop = next((c for c in crops if c in text_lower), "unknown")
    if detected_crop == "unknown":
        patterns = [r"my\s+(\w+)\s+plants?", r"(\w+)\s+plants?", r"(\w+)\s+crops?"]
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                detected_crop = match.group(1)
                break
    symptom = text
    if detected_crop != "unknown":
        remove_words = [detected_crop, "plants", "plant", "crop", "crops", "my", "have", "has"]
        symptom_words = [w for w in text.split() if w.lower() not in remove_words]
        symptom = " ".join(symptom_words).strip() or text
    return detected_crop.title(), symptom

# ---------------- KB Search ----------------
def find_solution(crop, symptom):
    if not knowledge_base:
        return None
    crop = crop.lower().strip()
    symptom = symptom.lower().strip()
    # Exact match
    for entry in knowledge_base:
        if entry["crop"].lower() == crop and entry["symptom"].lower() in symptom:
            return entry
    # Partial match
    for entry in knowledge_base:
        if crop in entry["crop"].lower() and any(kw in symptom for kw in entry["symptom"].lower().split()):
            return entry
    # Close match
    crop_matches = get_close_matches(crop, [e["crop"].lower() for e in knowledge_base], n=3, cutoff=0.6)
    if crop_matches:
        for entry in knowledge_base:
            if entry["crop"].lower() == crop_matches[0]:
                sw = set(re.findall(r"\w+", symptom))
                ew = set(re.findall(r"\w+", entry["symptom"].lower()))
                if sw & ew:
                    return entry
    # Symptom overlap
    for entry in knowledge_base:
        sw = set(re.findall(r"\w+", symptom))
        ew = set(re.findall(r"\w+", entry["symptom"].lower()))
        if len(sw & ew) >= 2:
            return entry
    return None

# ---------------- Routes ----------------
@app.route("/")
def home():
    if "username" not in session:
        return redirect(url_for("login"))
    return redirect(url_for("admin_dashboard" if session.get("role") == "admin" else "farmer_home"))

@app.route("/login", methods=["GET","POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        conn = sqlite3.connect("agrobot.db")
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username=?", (username,))
        user = cur.fetchone()
        conn.close()
        if user and check_password_hash(user[2], password):
            session["username"] = user[1]
            session["role"] = user[3]
            return redirect(url_for("home"))
        else:
            error = "Invalid credentials"
    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/farmer")
def farmer_home():
    return render_template("index.html", role="farmer")

# ---------------- Diagnose ----------------
@app.route("/diagnose", methods=["POST"])
def diagnose():
    if session.get("role") != "farmer":
        return jsonify({"error": "Unauthorized"}), 403
    data = request.get_json()
    user_text = data.get("text", "").strip()
    user_lang = data.get("lang", "en")
    location = data.get("location", "")

    if not user_text:
        msg = safe_translate("Please describe your crop issue.", dest=user_lang)
        return jsonify({"status": "error", "message": msg}), 400

    translated_text = safe_translate(user_text, dest="en")
    crop, symptom = parse_crop_and_symptom(translated_text)
    result = find_solution(crop, symptom)

    if result:
        fields = ["possible_issue", "season", "prevention", "treatment", "organic_treatment", "severity"]
        translated_result = {f: safe_translate(result.get(f, "N/A"), dest=user_lang) for f in fields}
        advice = f"Hello! Based on the symptoms for your {crop} crop at {location or 'your location'}, it might be affected by {translated_result['possible_issue']} (Severity: {translated_result['severity']}). Recommended actions: {translated_result['prevention']}. Treatment: {translated_result['treatment']}. Organic options: {translated_result['organic_treatment'] or 'Not specified'}."
        return jsonify({"status": "success", "advice": advice})

    msg = safe_translate("No diagnosis found. Please consult a local expert.", dest=user_lang)
    return jsonify({"status": "not_found", "advice": msg}), 404

# ---------------- Image Predict ----------------
@app.route("/predict", methods=["POST"])
def predict():
    if session.get("role") != "farmer":
        return jsonify({"error": "Unauthorized"}), 403
    if not IMAGE_PROCESSING_AVAILABLE:
        msg = "Image processing is not available."
        return jsonify({"status": "error", "advice": msg}), 400

    file = request.files.get('image') or request.files.get('file')
    user_lang = request.form.get("lang", "en")
    location = request.form.get("location", "")

    if not file or file.filename == '':
        msg = safe_translate("No file selected.", dest=user_lang)
        return jsonify({"status": "error", "advice": msg}), 400

    if allowed_file(file.filename):
        filename = secure_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)
        label, probs = classify_image(path)
        try: os.remove(path)
        except: pass

        # KB advice
        kb_entry = None
        for entry in knowledge_base:
            if label.lower() in entry.get("possible_issue", "").lower() or label.lower() in entry.get("symptom", "").lower():
                kb_entry = entry
                break

        if kb_entry:
            fields = ["possible_issue", "season", "prevention", "treatment", "organic_treatment", "severity"]
            translated_result = {f: safe_translate(kb_entry.get(f, "N/A"), dest=user_lang) for f in fields}
            advice = f"Your {label} prediction suggests the crop might be affected by {translated_result['possible_issue']} (Severity: {translated_result['severity']}). Recommended actions: {translated_result['prevention']}. Treatment: {translated_result['treatment']}. Organic options: {translated_result['organic_treatment'] or 'Not specified'}."
        else:
            advice = f"Your {label} prediction suggests the crop might be affected, but no detailed advice is available."
        return jsonify({"status": "success", "advice": advice})

    msg = safe_translate("Invalid file type.", dest=user_lang)
    return jsonify({"status": "error", "advice": msg}), 400

# ---------------- Admin Routes ----------------
@app.route("/admin")
def admin_dashboard():
    if session.get("role") != "admin":
        return redirect(url_for("login"))
    return render_template("admin.html")

@app.route("/admin/data", methods=["GET", "POST", "PUT", "DELETE"])
def admin_data():
    if session.get("role") != "admin":
        return jsonify({"error": "Unauthorized"}), 403
    
    global knowledge_base
    
    if request.method == "GET":
        # Return all knowledge base data
        return jsonify(load_kb())
    
    elif request.method == "POST":
        # Add new entry
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No data provided"}), 400
            
            # Validate required fields
            required_fields = ["crop", "symptom", "possible_issue"]
            for field in required_fields:
                if not data.get(field):
                    return jsonify({"error": f"Missing required field: {field}"}), 400
            
            new_entry = {
                "crop": data.get("crop", ""),
                "symptom": data.get("symptom", ""),
                "possible_issue": data.get("possible_issue", ""),
                "severity": data.get("severity", ""),
                "season": data.get("season", ""),
                "prevention": data.get("prevention", ""),
                "treatment": data.get("treatment", ""),
                "organic_treatment": data.get("organic_treatment", "")
            }
            
            knowledge_base.append(new_entry)
            if save_kb(knowledge_base):
                return jsonify({"message": "Entry added successfully", "data": new_entry}), 201
            else:
                return jsonify({"error": "Failed to save knowledge base"}), 500
                
        except Exception as e:
            return jsonify({"error": f"Error adding entry: {str(e)}"}), 500
    
    elif request.method == "PUT":
        # Update existing entry
        try:
            data = request.get_json()
            if not data or "index" not in data:
                return jsonify({"error": "No index provided"}), 400
            
            index = data["index"]
            if index < 0 or index >= len(knowledge_base):
                return jsonify({"error": "Invalid index"}), 400
            
            updated_entry = {
                "crop": data.get("crop", knowledge_base[index]["crop"]),
                "symptom": data.get("symptom", knowledge_base[index]["symptom"]),
                "possible_issue": data.get("possible_issue", knowledge_base[index]["possible_issue"]),
                "severity": data.get("severity", knowledge_base[index]["severity"]),
                "season": data.get("season", knowledge_base[index]["season"]),
                "prevention": data.get("prevention", knowledge_base[index]["prevention"]),
                "treatment": data.get("treatment", knowledge_base[index]["treatment"]),
                "organic_treatment": data.get("organic_treatment", knowledge_base[index]["organic_treatment"])
            }
            
            knowledge_base[index] = updated_entry
            if save_kb(knowledge_base):
                return jsonify({"message": "Entry updated successfully", "data": updated_entry})
            else:
                return jsonify({"error": "Failed to save knowledge base"}), 500
                
        except Exception as e:
            return jsonify({"error": f"Error updating entry: {str(e)}"}), 500
    
    elif request.method == "DELETE":
        # Delete entry
        try:
            data = request.get_json()
            if not data or "index" not in data:
                return jsonify({"error": "No index provided"}), 400
            
            index = data["index"]
            if index < 0 or index >= len(knowledge_base):
                return jsonify({"error": "Invalid index"}), 400
            
            deleted_entry = knowledge_base.pop(index)
            if save_kb(knowledge_base):
                return jsonify({"message": "Entry deleted successfully", "data": deleted_entry})
            else:
                return jsonify({"error": "Failed to save knowledge base"}), 500
                
        except Exception as e:
            return jsonify({"error": f"Error deleting entry: {str(e)}"}), 500

@app.route("/translate", methods=["POST"])
def translate_text():
    if session.get("role") != "admin":
        return jsonify({"error": "Unauthorized"}), 403
    
    data = request.get_json()
    text = data.get("text", "")
    target_lang = data.get("target_lang", "en")
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    try:
        translated_text = safe_translate(text, dest=target_lang)
        return jsonify({"translated_text": translated_text})
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return jsonify({"error": "Translation failed"}), 500

# ---------------- Errors ----------------
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500

# ---------------- Main ----------------
if __name__ == "__main__":
    logger.info("🚀 Starting AgroBot with multilingual KB + sentence outputs...")
    app.run(debug=True, host="0.0.0.0", port=5000)