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
    logger.info("✅ Basic image processing libraries imported successfully")
except ImportError as e:
    logger.warning(f"Image processing libraries not available: {e}")
    # Create better dummy numpy implementation
    class DummyNumpy:
        def expand_dims(self, arr, axis=0):
            return [arr] if isinstance(arr, (int, float)) else arr
        def argmax(self, arr, axis=None):
            return 0
        def array(self, data, dtype=None):
            return data
        def mean(self, arr, axis=None):
            if isinstance(arr, list):
                return sum(arr) / len(arr) if len(arr) > 0 else 0
            return 128
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    np = DummyNumpy()

# Optional TensorFlow with better error handling
ENABLE_ML = os.environ.get('ENABLE_ML', 'true').lower() == 'true'  # Default to enabled
if ENABLE_ML and IMAGE_PROCESSING_AVAILABLE:
    try:
        # Set environment to reduce conflicts
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        
        # Try importing TensorFlow components
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')  # Disable GPU if available
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing import image
        TENSORFLOW_AVAILABLE = True
        logger.info("✅ TensorFlow libraries imported successfully")
    except ImportError as e:
        logger.warning(f"TensorFlow not available: {e}")
        TENSORFLOW_AVAILABLE = False
    except Exception as e:
        logger.warning(f"TensorFlow import error: {e}")
        TENSORFLOW_AVAILABLE = False
else:
    TENSORFLOW_AVAILABLE = False

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
            confidence = float(predictions[0][predicted_class_index])
            return CLASS_NAMES[predicted_class_index], confidence, predictions[0].tolist()
        except Exception as e:
            logger.error(f"TensorFlow classification error: {e}")
            return "Classification failed", 0.0, [0.33, 0.33, 0.34]
    elif IMAGE_PROCESSING_AVAILABLE:
        try:
            img = Image.open(img_path)
            img_array = np.array(img)
            avg_color = np.mean(img_array, axis=(0, 1))
            green_ratio = avg_color[1] / (avg_color[0] + avg_color[2] + 1)
            
            if green_ratio > 0.7:
                confidence = min(0.95, green_ratio)
                return "Healthy", confidence, [confidence, 0.1, 0.1]
            elif avg_color[0] > avg_color[1]:
                confidence = 0.85
                return "Rust", confidence, [0.1, 0.1, confidence]
            else:
                confidence = 0.80
                return "Powdery", confidence, [0.1, confidence, 0.1]
        except Exception as e:
            logger.error(f"Basic image analysis error: {e}")
            return "Analysis failed", 0.0, [0.33, 0.33, 0.34]
    else:
        return "Image processing not available", 0.0, [0.33, 0.33, 0.34]

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
            },
            {
                "crop": "General",
                "symptom": "Healthy green leaves, good growth",
                "possible_issue": "Healthy",
                "severity": "None",
                "season": "All seasons",
                "prevention": "Continue good agricultural practices",
                "treatment": "No treatment needed",
                "organic_treatment": "Maintain organic practices"
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
    
    # Check for healthy cases first
    if any(word in symptom for word in ['healthy', 'green', 'good', 'normal', 'fine']):
        for entry in knowledge_base:
            if entry["possible_issue"].lower() == "healthy":
                return entry
    
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
    
    if session.get("role") == "admin":
        return redirect(url_for("admin_dashboard"))
    else:
        return redirect(url_for("farmer_home"))

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
    
    # Try to render template, fallback to simple HTML
    try:
        return render_template("login.html", error=error)
    except:
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AgroBot Login</title>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 400px; margin: 100px auto; padding: 20px; }}
                input {{ width: 100%; padding: 10px; margin: 10px 0; box-sizing: border-box; }}
                button {{ width: 100%; padding: 10px; background: #4CAF50; color: white; border: none; cursor: pointer; }}
                .error {{ color: red; }}
            </style>
        </head>
        <body>
            <h2>🌱 AgroBot Login</h2>
            {f'<p class="error">{error}</p>' if error else ''}
            <form method="POST">
                <input type="text" name="username" placeholder="Username" required>
                <input type="password" name="password" placeholder="Password" required>
                <button type="submit">Login</button>
            </form>
            <p><strong>Demo accounts:</strong><br>
            Farmer: farmer/farmer123<br>
            Admin: admin/admin123</p>
        </body>
        </html>
        """

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/farmer")
def farmer_home():
    if session.get("role") != "farmer":
        return redirect(url_for("login"))
    return render_template("index.html", role="farmer")

# ---------------- Diagnose ----------------
@app.route("/diagnose", methods=["POST"])
def diagnose():
    if session.get("role") != "farmer":
        return jsonify({"error": "Unauthorized"}), 403
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
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
            # Check if it's a healthy diagnosis
            is_healthy = result.get("possible_issue", "").lower() == "healthy"
            
            fields = ["possible_issue", "season", "prevention", "treatment", "organic_treatment", "severity"]
            translated_result = {f: safe_translate(result.get(f, "N/A"), dest=user_lang) for f in fields}
            
            if is_healthy:
                advice = f"Based on the symptoms for your {crop} crop at {location or 'your location'}, {translated_result['possible_issue']}. {translated_result['prevention']}"
            else:
                advice = f"Based on the symptoms for your {crop} crop at {location or 'your location'}, it might be affected by {translated_result['possible_issue']} (Severity: {translated_result['severity']}). Recommended actions: {translated_result['prevention']}. Treatment: {translated_result['treatment']}. Organic options: {translated_result['organic_treatment'] or 'Not specified'}."
            
            return jsonify({
                "status": "success", 
                "advice": advice,
                "is_healthy": is_healthy
            })

        msg = safe_translate("No diagnosis found. Please consult a local expert.", dest=user_lang)
        return jsonify({"status": "not_found", "advice": msg}), 404
        
    except Exception as e:
        logger.error(f"Diagnosis error: {e}")
        return jsonify({"error": "Internal server error"}), 500

# ---------------- Image Predict ----------------
@app.route("/predict", methods=["POST"])
def predict():
    if session.get("role") != "farmer":
        return jsonify({"error": "Unauthorized"}), 403
        
    if not IMAGE_PROCESSING_AVAILABLE:
        msg = "Image processing is not available."
        return jsonify({"status": "error", "advice": msg}), 400

    try:
        file = request.files.get('image') or request.files.get('file')
        user_lang = request.form.get("lang", "en")
        location = request.form.get("location", "")

        if not file or file.filename == '':
            msg = safe_translate("No file selected.", dest=user_lang)
            return jsonify({"status": "error", "advice": msg}), 400

        if not allowed_file(file.filename):
            msg = safe_translate("Invalid file type.", dest=user_lang)
            return jsonify({"status": "error", "advice": msg}), 400

        filename = secure_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)
        
        # Classify image
        label, confidence, probabilities = classify_image(path)
        
        # Clean up uploaded file
        try: 
            os.remove(path)
        except: 
            pass

        # Find matching KB entry
        kb_entry = None
        is_healthy = label.lower() == "healthy"
        
        for entry in knowledge_base:
            if (label.lower() in entry.get("possible_issue", "").lower() or 
                label.lower() in entry.get("symptom", "").lower() or
                (is_healthy and entry.get("possible_issue", "").lower() == "healthy")):
                kb_entry = entry
                break

        # Build response
        if kb_entry:
            fields = ["possible_issue", "season", "prevention", "treatment", "organic_treatment", "severity"]
            translated_result = {f: safe_translate(kb_entry.get(f, "N/A"), dest=user_lang) for f in fields}
            
            if is_healthy:
                advice = f"Based on the image analysis for your crop at {location or 'your location'}, {translated_result['possible_issue']}. {translated_result['prevention']}"
            else:
                advice = f"Based on the image analysis for your crop at {location or 'your location'}, it might be affected by {translated_result['possible_issue']} (Severity: {translated_result['severity']}). Recommended actions: {translated_result['prevention']}. Treatment: {translated_result['treatment']}. Organic options: {translated_result['organic_treatment'] or 'Not specified'}."
        else:
            if is_healthy:
                advice = f"Based on the image analysis for your crop at {location or 'your location'}, your crop appears to be Healthy. Continue with good agricultural practices like proper watering, balanced fertilization, and regular monitoring."
            else:
                advice = f"Your {label} prediction suggests the crop might be affected, but no detailed advice is available. Please consult with local agricultural experts for specific guidance."

        return jsonify({
            "status": "success", 
            "advice": advice,
            "prediction": label,
            "confidence": confidence,
            "probabilities": probabilities,
            "is_healthy": is_healthy
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        msg = safe_translate("Error processing image. Please try again.", dest=user_lang)
        return jsonify({"status": "error", "advice": msg}), 500

# ---------------- Admin Routes ----------------
@app.route("/admin")
def admin_dashboard():
    if session.get("role") != "admin":
        return redirect(url_for("login"))
    
    # Try to render template, fallback to simple HTML
    try:
        return render_template("admin.html")
    except:
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>AgroBot Admin</title>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }}
                .header {{ background: #2c7a2c; color: white; padding: 20px; border-radius: 10px; }}
                .content {{ padding: 20px; }}
                .logout {{ background: #f44336; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; }}
                .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
                .stat-card {{ background: #f9f9f9; padding: 20px; border-radius: 10px; text-align: center; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>🌱 AgroBot Admin Dashboard</h2>
                <a href="/logout" class="logout">Logout</a>
            </div>
            <div class="content">
                <div class="stats">
                    <div class="stat-card">
                        <h3>Knowledge Base</h3>
                        <p>{len(knowledge_base)} entries</p>
                    </div>
                    <div class="stat-card">
                        <h3>ML Status</h3>
                        <p>{'✅ Active' if TENSORFLOW_AVAILABLE else '⚠️ Disabled'}</p>
                    </div>
                    <div class="stat-card">
                        <h3>Image Processing</h3>
                        <p>{'✅ Available' if IMAGE_PROCESSING_AVAILABLE else '❌ Disabled'}</p>
                    </div>
                </div>
                <h3>📚 Knowledge Base Management</h3>
                <p>Use the API endpoints to manage knowledge base entries:</p>
                <ul>
                    <li>GET /admin/data - View all entries</li>
                    <li>POST /admin/data - Add new entry</li>
                    <li>PUT /admin/data - Update entry</li>
                    <li>DELETE /admin/data - Delete entry</li>
                </ul>
            </div>
        </body>
        </html>
        """

@app.route("/admin/data", methods=["GET", "POST", "PUT", "DELETE"])
def admin_data():
    if session.get("role") != "admin":
        return jsonify({"error": "Unauthorized"}), 403
    
    global knowledge_base
    
    try:
        if request.method == "GET":
            return jsonify(load_kb())
        
        elif request.method == "POST":
            data = request.get_json()
            if not data:
                return jsonify({"error": "No data provided"}), 400
            
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
        
        elif request.method == "PUT":
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
        
        elif request.method == "DELETE":
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
        logger.error(f"Admin data error: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route("/translate", methods=["POST"])
def translate_text():
    if session.get("role") != "admin":
        return jsonify({"error": "Unauthorized"}), 403
    
    try:
        data = request.get_json()
        text = data.get("text", "")
        target_lang = data.get("target_lang", "en")
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
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
    logger.info("🚀 Starting AgroBot with enhanced image processing...")
    logger.info(f"📷 Image processing: {IMAGE_PROCESSING_AVAILABLE}")
    logger.info(f"🤖 TensorFlow ML: {TENSORFLOW_AVAILABLE}")
    logger.info(f"📚 Knowledge base entries: {len(knowledge_base)}")
    
    # Disable auto-reload to prevent TensorFlow import issues
    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(debug=debug_mode, host="0.0.0.0", port=5000, use_reloader=False)