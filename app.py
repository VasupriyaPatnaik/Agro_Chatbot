from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import json, re, logging, sqlite3
from difflib import get_close_matches
from googletrans import Translator
from werkzeug.security import generate_password_hash, check_password_hash

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Flask App ----------------
app = Flask(__name__)
app.secret_key = "supersecretkey"  # ⚠️ change in production
translator = Translator()

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
    # Default admin
    cur.execute("SELECT * FROM users WHERE username=?", ("admin",))
    if not cur.fetchone():
        cur.execute("INSERT INTO users (username,password,role) VALUES (?,?,?)",
                    ("admin", generate_password_hash("admin123"), "admin"))
    # Default farmer
    cur.execute("SELECT * FROM users WHERE username=?", ("farmer",))
    if not cur.fetchone():
        cur.execute("INSERT INTO users (username,password,role) VALUES (?,?,?)",
                    ("farmer", generate_password_hash("farmer123"), "farmer"))
    conn.commit()
    conn.close()

init_db()

# ---------------- Knowledge Base ----------------
def load_kb():
    try:
        with open("knowledge_base.json", "r", encoding="utf-8") as f:
            kb = json.load(f)
        logger.info(f"Loaded {len(kb)} knowledge base entries")
        return kb
    except Exception as e:
        logger.error(f"Error loading knowledge_base.json: {e}")
        return []

def save_kb():
    global knowledge_base
    with open("knowledge_base.json", "w", encoding="utf-8") as f:
        json.dump(knowledge_base, f, ensure_ascii=False, indent=4)
    # Reload to ensure consistency
    knowledge_base = load_kb()

knowledge_base = load_kb()

# ---------------- Helper: Find Solution ----------------
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
    # Fuzzy match
    crop_matches = get_close_matches(crop, [entry["crop"].lower() for entry in knowledge_base], n=3, cutoff=0.6)
    if crop_matches:
        for entry in knowledge_base:
            if entry["crop"].lower() == crop_matches[0]:
                symptom_words = set(re.findall(r"\w+", symptom))
                entry_symptom_words = set(re.findall(r"\w+", entry["symptom"].lower()))
                if symptom_words & entry_symptom_words:
                    return entry
    # Symptom overlap
    for entry in knowledge_base:
        symptom_words = set(re.findall(r"\w+", symptom))
        entry_symptom_words = set(re.findall(r"\w+", entry["symptom"].lower()))
        if len(symptom_words & entry_symptom_words) >= 2:
            return entry
    return None

# ---------------- Routes ----------------
@app.route("/")
def index():
    if "username" not in session:
        return redirect(url_for("login"))
    if session.get("role") == "admin":
        return redirect(url_for("admin_dashboard"))
    return render_template("index.html", role=session.get("role"))

# ---------- Authentication ----------
@app.route("/login", methods=["GET", "POST"])
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
            if user[3] == "admin":
                return redirect(url_for("admin_dashboard"))
            else:
                return redirect(url_for("index"))
        else:
            error = "Invalid credentials"
    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ---------- Farmer Diagnosis ----------
@app.route("/diagnose", methods=["POST"])
def diagnose():
    if "role" not in session or session["role"] != "farmer":
        return jsonify({"error": "Unauthorized"}), 403

    try:
        data = request.get_json()
        crop = data.get("crop", "").strip()
        symptom = data.get("symptom", "").strip()
        user_lang = data.get("lang", "en")

        # Translate input to English for KB lookup
        try:
            translated_crop = translator.translate(crop, dest="en").text
            translated_symptom = translator.translate(symptom, dest="en").text
        except Exception:
            translated_crop, translated_symptom = crop, symptom

        result = find_solution(translated_crop, translated_symptom)

        if result:
            # Extract all fields from the KB entry
            possible_issue = result.get("possible_issue", "N/A")
            season = result.get("season", "N/A")
            prevention = result.get("prevention", "N/A")
            treatment = result.get("treatment", "N/A")
            organic_treatment = result.get("organic_treatment", "N/A")
            severity = result.get("severity", "N/A")

            # Translate all fields to user's language
            try:
                possible_issue = translator.translate(possible_issue, dest=user_lang).text
                season = translator.translate(season, dest=user_lang).text
                prevention = translator.translate(prevention, dest=user_lang).text
                treatment = translator.translate(treatment, dest=user_lang).text
                organic_treatment = translator.translate(organic_treatment, dest=user_lang).text
                severity = translator.translate(severity, dest=user_lang).text
            except Exception:
                pass

            return jsonify({
                "status": "success",
                "crop": crop,
                "symptom": symptom,
                "possible_issue": possible_issue,
                "season": season,
                "prevention": prevention,
                "treatment": treatment,
                "organic_treatment": organic_treatment,
                "severity": severity
            })
        else:
            msg = "No diagnosis found. Please consult a local expert."
            try:
                msg = translator.translate(msg, dest=user_lang).text
            except Exception:
                pass
            return jsonify({"status": "not_found", "message": msg}), 404

    except Exception as e:
        logger.error(f"Error in diagnose: {e}")
        return jsonify({"error": "Internal server error"}), 500

# ---------- Admin Dashboard ----------
@app.route("/admin")
def admin_dashboard():
    if "role" not in session or session["role"] != "admin":
        return redirect(url_for("login"))
    return render_template("admin.html")

@app.route("/admin/data")
def admin_data():
    if "role" not in session or session["role"] != "admin":
        return jsonify({"error": "Unauthorized"}), 403
    return jsonify(load_kb())

@app.route("/admin/add", methods=["POST"])
def admin_add():
    if "role" not in session or session["role"] != "admin":
        return jsonify({"error": "Unauthorized"}), 403
    data = request.get_json()

    required_fields = ["crop", "symptom", "treatment"]
    if not all(data.get(f) for f in required_fields):
        return jsonify({"status": "error", "message": "Missing required fields"}), 400

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
    save_kb()
    return jsonify({"status": "success", "message": "Entry added successfully", "data": knowledge_base})

@app.route("/admin/update/<int:idx>", methods=["PUT"])
def admin_update(idx):
    if "role" not in session or session["role"] != "admin":
        return jsonify({"error": "Unauthorized"}), 403
    try:
        data = request.get_json()
        knowledge_base[idx].update({
            "crop": data.get("crop", knowledge_base[idx]["crop"]),
            "symptom": data.get("symptom", knowledge_base[idx]["symptom"]),
            "possible_issue": data.get("possible_issue", knowledge_base[idx].get("possible_issue","")),
            "severity": data.get("severity", knowledge_base[idx].get("severity","")),
            "season": data.get("season", knowledge_base[idx].get("season","")),
            "prevention": data.get("prevention", knowledge_base[idx].get("prevention","")),
            "treatment": data.get("treatment", knowledge_base[idx]["treatment"]),
            "organic_treatment": data.get("organic_treatment", knowledge_base[idx].get("organic_treatment",""))
        })
        save_kb()
        return jsonify({"status":"success", "message": "Updated successfully", "data": knowledge_base})
    except Exception as e:
        return jsonify({"status":"error","message": str(e)}), 400

@app.route("/admin/delete/<int:idx>", methods=["DELETE"])
def admin_delete(idx):
    if "role" not in session or session["role"] != "admin":
        return jsonify({"error": "Unauthorized"}), 403
    try:
        knowledge_base.pop(idx)
        save_kb()
        return jsonify({"status": "success", "message": "Deleted successfully", "data": knowledge_base})
    except Exception as e:
        return jsonify({"status":"error","message": str(e)}), 400

# ---------- Translation Endpoint ----------
@app.route("/translate_text", methods=["POST"])
def translate_text():
    data = request.get_json()
    text = data.get("text","")
    lang = data.get("lang","en")
    if lang == "en" or not text.strip():
        return jsonify({"translated": text})
    try:
        translated = translator.translate(text, dest=lang).text
        return jsonify({"translated": translated})
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return jsonify({"translated": text})

# ---------------- Error Handlers ----------------
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500

# ---------------- Main ----------------
if __name__ == "__main__":
    logger.info("🚀 Starting AgroBot with authentication & multilingual support...")
    app.run(debug=True, host="0.0.0.0", port=5000)
