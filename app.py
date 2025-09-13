from flask import Flask, render_template, request, jsonify
import json
import re
from difflib import get_close_matches
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load knowledge base
try:
    with open("knowledge_base.json", "r", encoding='utf-8') as f:
        knowledge_base = json.load(f)
    logger.info(f"Successfully loaded {len(knowledge_base)} knowledge base entries")
except FileNotFoundError:
    logger.error("knowledge_base.json file not found! Using empty knowledge base.")
    knowledge_base = []
except json.JSONDecodeError as e:
    logger.error(f"Error parsing knowledge_base.json: {e}. Using empty knowledge base.")
    knowledge_base = []
except Exception as e:
    logger.error(f"Unexpected error loading knowledge base: {e}")
    knowledge_base = []

app = Flask(__name__)

def find_solution(crop, symptom):
    if not knowledge_base:
        logger.warning("Knowledge base is empty, cannot find solution")
        return None
        
    crop = crop.lower().strip()
    symptom = symptom.lower().strip()
    
    if not crop or not symptom:
        return None
    
    logger.info(f"Searching for solution: crop='{crop}', symptom='{symptom}'")
    
    # First try exact match for crop and symptom contained in description
    for entry in knowledge_base:
        if entry["crop"].lower() == crop and entry["symptom"].lower() in symptom:
            logger.info(f"Found exact match: {entry['crop']} - {entry['symptom']}")
            return entry
    
    # Then try if crop is contained in the knowledge base crop
    for entry in knowledge_base:
        if crop in entry["crop"].lower() and any(keyword in symptom for keyword in entry["symptom"].lower().split()):
            logger.info(f"Found partial match: {entry['crop']} - {entry['symptom']}")
            return entry
            
    # Try fuzzy matching for crops
    try:
        crop_matches = get_close_matches(crop, [entry["crop"].lower() for entry in knowledge_base], n=3, cutoff=0.6)
        if crop_matches:
            for entry in knowledge_base:
                if entry["crop"].lower() == crop_matches[0]:
                    # Check if any word from the symptom matches
                    symptom_words = set(re.findall(r'\w+', symptom))
                    entry_symptom_words = set(re.findall(r'\w+', entry["symptom"].lower()))
                    
                    if symptom_words & entry_symptom_words:
                        logger.info(f"Found fuzzy match: {entry['crop']} - {entry['symptom']}")
                        return entry
    except Exception as e:
        logger.warning(f"Fuzzy matching failed: {e}")
        pass
    
    # If no match found, try to find any crop with similar symptoms
    for entry in knowledge_base:
        symptom_words = set(re.findall(r'\w+', symptom))
        entry_symptom_words = set(re.findall(r'\w+', entry["symptom"].lower()))
        
        # If more than 2 words match
        if len(symptom_words & entry_symptom_words) >= 2:
            logger.info(f"Found symptom word match: {entry['crop']} - {entry['symptom']}")
            return entry
                
    logger.info("No matching solution found")
    return None

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None
    
    if request.method == "POST":
        crop = request.form.get("crop", "")
        symptom = request.form.get("symptom", "")
        
        if not crop or not symptom:
            error = "Please provide both crop and symptom information."
            logger.warning("Form submission missing crop or symptom")
        else:
            result = find_solution(crop, symptom)
            logger.info(f"Diagnosis result: {result is not None}")
    
    return render_template("index.html", result=result, error=error)

@app.route("/diagnose", methods=["POST"])
def diagnose():
    # Check if it's an AJAX request
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
    
    if is_ajax or request.is_json:
        try:
            if request.is_json:
                data = request.get_json()
            else:
                data = request.form
                
            crop = data.get("crop", "").strip()
            symptom = data.get("symptom", "").strip()
            
            if not crop or not symptom:
                return jsonify({"error": "Crop and symptom parameters are required"}), 400
            
            result = find_solution(crop, symptom)
            
            if result:
                return jsonify({
                    "status": "success",
                    "result": result
                })
            else:
                return jsonify({
                    "status": "not_found",
                    "message": "No diagnosis found. Please consult a local expert."
                }), 404
                
        except Exception as e:
            logger.error(f"Error in diagnose API: {e}")
            return jsonify({"error": "Internal server error"}), 500
    
    return jsonify({"error": "Invalid request"}), 400

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    logger.info("Starting AgroBot Flask server...")
    logger.info("Open http://127.0.0.1:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)