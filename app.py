from flask import Flask, render_template, request
import json

# Load knowledge base
with open("knowledge_base.json", "r") as f:
    knowledge_base = json.load(f)

app = Flask(__name__)

def find_solution(crop, symptom):
    crop = crop.lower()
    symptom = symptom.lower()
    
    for entry in knowledge_base:
        if entry["crop"].lower() == crop and entry["symptom"].lower() in symptom:
            return entry
    return None

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        crop = request.form["crop"]
        symptom = request.form["symptom"]
        result = find_solution(crop, symptom)
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
