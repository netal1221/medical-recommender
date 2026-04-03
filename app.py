from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('model/model.pkl')
le = joblib.load('model/label_encoder.pkl')
symptoms_list = joblib.load('model/symptoms_list.pkl')

# 🔥 DISEASE DATABASE
disease_info = {
    "Fungal infection": {
        "description": "A skin infection caused by fungi.",
        "medicines": ["Fluconazole", "Clotrimazole"],
        "diet": ["Garlic", "Probiotics"],
        "workout": ["Light yoga"]
    },
    "Diabetes": {
        "description": "Blood sugar imbalance condition.",
        "medicines": ["Metformin", "Insulin"],
        "diet": ["Low sugar foods", "Whole grains"],
        "workout": ["Walking", "Yoga"]
    },
    "Hypertension": {
        "description": "High blood pressure.",
        "medicines": ["Amlodipine", "Atenolol"],
        "diet": ["Low salt diet", "Fruits"],
        "workout": ["Cardio", "Meditation"]
    },
    "Migraine": {
        "description": "Severe headache.",
        "medicines": ["Sumatriptan", "Ibuprofen"],
        "diet": ["Avoid caffeine"],
        "workout": ["Rest", "Relaxation"]
    }
}

# 🔥 SYMPTOM MAPPING
symptom_data = {
    "fever": {"med": ["Paracetamol"], "diet": ["Soup", "Warm water"], "work": ["Rest"]},
    "cough": {"med": ["Cough syrup"], "diet": ["Honey tea"], "work": ["Rest"]},
    "headache": {"med": ["Ibuprofen"], "diet": ["Hydration"], "work": ["Relaxation"]},
    "nausea": {"med": ["Ondansetron"], "diet": ["Light food"], "work": ["Rest"]},
    "vomiting": {"med": ["ORS"], "diet": ["Banana", "Rice"], "work": ["Rest"]},
    "fatigue": {"med": ["Multivitamins"], "diet": ["Protein diet"], "work": ["Light yoga"]},
    "diarrhea": {"med": ["ORS", "Loperamide"], "diet": ["Curd", "Rice"], "work": ["Rest"]},
    "constipation": {"med": ["Laxatives"], "diet": ["Fiber food"], "work": ["Walking"]},
    "cold": {"med": ["Antihistamine"], "diet": ["Warm fluids"], "work": ["Rest"]},
    "body_pain": {"med": ["Pain reliever"], "diet": ["Protein"], "work": ["Stretching"]},
    "chest_pain": {"med": ["Aspirin"], "diet": ["Low fat"], "work": ["Rest"]},
    "dizziness": {"med": ["ORS"], "diet": ["Hydration"], "work": ["Rest"]},
    "shortness_of_breath": {"med": ["Inhaler"], "diet": ["Fruits"], "work": ["Breathing exercises"]},
    "joint_pain": {"med": ["Diclofenac"], "diet": ["Calcium foods"], "work": ["Stretching"]},
    "acidity": {"med": ["Antacid"], "diet": ["Avoid spicy"], "work": ["Walking"]},
    "sore_throat": {"med": ["Lozenges"], "diet": ["Warm tea"], "work": ["Rest"]},
    "runny_nose": {"med": ["Antihistamine"], "diet": ["Warm fluids"], "work": ["Rest"]},
    "back_pain": {"med": ["Muscle relaxant"], "diet": ["Calcium"], "work": ["Stretching"]},
    "skin_rash": {"med": ["Antifungal cream"], "diet": ["Avoid sugar"], "work": ["Hygiene"]},
    "itching": {"med": ["Antihistamine"], "diet": ["Healthy diet"], "work": ["Skin care"]},
}

# ✅ ROUTES
@app.route('/')
def index():
    return render_template('index.html', symptoms=symptoms_list)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/how')
def how():
    return render_template('how.html')

# 🔥 CLEAN FUNCTION
def clean_list(data_set):
    return list(set(
        item.strip() for item in data_set
        if item and item.strip() and item.strip() != "-"
    ))

# 🔥 PREDICT
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    selected_symptoms = data.get('symptoms', [])

    input_vector = [1 if s in selected_symptoms else 0 for s in symptoms_list]
    input_array = np.array(input_vector).reshape(1, -1)

    prediction_encoded = model.predict(input_array)[0]
    probabilities = model.predict_proba(input_array)[0]
    confidence = round(max(probabilities) * 100, 2)

    disease_name = le.inverse_transform([prediction_encoded])[0]

    info = disease_info.get(disease_name)

    if not info:
        meds, diet, workout = set(), set(), set()

        for s in selected_symptoms:
            d = symptom_data.get(s)
            if d:
                meds.update(d.get("med", []))
                diet.update(d.get("diet", []))
                workout.update(d.get("work", []))

        meds = clean_list(meds)
        diet = clean_list(diet)
        workout = clean_list(workout)

        if not meds:
            meds = ["Paracetamol", "General checkup"]
        if not diet:
            diet = ["Balanced diet", "Fruits", "Stay hydrated"]
        if not workout:
            workout = ["Walking", "Light exercise", "Proper rest"]

        info = {
            "description": f"Based on symptoms, possible condition: {disease_name}",
            "medicines": meds,
            "diet": diet,
            "workout": workout
        }

    return jsonify({
        "disease": disease_name,
        "confidence": confidence,
        "description": info["description"],
        "medicines": info["medicines"],
        "diet": info["diet"],
        "workout": info["workout"]
    })

# ✅ FINAL FIX (IMPORTANT)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
    