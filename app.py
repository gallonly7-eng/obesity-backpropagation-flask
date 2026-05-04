"""
app.py – Flask Web App Prediksi Obesitas
Backpropagation from scratch (NumPy)
Dataset: obesity.csv dari Kaggle
"""
from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle, json, os

app = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))

# =========================
# LOAD MODEL
# =========================
W, b          = pickle.load(open(os.path.join(BASE, "model/weights.pkl"), "rb"))
scaler        = pickle.load(open(os.path.join(BASE, "model/scaler.pkl"), "rb"))
le            = pickle.load(open(os.path.join(BASE, "model/label_encoder.pkl"), "rb"))
cat_encoders  = pickle.load(open(os.path.join(BASE, "model/cat_encoders.pkl"), "rb"))
feature_cols  = pickle.load(open(os.path.join(BASE, "model/feature_cols.pkl"), "rb"))
metrics       = json.load(open(os.path.join(BASE, "model/metrics.json")))

# =========================
# FUNCTION MODEL
# =========================
relu = lambda z: np.maximum(0, z)

def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)

def predict_proba(X_scaled):
    A = X_scaled.copy()
    for i in range(len(W) - 1):
        A = relu(A @ W[i] + b[i])
    return softmax(A @ W[-1] + b[-1])

# =========================
# LABEL & INFO
# =========================
LABEL_MAP = {
    "Insufficient_Weight": "Kurus",
    "Normal_Weight": "Normal",
    "Overweight_Level_I": "Kegemukan I",
    "Overweight_Level_II": "Kegemukan II",
    "Obesity_Type_I": "Obesitas Tipe I",
    "Obesity_Type_II": "Obesitas Tipe II",
    "Obesity_Type_III": "Obesitas Tipe III",
}

RISK_INFO = {
    "Insufficient_Weight": {
        "color": "#63b3ed", "icon": "🥗", "level": "rendah",
        "desc": "Berat badan Anda di bawah normal (Kurus). Disarankan meningkatkan asupan nutrisi bergizi.",
        "tips": ["Konsultasikan dengan ahli gizi", "Tingkatkan asupan kalori bergizi", "Latihan ringan", "Pantau berat badan"]
    },
    "Normal_Weight": {
        "color": "#68d391", "icon": "✅", "level": "normal",
        "desc": "Berat badan ideal. Pertahankan gaya hidup sehat!",
        "tips": ["Makan seimbang", "Olahraga rutin", "Tidur cukup", "Kelola stres"]
    },
    "Overweight_Level_I": {
        "color": "#f6ad55", "icon": "⚠️", "level": "sedang",
        "desc": "Mulai kegemukan. Perhatikan pola makan.",
        "tips": ["Kurangi kalori", "Olahraga", "Hindari junk food"]
    },
    "Overweight_Level_II": {
        "color": "#ed8936", "icon": "⚠️", "level": "sedang",
        "desc": "Kegemukan tingkat lanjut.",
        "tips": ["Diet teratur", "Kardio rutin", "Batasi gula"]
    },
    "Obesity_Type_I": {
        "color": "#fc8181", "icon": "🚨", "level": "tinggi",
        "desc": "Obesitas tipe I.",
        "tips": ["Konsultasi dokter", "Program diet"]
    },
    "Obesity_Type_II": {
        "color": "#e53e3e", "icon": "🚨", "level": "tinggi",
        "desc": "Obesitas tipe II.",
        "tips": ["Penanganan medis", "Pantau kesehatan"]
    },
    "Obesity_Type_III": {
        "color": "#9b2335", "icon": "🆘", "level": "sangat tinggi",
        "desc": "Obesitas parah (morbid).",
        "tips": ["Segera ke dokter", "Perawatan intensif"]
    },
}

# =========================
# ROUTES
# =========================
@app.route("/")
def index():
    return render_template("index.html", metrics=metrics)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        d = request.get_json()

        def enc(col, val):
            try:
                return float(cat_encoders[col].transform([val])[0])
            except:
                return 0.0

        # INPUT
        age    = float(d["age"])
        gender = enc("Gender", d["gender"])
        height = float(d["height"]) / 100.0
        weight = float(d["weight"])
        calc   = enc("CALC", d["calc"])
        favc   = enc("FAVC", d["favc"])
        fcvc   = float(d["fcvc"])
        ncp    = float(d["ncp"])
        scc    = enc("SCC", d["scc"])
        smoke  = enc("SMOKE", d["smoke"])
        ch2o   = float(d["ch2o"])
        fhist  = enc("family_history_with_overweight", d["family_history"])
        faf    = float(d["faf"])
        tue    = float(d["tue"])
        caec   = enc("CAEC", d["caec"])
        mtrans = enc("MTRANS", d["mtrans"])

        X_in = np.array([[age, gender, height, weight, calc, favc,
                          fcvc, ncp, scc, smoke, ch2o, fhist, faf, tue, caec, mtrans]])

        X_sc = scaler.transform(X_in)

        proba = predict_proba(X_sc)[0]
        pred_idx = int(np.argmax(proba))
        pred_raw = le.classes_[pred_idx]
        pred_label = LABEL_MAP.get(pred_raw, pred_raw)

        confidence = round(float(proba[pred_idx]) * 100, 2)
        all_proba = {
            LABEL_MAP.get(cls, cls): round(float(p)*100, 2)
            for cls, p in zip(le.classes_, proba)
        }

        info = RISK_INFO.get(pred_raw, RISK_INFO["Normal_Weight"])
        bmi = weight / (height ** 2)

        return jsonify({
            "success": True,
            "result": pred_label,
            "bmi": round(bmi, 2),
            "confidence": confidence,
            "probability": all_proba,
            "color": info["color"],
            "icon": info["icon"],
            "level": info["level"],
            "desc": info["desc"],
            "tips": info["tips"],
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/about")
def about():
    return render_template("about.html", metrics=metrics)

# =========================
# RUN (FIX RAILWAY)
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",   # penting!
        port=port,
        debug=False       # production
    )