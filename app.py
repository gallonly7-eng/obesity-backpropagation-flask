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

# Load model artifacts
W, b          = pickle.load(open(os.path.join(BASE, "model/weights.pkl"), "rb"))
scaler        = pickle.load(open(os.path.join(BASE, "model/scaler.pkl"), "rb"))
le            = pickle.load(open(os.path.join(BASE, "model/label_encoder.pkl"), "rb"))
cat_encoders  = pickle.load(open(os.path.join(BASE, "model/cat_encoders.pkl"), "rb"))
feature_cols  = pickle.load(open(os.path.join(BASE, "model/feature_cols.pkl"), "rb"))
metrics       = json.load(open(os.path.join(BASE, "model/metrics.json")))

relu     = lambda z: np.maximum(0, z)
def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)

def predict_proba(X_scaled):
    A = X_scaled.copy()
    for i in range(len(W) - 1):
        A = relu(A @ W[i] + b[i])
    return softmax(A @ W[-1] + b[-1])

# Label mapping dari nama asli dataset
LABEL_MAP = {
    "Insufficient_Weight": "Kurus",
    "Normal_Weight":        "Normal",
    "Overweight_Level_I":   "Kegemukan I",
    "Overweight_Level_II":  "Kegemukan II",
    "Obesity_Type_I":       "Obesitas Tipe I",
    "Obesity_Type_II":      "Obesitas Tipe II",
    "Obesity_Type_III":     "Obesitas Tipe III",
}

RISK_INFO = {
    "Insufficient_Weight": {
        "color": "#63b3ed", "icon": "🥗", "level": "rendah",
        "desc": "Berat badan Anda di bawah normal (Kurus). Disarankan meningkatkan asupan nutrisi bergizi.",
        "tips": ["Konsultasikan dengan ahli gizi", "Tingkatkan asupan kalori bergizi", "Latihan kekuatan ringan", "Pantau berat badan rutin"]
    },
    "Normal_Weight": {
        "color": "#68d391", "icon": "✅", "level": "normal",
        "desc": "Berat badan Anda dalam rentang ideal. Pertahankan gaya hidup sehat!",
        "tips": ["Pertahankan pola makan seimbang", "Olahraga teratur 150 menit/minggu", "Tidur cukup 7–9 jam", "Kelola stres dengan baik"]
    },
    "Overweight_Level_I": {
        "color": "#f6ad55", "icon": "⚠️", "level": "sedang",
        "desc": "Berat badan sedikit melebihi normal (Kegemukan Level I). Perhatikan pola makan.",
        "tips": ["Kurangi asupan kalori 300 kkal/hari", "Tingkatkan aktivitas fisik", "Hindari makanan olahan", "Konsultasikan dengan dokter"]
    },
    "Overweight_Level_II": {
        "color": "#ed8936", "icon": "⚠️", "level": "sedang",
        "desc": "Berat badan melebihi normal (Kegemukan Level II). Segera perbaiki gaya hidup.",
        "tips": ["Program diet terstruktur", "Olahraga kardio rutin", "Batasi gula & lemak jenuh", "Cek kesehatan berkala"]
    },
    "Obesity_Type_I": {
        "color": "#fc8181", "icon": "🚨", "level": "tinggi",
        "desc": "Obesitas Tipe I. Risiko penyakit jantung dan diabetes meningkat.",
        "tips": ["Segera konsultasi dokter", "Program diet dengan ahli gizi", "Aktivitas fisik bertahap", "Monitor tekanan darah & gula darah"]
    },
    "Obesity_Type_II": {
        "color": "#e53e3e", "icon": "🚨", "level": "tinggi",
        "desc": "Obesitas Tipe II. Risiko kesehatan serius. Penanganan medis sangat disarankan.",
        "tips": ["Konsultasi dokter segera", "Program penurunan berat badan medis", "Hindari aktivitas berat tanpa pengawasan", "Pantau kondisi kesehatan intensif"]
    },
    "Obesity_Type_III": {
        "color": "#9b2335", "icon": "🆘", "level": "sangat tinggi",
        "desc": "Obesitas Tipe III (Morbid). Kondisi darurat medis. Segera hubungi tenaga medis.",
        "tips": ["Penanganan medis segera", "Kemungkinan perlu intervensi bedah", "Dukungan psikologis dibutuhkan", "Pemantauan medis intensif"]
    },
}

@app.route("/")
def index():
    return render_template("index.html", metrics=metrics)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        d = request.get_json()

        # Encode categorical inputs
        def enc(col, val):
            try:
                return float(cat_encoders[col].transform([val])[0])
            except:
                return 0.0

        age    = float(d["age"])
        gender = enc("Gender", d["gender"])
        height = float(d["height"]) / 100.0  # cm to m
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

        proba      = predict_proba(X_sc)[0]
        pred_idx   = int(np.argmax(proba))
        pred_raw   = le.classes_[pred_idx]
        pred_label = LABEL_MAP.get(pred_raw, pred_raw)
        confidence = round(float(proba[pred_idx]) * 100, 2)
        all_proba  = {LABEL_MAP.get(cls, cls): round(float(p)*100, 2)
                      for cls, p in zip(le.classes_, proba)}
        info = RISK_INFO.get(pred_raw, RISK_INFO["Normal_Weight"])

        bmi = weight / (height ** 2)

        return jsonify({
            "success": True,
            "result": pred_label,
            "result_raw": pred_raw,
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

if __name__ == "__main__":
    app.run(debug=True, port=5000)
