import matplotlib
import numpy as np
# Sangat Penting: Gunakan 'Agg' agar matplotlib bisa berjalan di server web tanpa error tampilan
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import pickle
import shap
import io
import base64
from feature_extraction import extract_features, normalize_url

app = Flask(__name__)

# 1. KEMBALI MENGGUNAKAN PICKLE (Agar tidak perlu injeksi manual n_classes_)
with open('xgboost_phishing_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load urutan kolom
X_columns = joblib.load("feature_columns.pkl")

# 2. TETAP GUNAKAN BYPASS BOOSTER UNTUK SHAP (Agar terhindar dari bug base_score)
explainer = shap.TreeExplainer(model.get_booster())

# Kamus Terjemahan untuk Orang Awam
TERJEMAHAN_FITUR = {
    "NoOfSubDomain": "Struktur Subdomain",
    "URLLength": "Panjang Keseluruhan Tautan",
    "NoOfDegitsInURL": "Jumlah Angka pada Tautan",
    "Entropy": "Keacakan Susunan Huruf",
    "HyphenCount": "Penggunaan Tanda Hubung (-)",
    "SuspiciousWordCount": "Adanya Kata Mencurigakan (login, update, dll)",
    "TLDLength": "Panjang Akhiran Domain (.com, .id, dll)",
    "DomainLength": "Panjang Nama Website Utama",
    "DegitRatioInURL": "Proporsi/Rasio Angka",
    "NoOfLettersInURL": "Jumlah Huruf"
}

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        url_input = request.form['url']

        # 1. Ekstrak & Prediksi
        features = extract_features(url_input)
        features_df = pd.DataFrame([features])
        features_df = features_df[X_columns]

        pred = int(model.predict(features_df)[0])
        prob = model.predict_proba(features_df)

        phishing_prob = float(round(prob[0][0] * 100, 1))
        legitimate_prob = float(round(prob[0][1] * 100, 1))

        if pred == 1:
            status = "AMAN (LEGITIMATE)"
            confidence = legitimate_prob
        else:
            status = "BERBAHAYA (PHISHING)"
            confidence = phishing_prob

        # ==========================================
        # 2. PROSES XAI (SHAP) REAL-TIME
        # ==========================================
        shap_values = explainer(features_df)
        sv = shap_values[0]

        # Buat Gambar Grafik
        plt.figure(figsize=(7, 4))
        shap.plots.waterfall(sv, show=False)
        plt.title("Analisis Faktor Penentu Keputusan AI", pad=10, fontsize=12)
        plt.tight_layout()

        # Convert Gambar ke Base64 String
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=120)
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
        plt.close() # Tutup plot agar RAM tidak bocor

        # ==========================================
        # 3. BUAT PENJELASAN TEKS UNTUK AWAM
        # ==========================================
        feature_names = features_df.columns.tolist()
        vals = sv.values
        impacts = sorted(zip(feature_names, vals), key=lambda x: abs(x[1]), reverse=True)

        # Nilai negatif (v < 0) menarik ke arah 0 (Phishing) -> Ini adalah Alasan Bahaya
        top_red_flags = [f for f, v in impacts if v < 0][:2]

        # Nilai positif (v > 0) mendorong ke arah 1 (Aman) -> Ini adalah Alasan Aman
        top_green_flags = [f for f, v in impacts if v > 0][:2]

        alasan_teks = ""
        if pred == 0: # Phishing
            alasan_teks = "Sistem mencurigai tautan ini TERUTAMA karena: "
            alasan_teks += " dan ".join([TERJEMAHAN_FITUR.get(f, f) for f in top_red_flags]) + ". "
            if top_green_flags:
                alasan_teks += "Meskipun " + " dan ".join([TERJEMAHAN_FITUR.get(f, f) for f in top_green_flags]) + " terlihat normal, itu tidak cukup membuktikan tautan ini aman."
        else:
            alasan_teks = "Sistem menilai tautan ini aman TERUTAMA karena: "
            alasan_teks += " dan ".join([TERJEMAHAN_FITUR.get(f, f) for f in top_green_flags]) + " terlihat normal dan meyakinkan."

        # Kirim hasil lengkap ke Web
        return jsonify({
            'status': status,
            'confidence': confidence,
            'url_normalized': normalize_url(url_input),
            'shap_image': img_base64,
            'shap_text': alasan_teks
        })

    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()})

if __name__ == '__main__':
    app.run(debug=True)
