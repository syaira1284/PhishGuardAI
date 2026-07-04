import matplotlib
# Sangat Penting: Gunakan 'Agg' agar matplotlib bisa berjalan di server web tanpa error tampilan
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import shap
import io
import base64
import json  # TAMBAHAN 1: Import json wajib ditambahkan untuk menjalankan patch config
from feature_extraction import extract_features, normalize_url

app = Flask(__name__)

# Load model dan urutan kolom
with open('xgboost_phishing_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('feature_columns.pkl', 'rb') as f:
    X_columns = pickle.load(f)

# =======================================================
# TAMBAHAN 2: FIX SHAP & XGBOOST 2.0 (PATCH BOOSTER)
# =======================================================
# Mengambil config asli dari model XGBoost
booster = model.get_booster()
original_save_config = booster.save_config

def patched_save_config(*args, **kwargs):
    # Ambil konfigurasi, lalu ubah format string JSON ke dictionary Python
    config_str = original_save_config(*args, **kwargs)
    config = json.loads(config_str)
    try:
        # Cari nilai base_score yang menyebabkan error
        val = str(config["learner"]["learner_model_param"]["base_score"])
        
        # Jika ada kurung siku (misal: "[0.5]"), buang kurungnya menjadi "0.5"
        if val.startswith('[') and val.endswith(']'):
            config["learner"]["learner_model_param"]["base_score"] = val.strip('[]')
    except Exception:
        pass
        
    # Kembalikan lagi ke format string JSON
    return json.dumps(config)

# Pasang jebakan: Ganti fungsi bawaan dengan fungsi patch kita
booster.save_config = patched_save_config

# Buat explainer SHAP satu kali di awal agar proses cepat (Sekarang akan berhasil)
explainer = shap.TreeExplainer(model)

# Kembalikan fungsi ke normal agar tidak mengganggu proses XGBoost selanjutnya
booster.save_config = original_save_config
# =======================================================


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
        
        classes = list(model.classes_)
        phishing_idx   = classes.index(0) 
        legitimate_idx = classes.index(1) 
        
        phishing_prob = float(round(prob[0][phishing_idx] * 100, 1))
        legitimate_prob = float(round(prob[0][legitimate_idx] * 100, 1))
        
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
        sv = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
        
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