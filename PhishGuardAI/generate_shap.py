import pickle
import pandas as pd
import shap
import matplotlib.pyplot as plt
from feature_extraction import extract_features, normalize_url

# ==========================================
# 1. LOAD MODEL & URUTAN FITUR
# ==========================================
print("Memuat model XGBoost...")
with open('xgboost_phishing_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('feature_columns.pkl', 'rb') as f:
    X_columns = pickle.load(f)

# ==========================================
# 2. INPUT URL YANG INGIN DIANALISIS
# ==========================================
# Ganti URL ini dengan URL yang ingin kamu jadikan contoh di skripsi
url_input = "https://tinyurl.com/ut-vinsdessert"
print(f"Mengekstrak fitur dari: {url_input}")

features = extract_features(url_input)
features_df = pd.DataFrame([features])
features_df = features_df[X_columns] # Samakan urutan kolom

# ==========================================
# 3. PROSES EXPLAINABLE AI (SHAP)
# ==========================================
print("Menghitung nilai SHAP...")
# Karena XGBoost adalah model berbasis pohon (Tree), kita gunakan TreeExplainer
explainer = shap.TreeExplainer(model)
shap_values = explainer(features_df)

# ==========================================
# 4. MEMBUAT VISUALISASI GRAFIK (WATERFALL PLOT)
# ==========================================
print("Membuat grafik...")
plt.figure(figsize=(10, 6))

# Membuat Waterfall Plot untuk data ke-0 (karena kita hanya input 1 URL)
# shap_values bisa berbentuk list (tergantung versi xgboost/shap), kita ambil indeks kelas target
if isinstance(shap_values, list):
    shap.plots.waterfall(shap_values[1][0], show=False) # Kelas 1 (atau sesuaikan)
else:
    shap.plots.waterfall(shap_values[0], show=False)

plt.title(f"Analisis Keputusan AI (SHAP) untuk URL:\n{url_input}", pad=20)
plt.tight_layout()

# Simpan grafik menjadi file gambar PNG beresolusi tinggi (cocok untuk skripsi)
nama_file = "grafik_shap_analisis.png"
plt.savefig(nama_file, dpi=300, bbox_inches='tight')
print(f"\nSukses! Grafik analisis berhasil disimpan sebagai '{nama_file}'.")