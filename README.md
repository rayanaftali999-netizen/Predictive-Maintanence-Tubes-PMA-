# Predictive Maintenance

Proyek Tugas Besar Pengganti UAS **ABK4ABB3 Pembelajaran Mesin dan Aplikasi**
Semester Genap 2025/2026.

Sistem klasifikasi untuk memprediksi kegagalan mesin industri (predictive
maintenance) berdasarkan data sensor (suhu, kecepatan rotasi, torsi, keausan
alat), menggunakan dataset resmi **AI4I 2020 Predictive Maintenance Dataset**
(UCI Machine Learning Repository).

🔗 **Live Demo:** [2mmixpedtdwevg7wuexrmd.streamlit.app](https://2mmixpedtdwevg7wuexrmd.streamlit.app)
🔗 **Repository:** [github.com/rayanaftali999-netizen/Predictive-Maintanence-Tubes-PMA-](https://github.com/rayanaftali999-netizen/Predictive-Maintanence-Tubes-PMA-)

## 🎯 Ringkasan Proyek
- **Masalah:** Klasifikasi biner — apakah mesin akan gagal (*machine failure*) berdasarkan data sensor real-time?
- **Dataset:** AI4I 2020, 10.000 baris, 3.39% failure rate (imbalanced)
- **Model pembanding:** Random Forest vs XGBoost, dibandingkan terhadap baseline (DummyClassifier)
- **Metrik evaluasi:** Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix
- **Deployment:** Web App interaktif (Streamlit) — Opsi B

## 📊 Dataset

| Atribut | Nilai |
|---|---|
| Jumlah sampel | 10.000 |
| Jumlah fitur asli | 14 kolom |
| Failure rate | 3.39% (337 dari 10.000 — sangat imbalanced) |
| Split | Train 6.999 / Validation 1.501 / Test 1.500 (stratified) |

**Fitur yang digunakan untuk training** (9 fitur, setelah feature engineering):
```
Type, Air temperature [K], Process temperature [K], Rotational speed [rpm],
Torque [Nm], Tool wear [min], Temp diff [K], Power [W], Wear*Torque
```
Tiga fitur terakhir (`Temp diff`, `Power`, `Wear*Torque`) adalah hasil *feature
engineering* berbasis pengetahuan domain (selisih suhu, daya mekanik, dan
interaksi keausan×torsi) yang merepresentasikan mekanisme fisik kegagalan mesin.

Kolom label individual penyebab kegagalan (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`)
**tidak diikutkan** sebagai fitur karena akan menyebabkan *data leakage*
(kolom tersebut secara langsung membentuk label `Machine failure`).

## 📈 Hasil Eksperimen

### Baseline
Sebagai pembanding minimum, digunakan `DummyClassifier` (selalu menebak kelas
mayoritas/"OK"):

| Metrik | Nilai |
|---|---|
| Accuracy | 0.9660 |
| F1-score | 0.0000 |

> Baseline mencapai akurasi 96.6% tanpa mempelajari pola apa pun, namun **gagal
> total** mendeteksi kegagalan (F1=0). Ini menunjukkan accuracy adalah metrik
> yang menyesatkan pada data imbalanced seperti ini — recall dan F1-score
> jauh lebih relevan untuk dievaluasi.

### Hyperparameter Tuning (GridSearchCV, 5-fold Stratified CV)

| Model | Best Parameters | Best CV F1 |
|---|---|---|
| Random Forest | `n_estimators=200, max_depth=None, min_samples_split=2, class_weight='balanced'` | 0.8425 (±0.0567) |
| XGBoost | `n_estimators=200, max_depth=7, learning_rate=0.1` | 0.8099 (±0.0324) |

### Evaluasi pada Test Set (1.500 sampel, 51 kasus failure)

| Metrik | Random Forest | XGBoost |
|---|---|---|
| Accuracy | 0.9933 | 0.9913 |
| Precision | 0.9020 | 0.8519 |
| Recall | 0.9020 | 0.9020 |
| **F1-score** | **0.9020** | 0.8762 |
| ROC-AUC | 0.9740 | **0.9922** |

**Confusion Matrix — Random Forest**
```
                Predicted OK   Predicted Failure
True OK             1444               5
True Failure          5              46
```

**Confusion Matrix — XGBoost**
```
                Predicted OK   Predicted Failure
True OK             1441               8
True Failure          5              46
```

### 🏆 Model Terpilih: **Random Forest** (F1 = 0.9020)

Random Forest dipilih sebagai model produksi karena memberikan F1-score
tertinggi pada threshold operasional (0.5) — presisi dan recall yang seimbang
(90.2% / 90.2%), dengan hanya 5 false negative (kasus failure yang terlewat)
dan 5 false positive (false alarm) dari 1.500 sampel uji.

XGBoost mencatat ROC-AUC lebih tinggi (0.9922 vs 0.9740), yang berarti model
ini lebih unggul dalam meranking probabilitas secara umum di seluruh
kemungkinan threshold. Namun pada threshold default 0.5 yang dipakai sistem
saat deployment, precision XGBoost lebih rendah (85.2%), menghasilkan lebih
banyak false alarm dibanding Random Forest. Trade-off ini menjadi catatan
penting: pemilihan threshold klasifikasi dapat dioptimalkan lebih lanjut
sesuai kebutuhan operasional (mis. menurunkan threshold untuk memprioritaskan
recall jika biaya kegagalan tak terdeteksi sangat tinggi).

## 🗂️ Struktur Repo
```
.
├── ai4i2020.csv          # Dataset resmi AI4I 2020 (UCI)
├── train.py              # EDA, preprocessing, training, tuning, evaluasi
├── app.py                # Streamlit web app (Deployment Opsi B)
├── api.py                # FastAPI REST API (Deployment Opsi A)
├── requirements.txt
├── model.pkl, scaler.pkl, metrics.json   # Artefak model terlatih
└── confusion_matrix.png, roc_curves.png, feature_importance.png
```

## 🚀 Cara Menjalankan

### 1. Instalasi
```bash
pip install -r requirements.txt
```

### 2. (Opsional) Latih ulang model
```bash
python train.py
```

### 3a. Jalankan Web App (Streamlit) — lokal
```bash
streamlit run app.py
```
Buka http://localhost:8501

## 🌐 Deployment Publik
Aplikasi telah di-deploy dan dapat diakses publik melalui **Streamlit
Community Cloud**:

👉 **https://2mmixpedtdwevg7wuexrmd.streamlit.app**

Aplikasi terhubung langsung ke repository GitHub ini — setiap perubahan kode
yang di-push ke branch `main` akan otomatis ter-update di URL deployment.

## 📦 Sumber Dataset
Dataset: **AI4I 2020 Predictive Maintenance Dataset**, Matzka, S. (2020),
UCI Machine Learning Repository (CC BY 4.0).
https://archive.ics.uci.edu/dataset/601

## 👥 Anggota Kelompok
- Nama, NIM
- Nama, NIM
- Nama, NIM
- Nama, NIM
