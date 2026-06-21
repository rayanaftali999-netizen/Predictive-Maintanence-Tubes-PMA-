# Predictive Maintenance — Deteksi Kerusakan Mesin

Proyek Tugas Besar Pengganti UAS **ABK4ABB3 Pembelajaran Mesin dan Aplikasi**
Semester Genap 2025/2026.

Sistem klasifikasi untuk memprediksi kegagalan mesin industri (predictive
maintenance) berdasarkan data sensor (suhu, kecepatan rotasi, torsi, keausan
alat). Mengikuti struktur dataset **AI4I 2020 Predictive Maintenance** (UCI).

## 🎯 Ringkasan
- **Masalah:** Klasifikasi biner — apakah mesin akan gagal? (imbalanced, ~7% failure)
- **Model pembanding:** Random Forest vs XGBoost, dibandingkan dengan baseline DummyClassifier
- **Metrik:** Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
- **Deployment:** REST API (FastAPI) **dan** Web App (Streamlit)

## 📊 Hasil (test set)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| Baseline (most_frequent) | 0.927 | — | 0.000 | 0.000 | — |
| Random Forest | 0.987 | 0.960 | 0.864 | **0.909** | 0.986 |
| XGBoost | 0.982 | 0.874 | 0.882 | 0.878 | 0.991 |

> Baseline accuracy 92.7% dengan F1=0 menunjukkan mengapa accuracy menyesatkan
> pada data tidak seimbang — F1 dan recall adalah metrik yang tepat di sini.

## 🗂️ Struktur Repo
```
.
├── ai4i2020.csv          # Dataset (sintetis, ganti dengan data UCI asli bila perlu)
├── generate_dataset.py   # Generator dataset
├── train.py              # EDA, preprocessing, training, tuning, evaluasi
├── app.py                # Streamlit web app (Deployment Opsi B)
├── api.py                # FastAPI REST API (Deployment Opsi A)
├── curl_examples.sh      # Contoh pengujian endpoint
├── requirements.txt
├── model.pkl, scaler.pkl, metrics.json   # Artefak hasil training
└── confusion_matrix.png, roc_curves.png, feature_importance.png
```

## 🚀 Cara Menjalankan

### 1. Instalasi
```bash
pip install -r requirements.txt
```

### 2. (Opsional) Regenerasi dataset & latih ulang model
```bash
python generate_dataset.py
python train.py
```

### 3a. Jalankan Web App (Streamlit)
```bash
streamlit run app.py
```
Buka http://localhost:8501

### 3b. Jalankan REST API (FastAPI)
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```
- Dokumentasi interaktif: http://localhost:8000/docs
- Inference lokal: lihat `curl_examples.sh`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Type":"L","Air temperature [K]":298.0,"Process temperature [K]":308.0,"Rotational speed [rpm]":1320,"Torque [Nm]":65.0,"Tool wear [min]":220}'
```

## 🌐 Deployment Publik (gratis)
- **Streamlit:** push repo ke [Streamlit Community Cloud](https://share.streamlit.io) atau Hugging Face Spaces (SDK: streamlit)
- **FastAPI:** deploy ke [Render](https://render.com) / [Railway](https://railway.app), start command: `uvicorn api:app --host 0.0.0.0 --port $PORT`
- **TODO kelompok:** tempel URL deployment publik di sini → `https://...`

## 📦 Dataset
Versi yang disertakan adalah data **sintetis** dengan logika kegagalan fisik yang
sama seperti AI4I 2020 (TWF/HDF/PWF/OSF/RNF). Untuk memakai data asli, unduh
`ai4i2020.csv` dari [UCI](https://archive.ics.uci.edu/dataset/601) atau Kaggle
(lisensi CC BY 4.0) dan letakkan di folder ini — seluruh kode tetap berjalan.

## 👥 Anggota Kelompok
- Nama, NIM
- Nama, NIM
- Nama, NIM
- Nama, NIM
