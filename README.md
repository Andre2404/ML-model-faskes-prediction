# 🩺 Prediksi Rujukan FKTP untuk Pasien Diabetes Melitus

Aplikasi berbasis Machine Learning untuk memprediksi **fasilitas kesehatan tujuan rujukan** pasien diabetes melitus berdasarkan data kunjungan di FKTP (Fasilitas Kesehatan Tingkat Pertama), menggunakan data BPJS Kesehatan.

## 🔍 Fitur Prediksi
Model memprediksi **tiga aspek rujukan** sekaligus:
- **Jenis faskes tujuan** (Puskesmas, Klinik Pratama, Dokter Umum)
- **Tipe faskes tujuan** (RS Kelas B/C, RS Swasta Setara C, dll.)
- **Poli tujuan** (Penyakit Dalam, Diabetes Melitus, Jantung, dll.)

## 📊 Data & Model
- **Sumber data**: [BPJS Kesehatan 2022 — FKTP](https://drive.google.com/file/d/1-uGGCc7ETdILLe0fkqGugRw7Xy2Mx5xF/view)
- **Target**: `FKP19`, `FKP20`, `FKP21`
- **Fitur utama**: Poli asal (`FKP11`), diagnosis (`FKP15`), segmen peserta, dan indikator diabetes
- **Model**: `MultiOutputClassifier(RandomForest)`
- **Akurasi rata-rata**: ~63% (F1-score lebih informatif karena imbalance)

## 🚀 Cara Menjalankan (Lokal)

### Prasyarat
- Python ≥ 3.9
- Git

### Instalasi
```bash
git clone https://github.com/namamu/prediksi-rujukan-fktp.git
cd prediksi-rujukan-fktp
pip install -r requirements.txt