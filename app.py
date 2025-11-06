import streamlit as st
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Judul aplikasi
st.set_page_config(page_title="Prediksi Rujukan FKTP", page_icon="🩺")
st.title("🩺 Prediksi Rujukan FKTP - Diabetes Melitus")
st.write("Masukkan data kunjungan pasien di FKTP untuk prediksi fasilitas rujukan.")

# Load model & komponen
def load_assets():
    model = joblib.load('model_rujukan_fktp_rf.pkl')
    preprocessor = joblib.load('preprocessor_fktp.pkl') 
    label_encoders = joblib.load('label_encoders_fktp.pkl')
    label_mappings = joblib.load('label_mappings_fktp.pkl')
    return model, preprocessor, label_encoders, label_mappings
    
    # Rebuild preprocessor (hindari error pickle & versi scikit-learn)
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
             ['FKP08', 'FKP09', 'FKP11', 'FKP12', 'FKP22']),
            ('num', 'passthrough', ['bulan_kunjungan', 'hari_kunjungan', 'is_diabetes'])
        ]
    )
    
    # Fit dengan dummy data agar siap pakai
    dummy = pd.DataFrame([{
        'FKP08': 1, 'FKP09': 1, 'FKP11': 1, 'FKP12': 1, 'FKP22': 1,
        'bulan_kunjungan': 1, 'hari_kunjungan': 0, 'is_diabetes': 0
    }])
    preprocessor.fit(dummy)
    
    return model, preprocessor, label_encoders, label_mappings

# Load aset
try:
    model, preprocessor, label_encoders, label_mappings = load_assets()
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.stop()

# Input form
st.subheader("Formulir Data Kunjungan")

FKP08 = st.selectbox(
    "Jenis FKTP",
    options=[1, 2, 3],
    format_func=lambda x: {1: "Puskesmas", 2: "Klinik Pratama", 3: "Dokter Umum"}.get(x, f"Kode {x}")
)

FKP09 = st.selectbox(
    "Tipe FKTP",
    options=[1, 2, 3, 4, 5],
    format_func=lambda x: {
        1: "Klinik Non Rawat Inap",
        2: "Dokter Perorangan",
        3: "Non Rawat Inap",
        4: "Rawat Inap",
        5: "Klinik Rawat Inap"
    }.get(x, f"Kode {x}")
)

FKP11 = st.selectbox(
    "Poli FKTP",
    options=[1, 8, 22, 27],
    format_func=lambda x: {
        1: "Poli Umum",
        8: "Penyakit Dalam",
        22: "Poli Diabetes",
        27: "TB & Paru"
    }.get(x, f"Kode {x}")
)

FKP12 = st.selectbox(
    "Segmen Peserta",
    options=[1, 2, 3, 4, 5],
    format_func=lambda x: {
        1: "Bukan pekerja",
        2: "PBI APBN",
        3: "PBI APBD",
        4: "PBPU",
        5: "PPU"
    }.get(x, f"Kode {x}")
)

FKP22 = st.selectbox(
    "Jenis Kunjungan",
    options=[1, 2],
    format_func=lambda x: {1: "Sakit", 2: "Sehat"}.get(x, f"Kode {x}")
)

bulan = st.slider("Bulan Kunjungan", 1, 12, 6)
hari = st.slider("Hari Kunjungan (0=Senin, 6=Minggu)", 0, 6, 2)
is_dm = st.checkbox("Diagnosis Diabetes Mellitus (E10–E14)?", value=True)

# Tombol prediksi
if st.button("🔍 Prediksi Rujukan"):
    input_df = pd.DataFrame([{
        'FKP08': FKP08,
        'FKP09': FKP09,
        'FKP11': FKP11,
        'FKP12': FKP12,
        'FKP22': FKP22,
        'bulan_kunjungan': bulan,
        'hari_kunjungan': hari,
        'is_diabetes': 1 if is_dm else 0
    }])
    
    try:
        X_processed = preprocessor.transform(input_df)
        pred_encoded = model.predict(X_processed)[0]
        
        # Decode ke label teks
        hasil = {}
        for i, col in enumerate(['FKP19', 'FKP20', 'FKP21']):
            kode_asli = label_encoders[col].inverse_transform([int(pred_encoded[i])])[0]
            teks = label_mappings[col].get(kode_asli, f"Kode {kode_asli} tidak dikenali")
            hasil[col] = teks
        
        st.success("✅ Prediksi Berhasil!")
        st.markdown(f"**Jenis Faskes Tujuan**: {hasil['FKP19']}")
        st.markdown(f"**Tipe Faskes Tujuan**: {hasil['FKP20']}")
        st.markdown(f"**Poli Tujuan**: {hasil['FKP21']}")
        
    except Exception as e:
        st.error(f"❌ Error saat prediksi: {e}")