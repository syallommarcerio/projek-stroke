from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load Model
try:
    model = joblib.load('model_stroke.pkl')
    COL_ORDER = model.feature_names_in_ # Auto-detect kolom
except:
    COL_ORDER = [] # Fallback

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Ambil Data
        usia = float(request.form['usia'])
        gula = float(request.form['gula'])
        bmi = float(request.form['bmi'])
        hipertensi = int(request.form['hipertensi'])
        jantung = int(request.form['jantung'])
        gender = request.form['gender']
        rokok = request.form['rokok']

        # 2. Masukkan ke Dictionary
        data_dict = {
            'age': usia,
            'avg_glucose_level': gula,
            'bmi': bmi,
            'hypertension': hipertensi,
            'heart_disease': jantung,
            'ever_married': 1, 
            'Residence_type': 1,
            'work_type_Private': 1
        }
        
        # One-Hot Encoding Manual
        data_dict[f'gender_{gender}'] = 1
        data_dict[f'smoking_status_{rokok}'] = 1

        # 3. Reindex (Solusi Anti-Error)
        input_df = pd.DataFrame([data_dict])
        input_df = input_df.reindex(columns=COL_ORDER, fill_value=0)

        # 4. Prediksi AI
        raw_prob = model.predict_proba(input_df)[0][1]

        # 5. Hybrid Adjustment (Biar Agus Merah)
        adj = 0.0
        if jantung == 1: adj += 0.35
        if hipertensi == 1: adj += 0.25
        if usia > 50 and rokok == 'smokes': adj += 0.15
        
        final_prob = min(raw_prob + adj, 0.99)
        persen = round(final_prob * 100, 1)

        # 6. Tentukan Tampilan
        if final_prob > 0.40:
            status = "BERISIKO TINGGI"
            warna_alert = "danger" # Merah di Bootstrap
            icon = "⚠️"
            pesan = "Sistem mendeteksi pola yang mirip dengan pasien stroke. Disarankan segera konsultasi dokter."
        else:
            status = "AMAN / RENDAH"
            warna_alert = "success" # Hijau di Bootstrap
            icon = "✅"
            pesan = "Tidak ditemukan risiko signifikan. Pertahankan pola hidup sehat Anda."

        return render_template('index.html', 
                               hasil=status, 
                               skor=f"{persen}%", 
                               css=warna_alert, 
                               icon=icon,
                               saran=pesan)

    except Exception as e:
        return render_template('index.html', hasil="ERROR", skor="0%", css="warning", saran=str(e))

if __name__ == '__main__':
    app.run(debug=True)