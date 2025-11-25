from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load Model
try:
    model = joblib.load('model_stroke.pkl')
    COL_ORDER = model.feature_names_in_ 
except:
    COL_ORDER = [] 

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/analisa')
def analisa():
    return render_template('analisa.html')

@app.route('/edukasi')
def edukasi():
    return render_template('edukasi.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. AMBIL DATA (Dengan Nama)
        nama_pasien = request.form['nama']
        usia = float(request.form['usia'])
        gula = float(request.form['gula'])
        bmi = float(request.form['bmi'])
        hipertensi = int(request.form['hipertensi'])
        jantung = int(request.form['jantung'])
        gender = request.form['gender']
        rokok = request.form['rokok']

        # 2. Susun Data
        data_dict = {
            'age': usia, 'avg_glucose_level': gula, 'bmi': bmi,
            'hypertension': hipertensi, 'heart_disease': jantung,
            'ever_married': 1, 'Residence_type': 1, 'work_type_Private': 1
        }
        data_dict[f'gender_{gender}'] = 1
        data_dict[f'smoking_status_{rokok}'] = 1

        # 3. Reindex
        input_df = pd.DataFrame([data_dict])
        input_df = input_df.reindex(columns=COL_ORDER, fill_value=0)

        # 4. Prediksi AI
        raw_prob = model.predict_proba(input_df)[0][1]

        # 5. Hybrid Logic
        adj = 0.0
        if jantung == 1: adj += 0.35
        if hipertensi == 1: adj += 0.25
        if usia > 50 and rokok == 'smokes': adj += 0.15
        
        final_prob = min(raw_prob + adj, 0.99)
        persen = round(final_prob * 100, 1)

        # 6. Hasil
        if final_prob > 0.40:
            status = "BERISIKO TINGGI"
            warna_alert = "danger"
            icon = "⚠️"
            pesan = "Terdeteksi indikasi risiko kesehatan signifikan. Segera konsultasi dokter."
        else:
            status = "AMAN / NORMAL"
            warna_alert = "success"
            icon = "✅"
            pesan = "Kondisi kesehatan terpantau baik. Pertahankan pola hidup sehat."

        # Kirim Nama Balik ke HTML
        return render_template('analisa.html', 
                               nama=nama_pasien, 
                               hasil=status, 
                               skor=f"{persen}%", 
                               css=warna_alert, 
                               icon=icon, 
                               saran=pesan)

    except Exception as e:
        return render_template('analisa.html', hasil="ERROR", skor="0%", css="warning", saran=str(e))

if __name__ == '__main__':
    app.run(debug=True)