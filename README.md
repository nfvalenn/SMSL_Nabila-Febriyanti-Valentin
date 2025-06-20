# 🧠 Mental Health Prediction with MLOps Workflow

Proyek ini merupakan implementasi **Machine Learning untuk prediksi kebutuhan treatment kesehatan mental**, lengkap dengan pelacakan eksperimen, logging manual, dan integrasi dengan DagsHub.

---

## 📌 Tujuan Proyek

* Melatih model machine learning untuk **memperkirakan apakah seseorang membutuhkan treatment kesehatan mental atau tidak**.
* **Melakukan hyperparameter tuning** untuk mendapatkan model terbaik.
* Menerapkan konsep **MLOps** dengan pelacakan eksperimen berbasis **MLflow**.
* Melakukan **monitoring dan logging** hasil model untuk kebutuhan evaluasi.

---

## 🗂️ Struktur Direktori

```
SMSL_Nabila-Febriyanti-Valentin/
├── Eksperimen_Nabila-Febriyanti-Valentin.txt
│
├── Membangun_model/
│   ├── mlruns
|   ├── modelling.py              
│   ├── modelling_tuning.py        
|   ├── requirements.txt
|   ├── mental_health_cleaned.csv
|   ├── Screenshot-artifak
|   ├── Screenshot-Dashboard
|   ├── Dagshub.txt
│   └── .env.example
│
├── Monitoring_dan_Logging/
|   ├── bukti alreting Grafana
|   ├── bukti monitoring Grafana
|   ├── bukti monitoring Prometheus
|   ├── Bukti_serving.png
|   ├── inference.py
|   ├── prometheus_exporter.py
│   └── prometheus.yml        
├── Workflow-CI.txt                   
└── README.md
```

---

## 📄 Dataset

Dataset yang digunakan merupakan hasil dari **preprocessing** survei kesehatan mental di industri teknologi.

| Fitur                       | Deskripsi                                                      |
| --------------------------- | -------------------------------------------------------------- |
| Age                       | Usia responden                                                 |
| Gender                    | Jenis kelamin                                                  |
| self_employed             | Apakah wiraswasta                                              |
| family_history            | Ada riwayat keluarga terkait kesehatan mental                  |
| treatment                 | *Target* → Apakah pernah mendapat perawatan kesehatan mental |
| work_interfere            | Pengaruh kesehatan mental terhadap pekerjaan                   |
| no_employees              | Jumlah karyawan di tempat kerja                                |
| remote_work               | Bekerja remote atau tidak                                      |
| tech_company              | Apakah bekerja di perusahaan teknologi                         |
| benefits                  | Apakah perusahaan menyediakan fasilitas kesehatan mental       |
| care_options              | Tersedia atau tidak pilihan perawatan                          |
| wellness_program          | Apakah perusahaan punya program kesehatan mental               |
| seek_help                 | Kemudahan mencari bantuan terkait kesehatan mental             |
| anonymity                 | Privasi karyawan terkait masalah kesehatan mental              |
| leave                     | Kemudahan cuti untuk alasan kesehatan mental                   |
| mental_health_consequence | Dampak kesehatan mental terhadap karir                         |
| phys_health_consequence   | Dampak kesehatan fisik terhadap karir                          |
| coworkers                 | Nyaman berbicara dengan rekan kerja terkait kesehatan mental   |
| supervisor                | Nyaman berbicara dengan atasan terkait kesehatan mental        |
| mental_health_interview   | Nyaman bicara soal mental health saat interview kerja          |
| phys_health_interview     | Nyaman bicara soal kesehatan fisik saat interview              |
| mental_vs_physical        | Apakah mental health diperlakukan setara fisik health          |
| obs_consequence           | Pernah mengalami konsekuensi terkait masalah mental health     |

📌 *Target*: treatment

Dataset hasil preprocessing dapat dilihat di:
📁 [`preprocessing/mental_health_cleaned.csv`](https://github.com/nfvalenn/SMSL_Nabila-Febriyanti-Valentin/blob/main/preprocessing/mental_health_cleaned.csv)

---

## ⚙️ Komponen Proyek

### 1️⃣ **Pembangunan Model**

* **modelling.py**:
  → Melatih model **RandomForestClassifier** dengan **MLflow autolog**.

* **modelling\_tuning.py** (Skilled/Advanced):
  → Melakukan **hyperparameter tuning** dengan **GridSearchCV**.
  → Menggunakan **manual logging MLflow** untuk menyimpan parameter, akurasi, precision, recall, dan artefak model.

---

### 2️⃣ **Monitoring & Logging**

* **predict\_request.py** → Digunakan untuk menguji endpoint prediksi dari model yang sudah disajikan menggunakan `mlflow models serve` atau API lain.

* **MLflow Tracking URI**:
  📌 `https://dagshub.com/nfvalenn/mental-health-Nabila-Febriyanti-Valentin.mlflow`

* **Monitoring CI/CD (Jika Diaktifkan)** → Bisa ditambahkan di `.github/workflows/ci.yml`.

---

### 3️⃣ **Integrasi Eksternal**

| Komponen      | Endpoint                                                                                         |
| ------------- | -------------------------------------------------------------------------------------------------|
| MLflow Server | [dagshub](https://dagshub.com/nfvalenn/mental-health-Nabila-Febriyanti-Valentin.mlflow)        |
| Model Serving | [jika lokal](http://127.0.0.1:5000/)                                                           |
| Repository    | [Repositori Eksperimen](https://github.com/nfvalenn/Eksperimen_Nabila-Febriyanti-Valentinn.git)  |
|               |  [Repositori Workflow CI](https://github.com/nfvalenn/Workflow_CI.git)                           |

---

## 🖥️ Cara Menjalankan

### 🔧 1. Instalasi Dependensi

```bash
pip install -r requirements.txt
```

### 🔐 2. Buat File `.env`

```bash
MLFLOW_TRACKING_USERNAME=your_dagshub_username
MLFLOW_TRACKING_PASSWORD=your_dagshub_token
```

### ✅ 3. Jalankan Model

```bash
# Training model baseline
python modelling.py

# Hyperparameter tuning (Advanced)
python modelling_tuning.py
```

---

##  Contoh Penggunaan

Jalankan prediksi ke endpoint model (jika menggunakan model serving):

```bash
python "Monitoring dan Logging/inference.py"
```

pantauan premotheus
```bash
http://127.0.0.1:8000/metriks
```

---

## 📚 Referensi

* [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
* [DagsHub](https://dagshub.com/)
* [Scikit-Learn](https://scikit-learn.org/)
  
---

## 👩‍💻 Kontributor

**Nabila Febriyanti Valentin**
GitHub: [@nfvalenn](https://github.com/nfvalenn)
