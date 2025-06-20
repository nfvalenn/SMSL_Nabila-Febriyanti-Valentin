import requests
import json

# 🔸 Mapping input kategori → angka (sesuai dengan yang dipakai saat preprocessing training)
gender_mapping = {"Male": 0, "Female": 1, "Other": 2}
yes_no_mapping = {"Yes": 1, "No": 0}
work_interfere_mapping = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3, "Always": 4}
care_options_mapping = {"Yes": 1, "No": 0, "Not sure": 2}
wellness_mapping = {"Yes": 1, "No": 0, "Don't know": 2}
seek_help_mapping = {"Yes": 1, "No": 0, "Don't know": 2}
anonymity_mapping = {"Yes": 1, "No": 0, "Don't know": 2}
leave_mapping = {
    "Very easy": 0, "Somewhat easy": 1, "Don't know": 2,
    "Somewhat difficult": 3, "Very difficult": 4
}
consequence_mapping = {"Yes": 1, "No": 0, "Maybe": 2}
coworker_mapping = {"Yes": 1, "No": 0, "Some of them": 2}
mental_vs_physical_mapping = {"Yes": 1, "No": 0, "Don't know": 2}

def get_user_input():
    print("\n=== Input Data Mental Health Prediction ===")
    data = {
        "Age": int(input("Umur: ")),
        "Gender": gender_mapping[input("Gender (Male/Female/Other): ")],
        "self_employed": yes_no_mapping[input("Apakah wiraswasta? (Yes/No): ")],
        "family_history": yes_no_mapping[input("Ada riwayat keluarga terkait mental health? (Yes/No): ")],
        "work_interfere": input("Mental health mengganggu kerja? (Never/Sometimes/Often/Always): "),
        "no_employees": input("Jumlah karyawan di tempat kerja (contoh: 6-25): "),
        "remote_work": yes_no_mapping[input("Kerja remote? (Yes/No): ")],
        "tech_company": yes_no_mapping[input("Bekerja di perusahaan teknologi? (Yes/No): ")],
        "benefits": yes_no_mapping[input("Ada fasilitas kesehatan mental? (Yes/No): ")],
        "care_options": care_options_mapping[input("Ada pilihan perawatan mental health? (Yes/No/Not sure): ")],
        "wellness_program": wellness_mapping[input("Ada program kesehatan mental? (Yes/No/Don't know): ")],
        "seek_help": seek_help_mapping[input("Perusahaan memudahkan cari bantuan? (Yes/No/Don't know): ")],
        "anonymity": anonymity_mapping[input("Privasi karyawan dijaga? (Yes/No/Don't know): ")],
        "leave": leave_mapping[input("Kebijakan cuti untuk kesehatan mental (Very easy/Somewhat easy/Don't know/Somewhat difficult/Very difficult): ")],
        "mental_health_consequence": consequence_mapping[input("Kesehatan mental berpengaruh buruk pada karir? (Yes/No/Maybe): ")],
        "phys_health_consequence": consequence_mapping[input("Kesehatan fisik berpengaruh buruk pada karir? (Yes/No/Maybe): ")],
        "coworkers": coworker_mapping[input("Dukungan dari rekan kerja? (Yes/No/Some of them): ")],
        "supervisor": coworker_mapping[input("Dukungan dari atasan? (Yes/No/Some of them): ")],
        "mental_health_interview": consequence_mapping[input("Nyaman bicara mental health saat interview? (Yes/No/Maybe): ")],
        "phys_health_interview": consequence_mapping[input("Nyaman bicara kesehatan fisik saat interview? (Yes/No/Maybe): ")],
        "mental_vs_physical": mental_vs_physical_mapping[input("Perlakuan sama antara mental dan fisik? (Yes/No/Don't know): ")],
        "obs_consequence": yes_no_mapping[input("Pernah mengalami konsekuensi karena isu mental? (Yes/No): ")]
    }
    
    work_interfere_enc = {"Never": 0, "Sometimes": 1, "Often": 2, "Always": 3}
    data["work_interfere"] = work_interfere_enc[data["work_interfere"]]

    return data

def predict(data):
    url = "http://127.0.0.1:5000/invocations" 

    headers = {'Content-Type': 'application/json'}
    payload = {
        "inputs": [data]
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            hasil = response.json()
            pred = hasil.get("predictions", hasil)[0]
            print("\n=== Hasil Prediksi ===")
            print(f"Prediksi: {'Perlu Treatment' if pred == 1 else 'Tidak Perlu Treatment'}")
        else:
            print("❗ ERROR - Gagal melakukan prediksi:")
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"❗ ERROR - Gagal terhubung ke API: {e}")

if __name__ == "__main__":
    data = get_user_input()
    predict(data)
