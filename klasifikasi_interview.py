import pickle
import streamlit as st

# membaca model
model = pickle.load(open('klasifikasi_interview.sav', 'rb'))

#judul web
st.title('Prediksi Interview Kandidat Karyawan')

#membagi kolom
col1, col2 = st.columns(2)

with col1 :
    years_of_experience = st.slider("Jumlah tahun pengalaman yang dimiliki kandidat di bidangnya",1,30, step=1)
    functional_competency_score = st.number_input ('Skor yang mewakili kompetensi fungsional kandidat berdasarkan tes')
    top1_skills_score = st.number_input ('Skor keterampilan paling berharga yang dimiliki kandidat')
    top2_skills_score = st.number_input ('Skor keterampilan paling berharga kedua yang dimiliki kandidat')
    top3_skills_score = st.number_input ('Skor keterampilan paling berharga ketiga yang dimiliki kandidat')

with col2 :
    behavior_competency_score = st.number_input ('Skor yang mewakili kompetensi perilaku kandidat yang diperoleh dari tes SDM')
    top1_behavior_skill_score = st.number_input ('Skor keterampilan perilaku paling berharga yang dimiliki kandidat')
    top2_behavior_skill_score = st.number_input ('Skor keterampilan perilaku paling berharga kedua yang dimiliki kandidat')
    top3_behavior_skill_score = st.number_input ('Skor dari keterampilan perilaku paling berharga ketiga yang dimiliki kandidat')

# code untuk prediksi
predict = ''

# membuat tombol untuk prediksi
if st.button('Test Prediksi Interview'):
    prediction = model.predict([[years_of_experience, functional_competency_score,
                                       top1_skills_score, top2_skills_score, top3_skills_score,
                                       behavior_competency_score, top1_behavior_skill_score,
                                       top2_behavior_skill_score, top3_behavior_skill_score]])

    if(prediction[0] == 1):
        predict = 'Kandidat Dipanggil Untuk Interview'
    else:
        predict = 'Kandidat Tidak Dipanggil Untuk Interview'
st.success(predict)
