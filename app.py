import streamlit as st
import joblib
import numpy as np

# โหลดโมเดลที่บันทึกไว้
model = joblib.load("optimized_diabetes_model.pkl")

st.title("แอปวินิจฉัยโรคเบาหวานด้วย Machine Learning")

st.markdown("กรอกข้อมูลสุขภาพของคุณเพื่อประเมินความเสี่ยงเป็นโรคเบาหวาน")

# สร้างฟอร์มกรอกข้อมูล
with st.form(key='diabetes_form'):
    pregnancies = st.number_input("จำนวนครั้งที่ตั้งครรภ์", min_value=0, max_value=20, step=1)
    glucose = st.number_input("ระดับน้ำตาลในเลือด (Glucose)", min_value=0)
    blood_pressure = st.number_input("ความดันโลหิต (BloodPressure)", min_value=0)
    skin_thickness = st.number_input("ความหนาผิวหนัง (SkinThickness)", min_value=0)
    insulin = st.number_input("ระดับอินซูลิน (Insulin)", min_value=0)
    bmi = st.number_input("BMI", min_value=0.0, format="%.1f")
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
    age = st.number_input("อายุ", min_value=0, max_value=120, step=1)

    submit_button = st.form_submit_button(label='ทำนายผล')

if submit_button:
    # เตรียมข้อมูลอินพุต
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, diabetes_pedigree, age]])
    
    prediction = model.predict(input_data)
    proba = model.predict_proba(input_data)[0][1]

    if prediction[0] == 1:
        st.error(f"มีความเสี่ยงที่จะเป็นโรคเบาหวาน (ความมั่นใจ {proba:.2%})")
    else:
        st.success(f"มีโอกาสน้อยที่จะเป็นโรคเบาหวาน (ความมั่นใจ {1-proba:.2%})")
