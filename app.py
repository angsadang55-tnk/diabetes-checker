import streamlit as st
import sys
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import plotly.express as px  

# ทำให้ลิงก์เปลี่ยนหน้าได้
st.markdown("""
<script>
window.addEventListener("message", (event) => {
    if (event.data.type === "setPage") {
        window.parent.postMessage(
            { type: "streamlit:setComponentValue", data: event.data.page },
            "*"
        );
    }
});
</script>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    .input-error input {
        border: 2px solid #ff4d4d !important;
        background: #ffe6e6 !important;
    }
</style>
""", unsafe_allow_html=True)


# โหลดโมเดล
model = joblib.load("optimized_diabetes_model.pkl")

def logout_button():
    if st.sidebar.button("ออกจากระบบ"):
        st.session_state['logged_in'] = False
        st.session_state['user'] = None
        # แทนที่ st.experimental_rerun() ด้วยการรีโหลดโดยใช้ sys.exit()
        st.rerun()

def save_result(result, user_input):
    df = pd.DataFrame([{
        'user': st.session_state['user'],
        'datetime': datetime.now(),
        'result': result,
        **user_input
    }])
    df.to_csv("results.csv", mode='a', index=False, header=not pd.io.common.file_exists("results.csv"))

def load_users():
    try:
        return pd.read_csv("users.csv")
    except FileNotFoundError:
        df = pd.DataFrame(columns=["username", "password"])
        df.to_csv("users.csv", index=False)
        return df

def save_users(df):
    df.to_csv("users.csv", index=False)

if "page" not in st.session_state:
    st.session_state.page = "login"

def login_or_register():
    user_df = load_users()

    if st.session_state.page == "login":
        st.markdown("""
            <div style="display:flex; justify-content:center; align-items:center; height:90vh;">
                <div class="center-box">
                    <div style="text-align:center;">
                        <img src="https://cdn-icons-png.flaticon.com/512/2965/2965879.png" 
                             width="90" style="margin-bottom:10px;">
                        <h2>เข้าสู่ระบบ</h2>
                    </div>
        """, unsafe_allow_html=True)

        username = st.text_input("ชื่อผู้ใช้")
        password = st.text_input("รหัสผ่าน", type="password")

        if st.button("เข้าสู่ระบบ"):
            if ((user_df['username'] == username) & (user_df['password'] == password)).any():
                st.session_state['logged_in'] = True
                st.session_state['user'] = username
                st.success("เข้าสู่ระบบสำเร็จ")
                st.rerun()
            else:
                st.error("❌ ชื่อผู้ใช้หรือรหัสผ่านผิด")

        # ลิงก์ไปหน้า Register (ไม่ใช่ปุ่ม)
        st.markdown("""
            <p style="text-align:center; margin-top:15px;">
                ยังไม่มีบัญชี?
                <a href='?page=register' style='color:#0059b3; font-weight:bold;'>
                    สมัครสมาชิก
                </a>
            </p>
        """, unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

    elif st.session_state.page == "register":
        register_page()

def register_page():
    user_df = load_users()

    st.markdown("""
        <div style="display:flex; justify-content:center; align-items:center; height:90vh;">
            <div class="center-box">
                <div style="text-align:center;">
                    <img src="https://cdn-icons-png.flaticon.com/512/9131/9131529.png" 
                        width="85" style="margin-bottom:10px;">
                    <h2>สมัครสมาชิก</h2>
                </div>
    """, unsafe_allow_html=True)

    new_username = st.text_input("ชื่อผู้ใช้ใหม่")
    new_password = st.text_input("รหัสผ่านใหม่", type="password")
    confirm = st.text_input("ยืนยันรหัสผ่าน", type="password")

    if st.button("สมัครสมาชิก"):
        ...
    
    st.markdown("""
        <p style="text-align:center; margin-top:15px;">
            มีบัญชีแล้ว?
            <a href='?page=login' style='color:#0059b3; font-weight:bold;'>
                กลับไปเข้าสู่ระบบ
            </a>
        </p>
            </div>
        </div>
    """, unsafe_allow_html=True)



def profile_page():
    st.title("โปรไฟล์ของฉัน")
    user_df = load_users()
    current_user = st.session_state['user']
    user_data = user_df[user_df['username'] == current_user].iloc[0]

    st.write(f"**ชื่อผู้ใช้ปัจจุบัน:** {current_user}")

    with st.form(key='profile_form'):
        new_username = st.text_input("เปลี่ยนชื่อผู้ใช้", value=current_user)
        current_password = st.text_input("กรอกรหัสผ่านปัจจุบัน", type="password")
        new_password = st.text_input("รหัสผ่านใหม่ (ถ้าต้องการเปลี่ยน)", type="password")
        confirm_password = st.text_input("ยืนยันรหัสผ่านใหม่", type="password")
        submit = st.form_submit_button("บันทึกการเปลี่ยนแปลง")

    if submit:
        if current_password != user_data['password']:
            st.error("รหัสผ่านปัจจุบันไม่ถูกต้อง")
            return
        if new_username != current_user and new_username in user_df['username'].values:
            st.error("ชื่อผู้ใช้นี้มีอยู่แล้ว กรุณาเลือกชื่ออื่น")
            return
        if new_password or confirm_password:
            if new_password != confirm_password:
                st.error("รหัสผ่านใหม่กับการยืนยันไม่ตรงกัน")
                return
            if new_password.strip() == "":
                st.error("กรุณากรอกรหัสผ่านใหม่ให้ถูกต้อง")
                return
        idx = user_df.index[user_df['username'] == current_user][0]
        user_df.at[idx, 'username'] = new_username
        if new_password:
            user_df.at[idx, 'password'] = new_password
        save_users(user_df)
        st.session_state['user'] = new_username
        st.success("บันทึกข้อมูลโปรไฟล์เรียบร้อยแล้ว")

def diabetes_page():
    st.title("ระบบวินิจฉัยโรคเบาหวานด้วย Machine Learning")
    st.markdown("กรอกข้อมูลสุขภาพของคุณเพื่อประเมินความเสี่ยงเป็นโรคเบาหวาน")

    with st.form(key='diabetes_form'):
        pregnancies = st.number_input("จำนวนครั้งที่ตั้งครรภ์", min_value=0, max_value=20, step=1)
        glucose = st.number_input("ระดับน้ำตาลในเลือด (Glucose)", min_value=0)
        blood_pressure = st.number_input("ความดันโลหิต (BloodPressure)", min_value=0)
        skin_thickness = st.number_input("ความหนาผิวหนัง (SkinThickness)", min_value=0)
        insulin = st.number_input("ระดับอินซูลิน (Insulin)", min_value=0)
        weight = st.number_input("น้ำหนัก (กิโลกรัม)", min_value=1.0, step=0.1, format="%.1f")
        height_cm = st.number_input("ส่วนสูง (เซนติเมตร)", min_value=30.0, step=0.1, format="%.1f")

        if weight > 0 and height_cm > 0:
            height_m = height_cm / 100
            bmi = weight / (height_m ** 2)
            st.write(f"ค่า BMI ของคุณคือ: **{bmi:.2f}**")
        else:
            st.info("กรุณากรอกน้ำหนักและส่วนสูงเพื่อคำนวณ BMI")

        diabetes_pedigree = st.number_input(
            "ความเสี่ยงจากพันธุกรรม (Diabetes Pedigree Function)",
            min_value=0.0, format="%.3f",
            help="ค่านี้บ่งบอกความเสี่ยงจากประวัติครอบครัว ยิ่งสูงยิ่งเสี่ยง (ค่าปกติประมาณ 0-2)"
        )

        st.caption("""
        **ค่า Diabetes Pedigree Function** เป็นตัวชี้วัดความเสี่ยงโรคเบาหวานจากประวัติครอบครัว  
        - 0.0 - 0.2 : ความเสี่ยงต่ำ  
        - 0.2 - 0.5 : ความเสี่ยงปานกลาง  
        - มากกว่า 0.5 : ความเสี่ยงสูง  
        """)

        age = st.number_input("อายุ", min_value=0, max_value=120, step=1)
        submit_button = st.form_submit_button(label='ทำนายผล')

    if submit_button:

    # ตรวจว่ามีช่องไหนที่ยังไม่กรอก
        required_fields = [
            pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, weight, height_cm, diabetes_pedigree, age
        ]

        if any(v == 0 or v == "" for v in required_fields):
            st.error("⚠ กรุณากรอกข้อมูลให้ครบถ้วนก่อนทำการทำนายผล")
            return
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                insulin, bmi, diabetes_pedigree, age]])
        prediction = model.predict(input_data)
        proba = model.predict_proba(input_data)[0][1]

        if prediction[0] == 1:
            st.error(f"มีความเสี่ยงที่จะเป็นโรคเบาหวาน (ความมั่นใจ {proba:.2%})")
            st.warning("คำแนะนำ: ควรลดน้ำตาลในอาหาร และออกกำลังกายเพิ่ม")
            result_text = 'เสี่ยง'
        else:
            st.success(f"มีโอกาสน้อยที่จะเป็นโรคเบาหวาน (ความมั่นใจ {1-proba:.2%})")
            st.info("คำแนะนำ: รักษาสุขภาพให้ดีต่อเนื่อง ออกกำลังกายและควบคุมอาหาร")
            result_text = 'ไม่เสี่ยง'

        user_input = {
            'pregnancies': pregnancies,
            'glucose': glucose,
            'blood_pressure': blood_pressure,
            'skin_thickness': skin_thickness,
            'insulin': insulin,
            'weight': weight,
            'height_cm': height_cm,
            'bmi': bmi,
            'diabetes_pedigree': diabetes_pedigree,
            'age': age,
        }
        save_result(result_text, user_input)

def history_page():
    st.title("📊 ผลการทำนายย้อนหลังและแนวโน้มสุขภาพ")

    try:
        df = pd.read_csv("results.csv")
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df[df['user'] == st.session_state['user']]

        if df.empty:
            st.info("ยังไม่มีข้อมูลผลทำนายย้อนหลัง กรุณาไปที่เมนู 'วินิจฉัยโรคเบาหวาน' เพื่อทำการทำนายครั้งแรก")
            return

        # 1. การเลือกช่วงวันที่
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("วันที่เริ่มต้น", value=df['datetime'].min().date())
        with col2:
            end_date = st.date_input("วันที่สิ้นสุด", value=df['datetime'].max().date())

        if start_date > end_date:
            st.error("วันที่เริ่มต้นต้องไม่เกินวันที่สิ้นสุด")
            return

        mask = (df['datetime'].dt.date >= start_date) & (df['datetime'].dt.date <= end_date)
        filtered_df = df.loc[mask].sort_values(by='datetime', ascending=True).reset_index(drop=True)

        if filtered_df.empty:
            st.info("ไม่มีข้อมูลในช่วงวันที่ที่เลือก")
            return
        
        st.markdown("---")
        
        # 2. กราฟเส้นแสดงแนวโน้ม (Glucose & BMI)
        st.subheader("📈 แนวโน้มระดับน้ำตาลและค่า BMI ตามเวลา")
        
        chart_data = filtered_df[['datetime', 'glucose', 'bmi']].copy()
        chart_data['date_only'] = chart_data['datetime'].dt.date
        
        # กราฟเส้นสำหรับ Glucose
        fig_glucose = px.line(
            chart_data, 
            x='datetime', 
            y='glucose', 
            title='ระดับน้ำตาลในเลือด (Glucose) ย้อนหลัง',
            labels={'datetime': 'วันที่/เวลา', 'glucose': 'ระดับน้ำตาล'},
            markers=True
        )
        fig_glucose.update_layout(xaxis_title="วันที่/เวลา", yaxis_title="ระดับน้ำตาลในเลือด (mg/dL)", hovermode="x unified")
        fig_glucose.update_traces(line=dict(color='#4da3ff'), marker=dict(color='#1f82e8', size=8))
        st.plotly_chart(fig_glucose, use_container_width=True)

        # กราฟเส้นสำหรับ BMI
        fig_bmi = px.line(
            chart_data, 
            x='datetime', 
            y='bmi', 
            title='ค่าดัชนีมวลกาย (BMI) ย้อนหลัง',
            labels={'datetime': 'วันที่/เวลา', 'bmi': 'BMI'},
            markers=True
        )
        fig_bmi.update_layout(xaxis_title="วันที่/เวลา", yaxis_title="ค่า BMI", hovermode="x unified")
        fig_bmi.update_traces(line=dict(color='#00bfa5'), marker=dict(color='#00897b', size=8))
        st.plotly_chart(fig_bmi, use_container_width=True)
        
        st.markdown("---")

        # 3. กราฟแท่งแสดงสัดส่วนผลทำนาย
        st.subheader("📊 สัดส่วนผลการทำนาย")
        
        result_counts = filtered_df['result'].value_counts().reset_index()
        result_counts.columns = ['Result', 'Count']
        
        color_map = {'เสี่ยง': '#ff4d4d', 'ไม่เสี่ยง': '#4da3ff'}
        
        fig_bar = px.bar(
            result_counts, 
            x='Result', 
            y='Count', 
            title='จำนวนผลทำนาย (เสี่ยง vs ไม่เสี่ยง)',
            color='Result',
            color_discrete_map=color_map,
            text='Count'
        )
        fig_bar.update_layout(xaxis_title="ผลการทำนาย", yaxis_title="จำนวนครั้ง")
        st.plotly_chart(fig_bar, use_container_width=True)
        
        st.markdown("---")

        # 4. แสดงตารางข้อมูลดิบ
        st.subheader("📄 ตารางข้อมูลดิบ")
        display_df = filtered_df[['datetime', 'result', 'glucose', 'blood_pressure', 'bmi', 'age']]
        display_df.columns = ['วันที่/เวลา', 'ผลทำนาย', 'น้ำตาลในเลือด', 'ความดันโลหิต', 'BMI', 'อายุ']
        
        st.dataframe(
            display_df.sort_values(by='วันที่/เวลา', ascending=False),
            column_config={
                "วันที่/เวลา": st.column_config.DatetimeColumn("วันที่/เวลา", format="YYYY-MM-DD HH:mm"),
            },
            use_container_width=True
        )

    except FileNotFoundError:
        st.warning("ยังไม่มีข้อมูลผลทำนายย้อนหลัง")

def about_page():
    st.header("📘 เกี่ยวกับโรคเบาหวาน")
    st.markdown("""
    **โรคเบาหวาน (Diabetes Mellitus)** คือโรคที่เกิดจากระดับน้ำตาลในเลือดสูงผิดปกติจากการที่ร่างกายผลิตอินซูลินได้น้อย หรือใช้อินซูลินไม่ได้ผล

    ### 🔍 ประเภทของโรคเบาหวาน
    - **ชนิดที่ 1**: ร่างกายไม่สามารถผลิตอินซูลินได้เลย
    - **ชนิดที่ 2**: ร่างกายตอบสนองต่ออินซูลินได้น้อยลง (พบมากที่สุด)
    - **เบาหวานขณะตั้งครรภ์**: มักเกิดชั่วคราวในหญิงตั้งครรภ์

    ### 📈 ปัจจัยเสี่ยง
    - น้ำหนักเกินหรืออ้วน
    - ขาดการออกกำลังกาย
    - พันธุกรรม
    - อายุ 35 ปีขึ้นไป
    """)

    with st.expander("📌 สัญญาณเตือนโรคเบาหวาน"):
        st.markdown("""
        - ปัสสาวะบ่อยโดยเฉพาะตอนกลางคืน
        - กระหายน้ำบ่อย
        - น้ำหนักลดโดยไม่ทราบสาเหตุ
        - อ่อนเพลีย เหนื่อยง่าย
        - สายตาพร่ามัว
        """)

    with st.expander("🍽️ อาหารที่ควรหลีกเลี่ยง"):
        st.markdown("""
        - น้ำอัดลม น้ำหวาน
        - ขนมหวาน เบเกอรี่
        - ข้าวขัดขาว ขนมปังขาว
        - ของทอด ไขมันสูง
        """)

    with st.expander("✅ อาหารที่ควรทาน"):
        st.markdown("""
        - ข้าวกล้อง ธัญพืชไม่ขัดสี
        - ผักใบเขียว ผลไม้หวานน้อย
        - ปลา อกไก่
        - ดื่มน้ำเปล่าให้เพียงพอ
        """)

    with st.expander("🧠 คำถามที่พบบ่อย (FAQ)"):
        st.markdown("""
        **Q:** โรคเบาหวานรักษาหายไหม?  
        **A:** ไม่หายขาด แต่สามารถควบคุมระดับน้ำตาลได้ด้วยการดูแลตัวเอง

        **Q:** ต้องงดของหวานทั้งหมดไหม?  
        **A:** ไม่จำเป็น แต่อยู่ในปริมาณที่พอเหมาะ และเลือกของหวานจากธรรมชาติ

        **Q:** เบาหวานทำให้เกิดภาวะแทรกซ้อนอะไรได้บ้าง?  
        **A:** อาจเกิดโรคหัวใจ, ไตวาย, ตาบอด, แผลเรื้อรัง
        """)

    st.markdown("---")

    st.video("https://youtu.be/Y0rkx5M-hsg")

    st.markdown("""
    ---
    ### 🔗 แหล่งข้อมูลเพิ่มเติม
    - [กรมควบคุมโรค - เบาหวาน](https://ddc.moph.go.th/)
    - [สมาคมโรคเบาหวานไทย](https://www.dmthai.org/)
    - [World Health Organization (WHO)](https://www.who.int/news-room/fact-sheets/detail/diabetes)
    """)

# เริ่มต้นแอป
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "page" not in st.session_state:
    st.session_state.page = "login"
    
# ดึงค่าจากลิงก์ เช่น ?page=register
query_params = st.query_params
if "page" in query_params:
    st.session_state.page = query_params["page"]

if not st.session_state['logged_in']:
    login_or_register()
    st.stop()

# แสดงปุ่มออกจากระบบทุกหน้า
logout_button()

page = st.sidebar.selectbox("เมนู", ["วินิจฉัยโรคเบาหวาน", "ผลย้อนหลัง", "เกี่ยวกับโรคเบาหวาน", "โปรไฟล์ผู้ใช้"])

if page == "วินิจฉัยโรคเบาหวาน":
    diabetes_page()
elif page == "ผลย้อนหลัง":
    history_page()
elif page == "เกี่ยวกับโรคเบาหวาน":
    about_page()
elif page == "โปรไฟล์ผู้ใช้":
    profile_page()

