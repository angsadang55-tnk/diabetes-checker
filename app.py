#1. ส่วนการนำเข้าไลบรารี (Import Libraries)
import firebase_admin
from firebase_admin import credentials, firestore, auth
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import plotly.express as px

#2. ฟังก์ชันประเมินระดับความเสี่ยง
def get_risk_status(glucose, prediction):
    if prediction == "เสี่ยง" and glucose >= 126: 
        return "🔴 เสี่ยงสูง (น้ำตาลวิกฤต)"
    elif prediction == "เสี่ยง": 
        return "🟡 เฝ้าระวัง"
    else:
        return "🟢 ปกติ"
    
# 3. ส่วนตกแต่งหน้าตาเว็บไซต์ (UI & CSS)
def inject_custom_css():
    st.markdown("""
    
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;700&display=swap');
    
    /* ปรับฟอนต์ทั้งหน้าเว็บ */
    html, body, [class*="css"] {
        font-family: 'Sarabun', sans-serif;
    }

    /* ส่วนหัวแบบ Gradient */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 40px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* ปรับแต่งปุ่ม Sidebar */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        transition: 0.3s;
    }

    /* กล่องโปรไฟล์ใน Sidebar */
    .profile-card {
        background: #ffffff;
        padding: 15px;
        border-radius: 12px;
        border-left: 5px solid #2a5298;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 15px;
        color: #333;
    }
    /* ปรับแต่งปุ่มเมนูให้ดูนุ่มนวลขึ้น */
    div.stButton > button:first-child {
        border: none;
        height: 3rem;
        background-color: #f0f2f6;
        color: #1e3c72;
        font-weight: 500;
        text-align: left;
        padding-left: 20px;
        margin-bottom: -10px;
    }
    
    /* สไตล์เมื่อปุ่มถูกเลือก (Active) */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%) !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(30, 60, 114, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)

#4. ส่วนแสดงหัวข้อแบบตกแต่ง (Styled Header)
def render_styled_header(title, subtitle):
    st.markdown(f"""
    <div class="main-header">
        <h1 style="color: white !important; margin-bottom: 10px;">{title}</h1>
        <p style="font-size: 1.1rem; opacity: 0.85;">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

#5. การจัดการสถานะผู้ใช้งาน (Session State)
inject_custom_css()
# ===== Session State Init (ต้องอยู่บนสุดก่อนใช้งาน) =====
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user" not in st.session_state:
    st.session_state.user = None

#6.การเชื่อมต่อ Firebase ---
def init_firebase():
    if not firebase_admin._apps:
        try:
            cred_info = dict(st.secrets["firebase"])
            cred = credentials.Certificate(cred_info)
            firebase_admin.initialize_app(cred)
        except Exception as e:
            st.error(f"❌ Firebase init failed: {e}")
            st.stop()
    return firestore.client()

db = init_firebase()

#7. ดึงข้อมูลโปรไฟล์ผู้ใช้งาน
def get_current_user_profile():
    user_email = st.session_state.get("user")
    if not user_email:
        return {}

    doc = db.collection("users").document(user_email).get()
    if doc.exists:
        return doc.to_dict()

    return {
        "email": user_email,
        "name": "ยังไม่ระบุชื่อ",
        "role": "user"
    }

st.markdown("""
<style>
/* ทำปุ่มล่างให้เหมือนลิงก์ */
.link-btn button {
    background: none !important;
    border: none !important;
    padding: 0 !important;
    color: #ff4d4d !important;
    font-size: 16px !important;
    text-decoration: underline;
    cursor: pointer;
}

.link-btn button:hover {
    color: #ff7b7b !important;
}
</style>
""", unsafe_allow_html=True)

# ทำให้ลิงก์เปลี่ยนหน้าได้
st.markdown("""
<style>
    .input-error input {
        border: 2px solid #ff4d4d !important;
        background: #ffe6e6 !important;
    }
</style>
""", unsafe_allow_html=True)

# 8. การโหลดโมเดล Machine Learning
@st.cache_resource
def load_model():
    with st.spinner("กำลังเตรียมระบบ..."):
        return joblib.load("optimized_diabetes_model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error("❌ ไม่พบโมเดลสำหรับทำนายผล กรุณาตรวจสอบไฟล์ optimized_diabetes_model.pkl")
    st.stop()

def logout_button():
    if st.sidebar.button("ออกจากระบบ"):
        st.session_state['logged_in'] = False
        st.session_state['user'] = None
        # แทนที่ st.experimental_rerun() ด้วยการรีโหลดโดยใช้ sys.exit()
        st.rerun()

from datetime import datetime

#9. การบันทึกผลการวิเคราะห์
def save_result(result_text, user_input):

    db.collection("results").add({
        # 🔐 ข้อมูลผู้ใช้
        "user": st.session_state.get("user"),          # email
        "name": user_profile.get("name", ""),          # ชื่อจริง
        "role": user_profile.get("role", "user"),

        # 📊 ผลการทำนาย
        "result": result_text,

        # 🩺 ข้อมูลสุขภาพ
        "pregnancies": user_input["pregnancies"],
        "glucose": user_input["glucose"],
        "blood_pressure": user_input["blood_pressure"],
        "skin_thickness": user_input["skin_thickness"],
        "insulin": user_input["insulin"],
        "weight": user_input["weight"],
        "height_cm": user_input["height_cm"],
        "bmi": user_input["bmi"],
        "diabetes_pedigree": user_input["diabetes_pedigree"],
        "age": user_input["age"],

        # ⏰ เวลา
        "datetime": datetime.now()
    })
#10. ระบบสมัครสมาชิกและเข้าสู่ระบบ
def auth_page():
    from firebase_auth import firebase_login

    if "auth_mode" not in st.session_state:
        st.session_state.auth_mode = "login"

    if st.session_state.auth_mode == "login":
        st.subheader("🔐 เข้าสู่ระบบ")

        email = st.text_input("อีเมล", key="login_email")
        password = st.text_input("รหัสผ่าน", type="password", key="login_pass")

        if st.button("เข้าสู่ระบบ"):
            if not email or not password:
                st.error("กรุณากรอกอีเมลและรหัสผ่าน")
                return

            result = firebase_login(email, password)

            if "idToken" in result:
                st.session_state.logged_in = True
                st.session_state.user = email
                st.rerun()
            else:
                st.error(result.get("error", {}).get("message", "เข้าสู่ระบบไม่สำเร็จ"))

        if st.button("ยังไม่มีบัญชี? สมัครสมาชิก"):
            st.session_state.auth_mode = "register"
            st.rerun()

    else:
        st.subheader("📝 สมัครสมาชิก")

        email = st.text_input("อีเมลใหม่", key="reg_email")
        password = st.text_input("รหัสผ่านใหม่", type="password", key="reg_pass")

        if st.button("สมัครสมาชิก"):
            if not email or not password:
                st.error("กรุณากรอกข้อมูลให้ครบ")
                return

        from firebase_admin import auth   # ← ต้องอยู่นอก try

        try:
            auth.create_user(email=email, password=password)

            db.collection("users").document(email).set({
                "email": email,
                "name": "",
                "role": "user",
                "created_at": datetime.now()
            })

            st.success("สมัครสมาชิกสำเร็จ! กรุณาเข้าสู่ระบบ")
            st.session_state.auth_mode = "login"
            st.rerun()

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาด: {e}")
                    

        if st.button("มีบัญชีแล้ว? กลับเข้าสู่ระบบ"):
            st.session_state.auth_mode = "login"
            st.rerun()
#11. หน้า “วินิจฉัยและประเมินความเสี่ยง”
def diabetes_page():
    render_styled_header("🩺 วินิจฉัยและประเมินความเสี่ยง", "วิเคราะห์สุขภาพด้วยระบบ AI จากพฤติกรรมและผลแล็บ")
    
    # --- หน้าเตรียมตัวก่อนไปตรวจ ---
    with st.expander("📝 วิธีเตรียมตัวก่อนไปตรวจเลือด (เพื่อให้ได้ค่าที่แม่นยำที่สุด)"):
        st.markdown("""
        1. **งดอาหารและเครื่องดื่มทุกชนิด** (ดื่มน้ำเปล่าได้) อย่างน้อย 8-10 ชั่วโมงก่อนตรวจ
        2. **รายการที่ควรขอตรวจ:**
            * **Fasting Glucose:** ค่าน้ำตาลปกติ
            * **HbA1c:** ค่าน้ำตาลสะสม (แนะนำมาก!)
            * **Fasting Insulin:** เพื่อดูภาวะดื้ออินซูลิน
        """)
        st.info("💡 นำค่าเหล่านี้กลับมากรอกในแอปอีกครั้งเพื่อความแม่นยำ 100%")

    with st.form(key='diabetes_form'):
        # --- ส่วนที่ 1: ข้อมูลร่างกายพื้นฐาน ---
        st.subheader("📏 1. ข้อมูลร่างกายพื้นฐาน")
        c1, c2, c3 = st.columns(3)
        with c1:
            # ปรับ value เป็น 0.0 เพื่อเช็คการกรอกข้อมูล
            weight = st.number_input("น้ำหนัก (กิโลกรัม)", min_value=0.0, value=0.0, format="%.1f")
        with c2:
            height_cm = st.number_input("ส่วนสูง (เซนติเมตร)", min_value=0.0, value=0.0, format="%.1f")
        with c3:
            age = st.number_input("อายุ (ปี)", min_value=0, step=1, value=0)

        # ตรวจสอบการกรอกข้อมูลก่อนคำนวณ
        if weight > 0 and height_cm > 0:
            bmi = weight / ((height_cm/100)**2)
            
            # กำหนดข้อความและสี (เพิ่ม color: #000 เพื่อให้เห็นตัวหนังสือชัดเจน)
            if bmi < 18.5:
                label, color, border = "น้ำหนักน้อยกว่าเกณฑ์", "#e3f2fd", "#2196f3"
            elif bmi < 23:
                label, color, border = "ปกติ สุขภาพดี", "#e8f5e9", "#4caf50"
            elif bmi < 25:
                label, color, border = "น้ำหนักเกิน/ท้วม", "#fffde7", "#fbc02d"
            else:
                label, color, border = "อ้วน/ความเสี่ยงสูง", "#ffebee", "#f44336"
            
            # แสดงผล BMI แบบเน้นตัวหนังสือสีดำชัดเจน
            st.markdown(f"""
                <div style='background-color: {color}; padding: 15px; border-radius: 10px; border-left: 8px solid {border};'>
                    <strong style='color: #000; font-size: 18px;'>📊 ผลลัพธ์ BMI: {bmi:.1f}</strong><br>
                    <span style='color: #000; font-size: 16px;'>สถานะ: {label}</span>
                </div>
            """, unsafe_allow_html=True)
        else:
            # แจ้งเตือนถ้ายังกรอกไม่ครบ
            st.warning("⚠️ กรุณากรอกข้อมูล น้ำหนัก และ ส่วนสูง เพื่อคำนวณค่า BMI")

        st.markdown("---")

        # --- ส่วนที่ 2: ข้อมูลแบบประเมินจากพฤติกรรมและอาการ ---
        st.subheader("📋 2. ข้อมูลแบบประเมินจากพฤติกรรมและอาการ")
        st.write("<small>เลือกหัวข้อที่ตรงกับพฤติกรรมหรืออาการของคุณในช่วงนี้</small>", unsafe_allow_html=True)
        col_q1, col_q2 = st.columns(2)
        with col_q1:
            q_sugar = st.checkbox("🥤 ดื่มน้ำหวานบ่อย (>3 แก้ว/สัปดาห์)")
            q_night = st.checkbox("🌙 ปัสสาวะบ่อยตอนกลางคืน (เกิน 2 ครั้ง)")
        with col_q2:
            q_wound = st.checkbox("🩹 มีแผลตามตัวแล้วหายช้าผิดปกติ")
            q_family = st.checkbox("🧬 มีคนในครอบครัวสายตรงเป็นเบาหวาน")
        
        behavior_score = sum([q_sugar, q_night, q_wound, q_family])

        st.markdown("---")

        # --- ส่วนที่ 3: ผลตรวจทางการแพทย์ ---
        st.subheader("🧪 3. ผลตรวจทางการแพทย์ (จากใบตรวจสุขภาพ)")
        st.caption("หากไม่มีผลตรวจบางรายการ ให้คงค่าเริ่มต้นที่เรากำหนดไว้ให้")
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            glucose = st.number_input("ระดับน้ำตาลในเลือด (Glucose)", value=95)
            st.caption("ดูที่ค่า FBS หรือ Fasting Glucose (ปกติ: 70-99 mg/dL)")
            
            blood_pressure = st.number_input("ความดันโลหิต (Blood Pressure)", value=80)
            st.caption("ใช้ค่าตัวล่าง (Diastolic) (ปกติ: 60-80 mmHg)")
        
        with col_m2:
            insulin = st.number_input("ระดับอินซูลิน (Insulin)", value=0)
            st.caption("หากไม่มีผลตรวจจาก รพ. ให้ใส่ 0 (ปกติ: 15-25 μU/mL)")
            
            q_preg = st.number_input("จำนวนครั้งที่ตั้งครรภ์", min_value=0, step=1, value=0)
            st.caption("นับตามจำนวนครั้งจริง (หากเป็นเพศชายให้ใส่ 0)")

        skin_thickness = 20 # ค่าคงที่มาตรฐานสำหรับ Model
        
        st.markdown("<br>", unsafe_allow_html=True)
        has_family_radio = st.radio(
            "ระดับความเข้มข้นทางพันธุกรรม (เลือกตามจำนวนญาติสายตรงที่เป็น):",
            ["ไม่มีประวัติ", "มี 1 ท่าน", "มีมากกว่า 1 ท่าน"], horizontal=True
        )
        pedigree_map = {"ไม่มีประวัติ": 0.2, "มี 1 ท่าน": 0.5, "มีมากกว่า 1 ท่าน": 0.8}
        diabetes_pedigree = pedigree_map[has_family_radio]

        submit_button = st.form_submit_button(label='🚀 วิเคราะห์ผลความเสี่ยง')

#ซ้ำกับวิเคราะห์
def diabetes_page():
    render_styled_header("วินิจฉัยและประเมินความเสี่ยง", "วิเคราะห์สุขภาพด้วยระบบ AI จากพฤติกรรมและผลแล็บ")
    
    # --- หน้าเตรียมตัวก่อนไปตรวจ ---
    with st.expander("📝 วิธีเตรียมตัวก่อนไปตรวจเลือด (เพื่อให้ได้ค่าที่แม่นยำที่สุด)"):
        st.markdown("""
        1. **งดอาหารและเครื่องดื่มทุกชนิด** (ดื่มน้ำเปล่าได้) อย่างน้อย 8-10 ชั่วโมงก่อนตรวจ
        2. **รายการที่ควรขอตรวจ:**
            * **Fasting Glucose:** ค่าน้ำตาลปกติ
            * **HbA1c:** ค่าน้ำตาลสะสม (แนะนำมาก!)
            * **Fasting Insulin:** เพื่อดูภาวะดื้ออินซูลิน
        """)
        st.info("💡 นำค่าเหล่านี้กลับมากรอกในแอปอีกครั้งเพื่อความแม่นยำ 100%")

    with st.form(key='diabetes_form'):
        # --- ส่วนที่ 1: ข้อมูลร่างกายพื้นฐาน ---
        st.subheader("📏 1. ข้อมูลร่างกายพื้นฐาน")
        c1, c2, c3 = st.columns(3)
        with c1:
            weight = st.number_input("น้ำหนัก (กิโลกรัม)", min_value=0.0, value=0.0, format="%.1f")
        with c2:
            height_cm = st.number_input("ส่วนสูง (เซนติเมตร)", min_value=0.0, value=0.0, format="%.1f")
        with c3:
            age = st.number_input("อายุ (ปี)", min_value=0, step=1, value=0)

        # ตรวจสอบการกรอกข้อมูลก่อนคำนวณ BMI
        bmi = 0.0
        if weight > 0 and height_cm > 0:
            bmi = weight / ((height_cm/100)**2)
            if bmi < 18.5: label, color, border = "น้ำหนักน้อยกว่าเกณฑ์", "#e3f2fd", "#2196f3"
            elif bmi < 23: label, color, border = "ปกติ สุขภาพดี", "#e8f5e9", "#4caf50"
            elif bmi < 25: label, color, border = "น้ำหนักเกิน/ท้วม", "#fffde7", "#fbc02d"
            else: label, color, border = "อ้วน/ความเสี่ยงสูง", "#ffebee", "#f44336"
            
            st.markdown(f"""
                <div style='background-color: {color}; padding: 15px; border-radius: 10px; border-left: 8px solid {border};'>
                    <strong style='color: #000; font-size: 18px;'>📊 ผลลัพธ์ BMI: {bmi:.1f}</strong><br>
                    <span style='color: #000; font-size: 16px;'>สถานะ: {label}</span>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ กรุณากรอกข้อมูล น้ำหนัก และ ส่วนสูง เพื่อคำนวณค่า BMI")

        st.markdown("---")

        # --- ส่วนที่ 2: ข้อมูลแบบประเมินจากพฤติกรรมและอาการ ---
        st.subheader("📋 2. ข้อมูลแบบประเมินจากพฤติกรรมและอาการ")
        st.write("<small>เลือกหัวข้อที่ตรงกับพฤติกรรมหรืออาการของคุณในช่วงนี้</small>", unsafe_allow_html=True)
        col_q1, col_q2 = st.columns(2)
        with col_q1:
            q_sugar = st.checkbox("🥤 ดื่มน้ำหวานบ่อย (>3 แก้ว/สัปดาห์)")
            q_night = st.checkbox("🌙 ปัสสาวะบ่อยตอนกลางคืน (เกิน 2 ครั้ง)")
        with col_q2:
            q_wound = st.checkbox("🩹 มีแผลตามตัวแล้วหายช้าผิดปกติ")
            q_family = st.checkbox("🧬 มีคนในครอบครัวสายตรงเป็นเบาหวาน")
        
        behavior_score = sum([q_sugar, q_night, q_wound, q_family])

        st.markdown("---")

        # --- ส่วนที่ 3: ผลตรวจทางการแพทย์ (ตั้งค่าเริ่มต้นเป็น 0) ---
        st.subheader("🧪 3. ผลตรวจทางการแพทย์ (จากใบตรวจสุขภาพ)")
        st.caption("หากไม่มีผลตรวจบางรายการ ให้คงค่าเริ่มต้นที่เรากำหนดไว้ให้")
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            glucose = st.number_input("ระดับน้ำตาลในเลือด (Glucose)", value=0)
            st.caption("ดูที่ค่า FBS หรือ Fasting Glucose (ปกติ: 70-99 mg/dL)")
            
            blood_pressure = st.number_input("ความดันโลหิต (Blood Pressure)", value=0)
            st.caption("ใช้ค่าตัวล่าง (Diastolic) (ปกติ: 60-80 mmHg)")
        
        with col_m2:
            insulin = st.number_input("ระดับอินซูลิน (Insulin)", value=0)
            st.caption("หากไม่มีผลตรวจจาก รพ. ให้ใส่ 0 (ปกติ: 15-25 μU/mL)")
            
            q_preg = st.number_input("จำนวนครั้งที่ตั้งครมภ์", min_value=0, step=1, value=0)
            st.caption("นับตามจำนวนครั้งจริง (หากเป็นเพศชายให้ใส่ 0)")

        skin_thickness = 20 # ค่าคงที่มาตรฐาน
        
        st.markdown("<br>", unsafe_allow_html=True)
        has_family_radio = st.radio(
            "ระดับความเข้มข้นทางพันธุกรรม (เลือกตามจำนวนญาติสายตรงที่เป็น):",
            ["ไม่มีประวัติ", "มี 1 ท่าน", "มีมากกว่า 1 ท่าน"], horizontal=True
        )
        pedigree_map = {"ไม่มีประวัติ": 0.2, "มี 1 ท่าน": 0.5, "มีมากกว่า 1 ท่าน": 0.8}
        diabetes_pedigree = pedigree_map[has_family_radio]

        submit_button = st.form_submit_button(label='🚀 วิเคราะห์ผลความเสี่ยง')

    # --- ส่วนประมวลผล (ต้องอยู่ระดับเดียวกับ with st.form) ---
    if submit_button:
        # ตรวจสอบข้อมูลก่อนว่ากรอกครบหรือยัง (ต้องไม่เป็น 0)
        if weight <= 0 or height_cm <= 0 or age <= 0 or glucose <= 0 or blood_pressure <= 0:
            st.error("❌ ไม่สามารถวิเคราะห์ผลได้: กรุณากรอกข้อมูลให้ครบถ้วน (น้ำหนัก, ส่วนสูง, อายุ, ระดับน้ำตาล และความดัน)")
            st.warning("💡 หากไม่ทราบค่าระดับน้ำตาลหรือความดัน แนะนำให้ใช้ค่าเฉลี่ยสุขภาพดี (Glucose: 95, Blood Pressure: 80)")
        else:
            with st.spinner("🤖 AI กำลังวิเคราะห์ข้อมูลของคุณ..."):
                input_data = np.array([[q_preg, glucose, blood_pressure, skin_thickness, 
                                        insulin, bmi, diabetes_pedigree, age]])
                
                prediction = model.predict(input_data)
                proba = model.predict_proba(input_data)[0][1]

                st.markdown("---")
                
                if prediction[0] == 1 or behavior_score >= 2:
                    st.error(f"### ⚠️ ผลการวิเคราะห์: มีความเสี่ยง")
                    st.write(f"ความน่าจะเป็นจากการประเมิน: **{proba:.1%}**")
                    
                    tips = []
                    if behavior_score >= 2: tips.append("- **สัญญาณเตือน:** พบอาการทางกายภาพที่สอดคล้องกับเบาหวาน")
                    if bmi >= 25: tips.append("- **น้ำหนัก:** ค่า BMI สูงกว่าเกณฑ์ (ภาวะอ้วน) ควรลดการทานแป้งและน้ำตาล")
                    if q_sugar: tips.append("- **พฤติกรรม:** ควรลดเครื่องดื่มรสหวานและน้ำหวานทุกชนิด")
                    
                    st.write("**คำแนะนำเบื้องต้น:**")
                    for tip in tips: st.write(tip)
                    st.warning("👨‍⚕️ ควรนัดพบแพทย์เพื่อตรวจเลือดอย่างละเอียด (HbA1c) อีกครั้ง")
                    result_status = "เสี่ยง"
                else:
                    st.success(f"### ✅ ผลการวิเคราะห์: ความเสี่ยงต่ำ")
                    st.write(f"ความน่าจะเป็นจากการประเมิน: **{proba:.1%}**")
                    st.info("💡 คำแนะนำ: รักษาสุขภาพที่ดีแบบนี้ต่อไป และตรวจสุขภาพประจำปีสม่ำเสมอ")
                    result_status = "ไม่เสี่ยง"
                
                # บันทึกข้อมูล
                save_result(result_status, {
                    "pregnancies": q_preg, "glucose": glucose, "blood_pressure": blood_pressure,
                    "skin_thickness": skin_thickness, "insulin": insulin, "bmi": bmi,
                    "diabetes_pedigree": diabetes_pedigree, "age": age, "weight": weight, "height_cm": height_cm
                })
                st.balloons()

#13.หน้าแสดงประวัติการตรวจย้อนหลัง
def history_page():
    # แทนที่ st.title("📊 ผลการทำนายย้อนหลัง...") ด้วย:
    render_styled_header("📊 ประวัติสุขภาพย้อนหลัง", "ติดตามแนวโน้มระดับน้ำตาลและค่า BMI ของคุณ")

    try:
        docs = (
            db.collection("results")
            .where("user", "==", st.session_state["user"])
            .order_by("datetime")
            .stream()
        )

        data = []
        for doc in docs:
            d = doc.to_dict()

            # ✅ ป้องกัน KeyError: datetime
            if "datetime" not in d:
                continue

            # Firestore timestamp → python datetime
            d["datetime"] = d["datetime"].replace(tzinfo=None)
            data.append(d)

        # ✅ ถ้าไม่มีข้อมูลเลย
        if not data:
            st.info("ยังไม่มีข้อมูลผลทำนายย้อนหลัง กรุณาไปที่เมนู 'วินิจฉัยโรคเบาหวาน'")
            return

        df = pd.DataFrame(data)
        df["datetime"] = pd.to_datetime(df["datetime"])

        # 1. การเลือกช่วงวันที่
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "วันที่เริ่มต้น",
                value=df["datetime"].min().date()
            )
        with col2:
            end_date = st.date_input(
                "วันที่สิ้นสุด",
                value=df["datetime"].max().date()
            )

        if start_date > end_date:
            st.error("วันที่เริ่มต้นต้องไม่เกินวันที่สิ้นสุด")
            return

        mask = (
            (df["datetime"].dt.date >= start_date) &
            (df["datetime"].dt.date <= end_date)
        )
        filtered_df = df.loc[mask].sort_values(
            by="datetime", ascending=True
        ).reset_index(drop=True)

        if filtered_df.empty:
            st.info("ไม่มีข้อมูลในช่วงวันที่ที่เลือก")
            return

        st.markdown("---")

        st.subheader("📄 ข้อมูลการตรวจทั้งหมด (เรียงจากล่าสุด)")
        display_df = filtered_df[["datetime", "result", "glucose", "bmi", "age"]].sort_values(by="datetime", ascending=False)
        st.dataframe(display_df, use_container_width=True)
        # 2. กราฟแนวโน้ม Glucose & BMI
        st.subheader("📈 แนวโน้มระดับน้ำตาลและค่า BMI ตามเวลา")

        chart_data = filtered_df[["datetime", "glucose", "bmi"]]

        fig_glucose = px.line(
            chart_data,
            x="datetime",
            y="glucose",
            title="ระดับน้ำตาลในเลือด (Glucose) ย้อนหลัง",
            markers=True
        )
        st.plotly_chart(fig_glucose, use_container_width=True)

        fig_bmi = px.line(
            chart_data,
            x="datetime",
            y="bmi",
            title="ค่าดัชนีมวลกาย (BMI) ย้อนหลัง",
            markers=True
        )
        st.plotly_chart(fig_bmi, use_container_width=True)

        st.markdown("---")

        # 3. กราฟแท่งผลทำนาย
        st.subheader("📊 สัดส่วนผลการทำนาย")

        result_counts = filtered_df["result"].value_counts().reset_index()
        result_counts.columns = ["Result", "Count"]

        fig_bar = px.bar(
            result_counts,
            x="Result",
            y="Count",
            title="จำนวนผลทำนาย (เสี่ยง vs ไม่เสี่ยง)",
            text="Count"
        )
        st.plotly_chart(fig_bar, use_container_width=True)


        st.markdown("---")

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาด: {e}")

#14. หน้าโปรไฟล์ผู้ใช้งาน        
def profile_page():
    # --- เพิ่มส่วนหัวแบบมีสไตล์ (Gradient Header) ---
    render_styled_header("👤 โปรไฟล์ของฉัน", "จัดการข้อมูลส่วนตัวและข้อมูลติดต่อเพื่อความสะดวกในการรับบริการ")
    
    # 1. ดึง ID จาก Session
    user_id = st.session_state.user['localId'] if isinstance(st.session_state.user, dict) else st.session_state.user
    user_ref = db.collection('users').document(user_id)
    doc = user_ref.get()
    u_data = doc.to_dict() if doc.exists else {}
    
    # ดึงสิทธิ์จากข้อมูลล่าสุดใน DB (เพื่อความแม่นยำในการเช็ค)
    is_admin = u_data.get("role") == "admin"

    with st.form("profile_form"):
        # ส่วนข้อมูลทั่วไป (ใช้ได้ทั้ง Admin และ User)
        st.subheader("📌 ข้อมูลพื้นฐาน")
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("ชื่อ", value=u_data.get('name', ''))
            lastname = st.text_input("นามสกุล", value=u_data.get('lastname', ''))
        
        with col2:
            phone = st.text_input("เบอร์โทรศัพท์ (สำหรับติดต่อ)", value=u_data.get('phone', ''))
            if not is_admin:
                blood_list = ["A", "B", "AB", "O", "ไม่ระบุ"]
                current_blood = u_data.get('blood_type', 'ไม่ระบุ')
                blood_index = blood_list.index(current_blood) if current_blood in blood_list else 4
                blood_type = st.selectbox("หมู่เลือด", blood_list, index=blood_index)
            else:
                # ส่วนแสดงสถานะสำหรับแอดมินเท่านั้น
                st.markdown("<br>", unsafe_allow_html=True)
                st.info("🛡️ สถานะบัญชี: ผู้ดูแลระบบ (Admin)")

        # ส่วนข้อมูลสุขภาพ (แสดงเฉพาะคนไข้)
        if not is_admin:
            st.markdown("---")
            st.subheader("🏥 ข้อมูลสุขภาพเชิงลึก (เฉพาะคนไข้)")
            c3, c4 = st.columns(2)
            with c3:
                emergency = st.text_input("ติดต่อฉุกเฉิน", value=u_data.get('emergency', ''))
            with c4:
                gender_list = ["ชาย", "หญิง", "อื่นๆ"]
                current_gender = u_data.get('gender', 'ชาย')
                gender_index = gender_list.index(current_gender) if current_gender in gender_list else 0
                gender = st.radio("เพศ", gender_list, index=gender_index, horizontal=True)
            
            disease = st.text_area("โรคประจำตัว", value=u_data.get('disease', ''))
            allergy = st.text_area("ประวัติการแพ้ยา", value=u_data.get('allergy', ''))
        
        st.markdown("<br>", unsafe_allow_html=True)
        submit = st.form_submit_button("💾 บันทึกข้อมูลโปรไฟล์")
        
        if submit:
            # เตรียมข้อมูลบันทึก
            save_data = {
                'name': name,
                'lastname': lastname,
                'phone': phone,
                'updated_at': datetime.now()
            }
            
            if not is_admin:
                save_data.update({
                    'blood_type': blood_type,
                    'emergency': emergency,
                    'gender': gender,
                    'disease': disease,
                    'allergy': allergy
                })
            
            # บันทึกด้วย merge=True
            user_ref.set(save_data, merge=True)
            
            # อัปเดตข้อมูลใน session_state ทันทีเพื่อให้ Sidebar เปลี่ยนตาม
            if 'user_profile' in st.session_state:
                st.session_state.user_profile.update(save_data)
            
            st.success("✅ บันทึกข้อมูลสำเร็จ! ระบบทำการอัปเดตข้อมูลล่าสุดของคุณแล้ว")
            st.rerun()
       
# 15.ฟังก์ชัน delete_user
def delete_user(email):
    try:
        # เรียกใช้ auth ได้เลยเพราะ import ไว้ข้างบนแล้ว
        user = auth.get_user_by_email(email)
        auth.delete_user(user.uid)
        db.collection("users").document(email).delete()
        return True
    except Exception as e:
        st.error(f"ไม่สามารถลบผู้ใช้ได้: {e}")
        return False
    
#16.หน้าผู้ดูแลระบบ (Admin)
def admin_page():
    if user_profile.get("role") != "admin":
        st.error("⛔ คุณไม่มีสิทธิ์เข้าถึงหน้านี้")
        st.stop()

    st.subheader("🛠 ระบบจัดการผู้ใช้")

    # 1. ดึงข้อมูลจาก Collection users
    users_ref = db.collection("users").stream()
    users_data = {u.id: u.to_dict() for u in users_ref}

    # 2. ดึงอีเมลทั้งหมดที่เคยมาทำนายผลจาก Collection results (เพื่อหาคนที่ตกหล่น)
    results_ref = db.collection("results").stream()
    all_emails_from_results = set()
    for r in results_ref:
        email = r.to_dict().get("user")
        if email:
            all_emails_from_results.add(email)

    # 3. รวมรายชื่อเข้าด้วยกัน
    combined_emails = set(users_data.keys()).union(all_emails_from_results)
    
    final_users = []
    for email in combined_emails:
        user_info = users_data.get(email, {})
        final_users.append({
            "email": email,
            "name": user_info.get("name", "ผู้ใช้ใหม่ (ไม่มีข้อมูลโปรไฟล์)"),
            "role": user_info.get("role", "user")
        })

    df = pd.DataFrame(final_users)
    st.subheader(f"👥 รายชื่อผู้ใช้ทั้งหมด ({len(df)} คน)")

    # แสดงรายชื่อพร้อมปุ่มลบ
    # แสดงรายชื่อผู้ใช้
    for _, row in df.iterrows():
        col1, col2, col3 = st.columns([2.5, 2.5, 2]) # ปรับขนาดคอลัมน์เพิ่มที่ว่างให้ปุ่ม
        col1.write(f"**{row['name']}**")
        col2.write(row["email"])

        with col3:
            btn_col1, btn_col2 = st.columns(2)
            
            # 1. ปุ่มตั้งรหัสผ่านใหม่
            if btn_col1.button("🔑", key=f"pw_{row['email']}", help="ตั้งรหัสผ่านใหม่"):
                st.session_state[f"reset_mode_{row['email']}"] = True

            # 2. ปุ่มลบ (เฉพาะผู้ใช้ที่ไม่ใช่ตัวเองและไม่ใช่แอดมินคนอื่น)
            if row["email"] != st.session_state.get("user") and row["role"] != "admin":
                if btn_col2.button("🗑", key=f"del_{row['email']}", help="ลบผู้ใช้"):
                    if delete_user(row["email"]):
                        st.success(f"ลบ {row['email']} แล้ว")
                        st.rerun()

        # --- ส่วนขยายสำหรับกรอกรหัสผ่านใหม่ (จะปรากฏเมื่อกดปุ่ม 🔑) ---
        if st.session_state.get(f"reset_mode_{row['email']}", False):
            with st.expander(f"🔐 ตั้งรหัสผ่านใหม่สำหรับ {row['email']}", expanded=True):
                new_pw = st.text_input("รหัสผ่านใหม่ (อย่างน้อย 6 ตัวอักษร)", type="password", key=f"input_{row['email']}")
                c1, c2 = st.columns(2)
                if c1.button("บันทึกรหัสใหม่", key=f"save_{row['email']}"):
                    if len(new_pw) < 6:
                        st.error("รหัสผ่านต้องมีความยาวอย่างน้อย 6 ตัวอักษร")
                    else:
                        try:
                            # สั่งอัปเดตรหัสผ่านใน Firebase Auth
                            user_auth = auth.get_user_by_email(row['email'])
                            auth.update_user(user_auth.uid, password=new_pw)
                            st.success("✅ เปลี่ยนรหัสผ่านเรียบร้อยแล้ว")
                            st.session_state[f"reset_mode_{row['email']}"] = False
                            # st.rerun() # เลือกจะรีรันหรือไม่ก็ได้
                        except Exception as e:
                            st.error(f"เกิดข้อผิดพลาด: {e}")
                
                if c2.button("ยกเลิก", key=f"cancel_{row['email']}"):
                    st.session_state[f"reset_mode_{row['email']}"] = False
                    st.rerun()

    st.markdown("---")
    st.subheader("🔄 เปลี่ยนสิทธิ์ผู้ใช้")

    # สร้างรายชื่อสำหรับแสดงผลในตัวเลือก (เช่น "ชื่อ - email@example.com")
    search_options = [f"{u['name']} ({u['email']})" for u in final_users]
    
    # ใช้ selectbox ซึ่ง Streamlit รองรับการพิมพ์ค้นหาในตัวอยู่แล้ว
    selected_display = st.selectbox(
        "ค้นหาชื่อหรืออีเมลที่ต้องการเปลี่ยนสิทธิ์",
        options=search_options,
        index=None,
        placeholder="พิมพ์เพื่อค้นหาชื่อหรืออีเมล..."
    )

    if selected_display:
        # ดึง email ออกมาจากข้อความที่เลือก (ค่าในวงเล็บสุดท้าย)
        target_email = selected_display.split("(")[-1].replace(")", "")
        
        # ค้นหาข้อมูลผู้ใช้ที่เลือก
        user_to_update = next((u for u in final_users if u["email"] == target_email), None)
        
        if user_to_update:
            current_role = user_to_update["role"]
            
            col_role, col_btn = st.columns([3, 1])
            with col_role:
                new_role = st.selectbox(
                    f"กำหนดสิทธิ์ใหม่สำหรับ {target_email}",
                    ["user", "admin"],
                    index=0 if current_role == "user" else 1
                )
            
            with col_btn:
                st.write("") # เว้นระยะให้ตรงกับปุ่ม
                st.write("") 
                if st.button("บันทึกสิทธิ์", use_container_width=True):
                    db.collection("users").document(target_email).set({
                        "email": target_email,
                        "role": new_role,
                        "name": user_to_update['name'] if user_to_update['name'] != "ผู้ใช้ใหม่ (ไม่มีข้อมูลโปรไฟล์)" else ""
                    }, merge=True)
                    st.success(f"✅ เปลี่ยนสิทธิ์ {target_email} เป็น {new_role} แล้ว")
                    st.rerun()

#17. ระบบจัดการข้อมูลผู้ป่วย (Admin Results)
def admin_results_page():
    if user_profile.get("role") != "admin":
        st.error("⛔ ไม่มีสิทธิ์")
        st.stop()

    render_styled_header("👨‍⚕️ ระบบบริหารจัดการข้อมูลคนไข้", "จัดการผลการคัดกรองและส่งออกรายงาน")

    # 1. โหลดข้อมูลพื้นฐาน
    users_docs = db.collection("users").stream()
    users_map = {}
    for u in users_docs:
        u_data = u.to_dict()
        if u_data.get("role") != "admin":
            users_map[u_data.get("email")] = u_data
    
    records = db.collection("results").order_by("datetime", direction=firestore.Query.DESCENDING).stream()
    
    all_results = []
    for r in records:
        d = r.to_dict()
        d["สถานะ"] = get_risk_status(d.get("glucose", 0), d.get("result"))
        p_info = users_map.get(d.get('user'), {})
        d["ชื่อ-นามสกุล"] = f"{p_info.get('name', '')} {p_info.get('lastname', '')}"
        all_results.append(d)

    if not all_results:
        st.info("ยังไม่มีข้อมูลการทำนายในระบบ"); return

    # --- [ส่วนที่ 1] สลับเอาตารางรวมและตัวกรองขึ้นมาก่อน ---
    st.subheader("📂 ตารางรวมคนไข้และจัดการข้อมูลทั้งหมด")
    full_df = pd.DataFrame(all_results)
    
    # ส่วนคัดกรองและส่งออกข้อมูล
    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        search_query = st.text_input("🔍 ค้นหาชื่อหรืออีเมลในระบบ:", key="admin_search_main")
    with col_f2:
        risk_filter = st.selectbox("🚑 กรองตามระดับความเสี่ยง:", 
                                 ["ทั้งหมด", "🔴 เสี่ยงสูง (น้ำตาลวิกฤต)", "🟡 เฝ้าระวัง", "🟢 ปกติ"])

    # ตรรกะการกรองข้อมูล
    df_to_show = full_df.copy()
    if search_query:
        df_to_show = df_to_show[df_to_show["ชื่อ-นามสกุล"].str.contains(search_query, case=False, na=False) |
                              df_to_show["user"].str.contains(search_query, case=False, na=False)]
    
    if risk_filter != "ทั้งหมด":
        df_to_show = df_to_show[df_to_show["สถานะ"] == risk_filter]

    # แสดงผลทุกคอลัมน์ (เอาคอลัมน์สำคัญไว้หน้า)
    important_cols = ["สถานะ", "ชื่อ-นามสกุล", "result", "glucose", "bmi", "datetime", "user"]
    other_cols = [c for c in df_to_show.columns if c not in important_cols]
    final_df = df_to_show[[c for c in (important_cols + other_cols) if c in df_to_show.columns]]

    st.dataframe(final_df, use_container_width=True)

    # ปุ่มดาวน์โหลด 2 รูปแบบ
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            label=f"📥 ดาวน์โหลดเฉพาะกลุ่ม: {risk_filter}",
            data=final_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"report_{risk_filter}.csv",
            mime="text/csv",
            key="dl_filtered_top"
        )
    with col_dl2:
        st.download_button(
            label="📥 ดาวน์โหลดข้อมูลทั้งหมดทุกรายการ",
            data=full_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="all_patient_data.csv",
            mime="text/csv",
            key="dl_all_top"
        )

    st.markdown("---")

    # --- [ส่วนที่ 2] ระบบค้นหาและดูประวัติรายคน (ย้ายลงมาข้างล่าง) ---
    with st.expander("🔍 ค้นหาและดูประวัติเชิงลึกรายบุคคล"):
        st.subheader("ข้อมูลคนไข้รายบุคคล")
        patient_emails = list(users_map.keys())
        selected_email = st.selectbox(
            "เลือกชื่อคนไข้เพื่อดูโปรไฟล์และประวัติ:",
            options=[""] + patient_emails,
            format_func=lambda x: f"{users_map.get(x, {}).get('name', '')} {users_map.get(x, {}).get('lastname', '')} ({x})" if x else "เลือกรายชื่อ..."
        )

        if selected_email:
            p = users_map.get(selected_email, {})
            st.markdown(f"### 👤 ข้อมูลคนไข้: {p.get('name')} {p.get('lastname')}")
            
            c1, c2 = st.columns(2)
            with c1:
                st.info(f"📞 **เบอร์โทร:** {p.get('phone', '-')}\n\n🚨 **ติดต่อฉุกเฉิน:** {p.get('emergency', '-')}")
            with c2:
                st.warning(f"💊 **โรคประจำตัว:** {p.get('disease', 'ไม่มี')}\n\n🚫 **ประวัติแพ้ยา:** {p.get('allergy', 'ไม่มี')}")

            # ตารางประวัติ 10 รายการล่าสุดของคนนั้น
            user_history = [r for r in all_results if r.get('user') == selected_email]
            if user_history:
                h_df = pd.DataFrame(user_history)
                st.write("**ประวัติการวินิจฉัยล่าสุด**")
                st.table(h_df[["datetime", "result", "glucose", "bmi", "age"]].head(10))
            else:
                st.write("ยังไม่พบประวัติการวินิจฉัยของคนไข้รายนี้")
    # ----------------------------
    # st.subheader("👤 ดูผลเฉพาะรายบุคคล")

    # users = sorted(df["user"].dropna().unique())

    # if not users:
    #     st.info("ยังไม่มีข้อมูลผู้ใช้")
    #     return

    # selected_user = st.selectbox("เลือกผู้ใช้", users)

    # user_df = df[df["user"] == selected_user]

    # st.dataframe(
    #     user_df.sort_values("datetime", ascending=False),
    #     use_container_width=True
    # )
# ----------------------------

#18.หน้า Dashboard ภาพรวมระบบ
def dashboard_page():
    if user_profile.get("role") != "admin":
        st.error("⛔ ไม่มีสิทธิ์")
        st.stop()

    st.subheader("📊 Dashboard ภาพรวมระบบ")

    users = list(db.collection("users").stream())
    results = list(db.collection("results").stream())

    users_df = pd.DataFrame([u.to_dict() for u in users])
    results_df = pd.DataFrame([r.to_dict() for r in results])

    col1, col2, col3 = st.columns(3)
    col1.metric("👥 ผู้ใช้ทั้งหมด", len(users_df))
    col2.metric("🧪 การทำนายทั้งหมด", len(results_df))
    col3.metric(
        "⚠ ผู้ที่เสี่ยง",
        (results_df["result"] == "เสี่ยง").sum()
    )

    if not results_df.empty:
        results_df["datetime"] = pd.to_datetime(results_df["datetime"])
        st.line_chart(
            results_df.groupby(results_df["datetime"].dt.date)["glucose"].mean()
        )

#19.หน้าให้ความรู้เกี่ยวกับโรคเบาหวาน
def about_page():
    render_styled_header("📘 เกี่ยวกับโรคเบาหวาน", "เรียนรู้วิธีการดูแลตนเองและสัญญาณเตือนของโรค")
    
    # --- ส่วนที่ 1: แนะนำโรคเบาหวานและรูปภาพ ---
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        ### 🧐 โรคเบาหวานคืออะไร?
        **โรคเบาหวาน (Diabetes Mellitus)** คือโรคที่เกิดจากระดับน้ำตาลในเลือดสูงผิดปกติจากการที่ร่างกายผลิตอินซูลินได้น้อย หรือใช้อินซูลินไม่ได้ผล
        
        หากปล่อยไว้โดยไม่ควบคุม จะส่งผลเสียต่อระบบต่างๆ ในร่างกาย เช่น หลอดเลือด หัวใจ และไต
                  
        """)
    with col2:
        # ใส่รูปภาพประกอบ (สามารถเปลี่ยน URL เป็นรูปที่คุณชอบได้)
        st.image("https://static.bangkokhospital.com/uploads/2024/07/%E0%B8%A0%E0%B8%B2%E0%B8%9E%E0%B8%9B%E0%B8%A3%E0%B8%B0%E0%B8%81%E0%B8%AD%E0%B8%9A-BGH_%E0%B8%A0%E0%B8%B2%E0%B8%A7%E0%B8%B0%E0%B8%81%E0%B9%88%E0%B8%AD%E0%B8%99%E0%B9%80%E0%B8%9A%E0%B8%B2%E0%B8%AB%E0%B8%A7%E0%B8%B2%E0%B8%99_shutterstock_1011634711.jpg", caption="จาก www.static.bangkokhospital.com")
    
    st.markdown("""
    <div>
        <p style="margin: 0; font-size: 1.05em;">สาเหตุของโรคเบาหวานเกิดจากการที่ระดับน้ำตาลในกระแสเลือดสูงมากขึ้นถึงระดับหนึ่ง จนทำให้ไตดูดกลับน้ำตาลได้ไม่หมด ซึ่งปกติไตจะมีหน้าที่ดูดกลับน้ำตาลจากสารที่ถูกกรองจากหน่วยไตไปใช้ ส่งผลให้มีน้ำตาลรั่วออกมากับปัสสาวะ จึงเป็นที่มาของคำว่า<b>“โรคเบาหวาน” </b>หากเราปล่อยให้เกิดภาวะเช่นนี้ไปนาน ๆ โดยไม่ได้รับการรักษาอย่างถูกวิธี จะทำให้เกิดภาวะแทรกซ้อนที่ร้ายแรงตามมาในที่สุด</p>
    </div>
    """, unsafe_allow_html=True)
    st.write("") # เว้น 1 บรรทัด
    st.write("") # เว้น 1 บรรทัด
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        ### 🧐 อินซูลินคืออะไร?
        **อินซูลิน (Insulin)** คือฮอร์โมนที่สร้างมาจากตับอ่อนทำหน้าที่ในการเผาผลาญคาร์โบไฮเดรตและน้ำตาลเพื่อเปลี่ยนให้เป็นในรูปของพลังงาน เข้าไปสู่กล้ามเนื้อและเซลล์ต่าง ๆ ทั่วร่างกาย ซึ่งถ้าหากอินซูลินทำงานผิดปกติ น้ำตาลที่ได้ก็จะตกค้างอยู่ในกระแสเลือด
                  
        """)
    with col2:
        # ใส่รูปภาพประกอบ (สามารถเปลี่ยน URL เป็นรูปที่คุณชอบได้)
        st.image("https://www.nakornthon.com/Upload/Images/Content/638790179276148521/Image_Cover_Insulin.jpg", caption="จาก www.nakornthon.com")
    
    st.markdown("---")
    # --- ส่วนที่ 1: ประเภทของโรคเบาหวาน (อัปเดต 4 ชนิด) ---
    st.markdown("### โรคเบาหวานแบ่งตามสาเหตุการเกิดโรค (4 ชนิด)")
    
    # ชนิดที่ 1
    st.markdown("""
    <div>
        <h4 style="color: #0d47a1; margin-bottom: 5px;">🟦 ชนิดที่ 1 (Type 1 Diabetes)</h4>
        <p style="margin: 0; font-size: 1.05em;">เกิดจากเซลล์ตับอ่อนถูกทำลายจากภูมิคุ้มกันของร่างกาย ทำให้ร่างกายขาดอินซูลินโดยสิ้นเชิง <b>มักพบในเด็กหรือผู้ที่มีอายุน้อย</b></p>
    </div>
    """, unsafe_allow_html=True)

    # ชนิดที่ 2
    st.markdown("""
    <div>
        <h4 style="color: #1b5e20; margin-bottom: 5px;">🟩 ชนิดที่ 2 (Type 2 Diabetes)</h4>
        <p style="margin: 0; font-size: 1.05em;"><b>เป็นชนิดที่พบบ่อยที่สุด (ร้อยละ 95 ของผู้ป่วยทั้งหมด)</b> เกิดจากภาวะดื้อต่ออินซูลิน มักพบในผู้ใหญ่ที่มีน้ำหนักเกินหรืออ้วนร่วมด้วย</p>
    </div>
    """, unsafe_allow_html=True)

    # ชนิดที่ 3
    st.markdown("""
    <div>
        <h4 style="color: #e65100; margin-bottom: 5px;">🟧 เบาหวานขณะตั้งครรภ์ (Gestational Diabetes)</h4>
        <p style="margin: 0; font-size: 1.05em;">เป็นโรคเบาหวานที่เกิดขึ้นขณะตั้งครรภ์ มักตรวจพบในช่วงไตรมาสที่ 2 หรือ 3 ของการตั้งครรภ์</p>
    </div>
    """, unsafe_allow_html=True)

    # ชนิดที่ 4
    st.markdown("""
    <div>
        <h4 style="color: #4a148c; margin-bottom: 5px;">🟪 โรคเบาหวานที่มีสาเหตุจำเพาะ (Specific Types)</h4>
        <p style="margin: 0; font-size: 1.05em;">เกิดจากสาเหตุอื่นๆ เช่น โรคทางพันธุกรรม, โรคของตับอ่อน, โรคทางต่อมไร้ท่อ หรือเกิดจากการใช้ยาบางชนิดเป็นเวลานาน</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    # --- ส่วนที่ 2: รูปภาพ Infographic จากสมาคมฯ (Gallery) ---
    st.markdown("### ความรู้จากสมาคมโรคเบาหวานฯ")
    img_col1, img_col2, img_col3 = st.columns(3)
    with img_col1:
        st.image("https://www.dmthai.org/new/images/knowledge/knowledge_2561/9301.jpg", caption="จาก www.dmthai.org", use_container_width=True)
    with img_col2:
        st.image("https://www.dmthai.org/new/images/knowledge/knowledge_2561/9302.jpg", caption="จาก www.dmthai.org", use_container_width=True)
    with img_col3:
        st.image("https://www.dmthai.org/new/images/knowledge/knowledge_2561/9303.jpg", caption="จาก www.dmthai.org", use_container_width=True)

    # --- ส่วนที่ 3: เกณฑ์การวินิจฉัย (ละเอียด) ---
    st.markdown("---")
    st.subheader("วิธีการวินิจฉัย (Diagnosis)")
    st.write("การวินิจฉัยทำได้โดยวิธีใดวิธีหนึ่งใน 4 วิธี ดังนี้:")
    
    diag_c1, diag_c2 = st.columns(2)
    with diag_c1:
        st.markdown("""
        * **มีอาการชัดเจน + น้ำตาลสุ่ม ≥ 200 มก./ดล.** (หิวน้ำบ่อย, ปัสสาวะมาก, น้ำหนักลดโดยไม่มีสาเหตุ)
        * **น้ำตาลหลังอดอาหาร (FBS) ≥ 126 มก./ดล.** (ต้องอดอาหารอย่างน้อย 8 ชั่วโมง)
        """)
    with diag_c2:
        st.markdown("""
        * **ทดสอบการทนกลูโคส (OGTT) ≥ 200 มก./ดล.** (ตรวจที่ 2 ชั่วโมง หลังดื่มน้ำตาล 75 กรัม)
        * **น้ำตาลสะสม (HbA1c) ≥ 6.5%** (ต้องตรวจใน Lab ที่ได้มาตรฐาน)
        """)
    st.warning("⚠️ หมายเหตุ: วิธีที่ 2-4 ต้องมีการตรวจยืนยันซ้ำอีกครั้ง (Confirm) ในวันถัดไป หรือตามดุลยพินิจของแพทย์")

    # --- ส่วนที่ 4: เป้าหมายการรักษา (Personalized Goal) ---
    st.markdown("---")
    st.subheader("เป้าหมายการรักษา (Target HbA1c)")
    st.write("ระดับน้ำตาลที่เหมาะสมสำหรับแต่ละบุคคล (Individualized Target):")
    
    # สร้างตารางเพื่อความสวยงาม
    goal_data = {
        "กลุ่มผู้ป่วย": ["เบาหวานไม่นาน/ไม่มีโรคแทรกซ้อน", "เบาหวานมานาน/มีโรคแทรกซ้อนรุนแรง", "ผู้สูงอายุ (>65 ปี) สุขภาพดี", "ผู้สูงอายุเปราะบาง/มีโรคร่วม"],
        "เป้าหมาย HbA1c": ["< 6.5% - 7.0%", "7.0% - 8.0%", "< 7.0%", "สูงถึง 8.5%"]
    }
    st.table(goal_data)
    st.write("") # เว้น 1 บรรทัด
    st.write("") # เว้น 1 บรรทัด
    st.write("") # เว้น 1 บรรทัด
    # --- ส่วนที่ 2: ประเภทและปัจจัยเสี่ยง (ใช้ Columns เพื่อให้ดูไม่เป็นพืด) ---
    c1, c2 = st.columns(2)
    with c1:
        st.image("https://icarenursinghome.com/wp-content/uploads/2024/12/%E0%B9%82%E0%B8%A3%E0%B8%84%E0%B9%80%E0%B8%9A%E0%B8%B2%E0%B8%AB%E0%B8%A7%E0%B8%B2%E0%B8%99-01.webp", caption="จาก www.icarenursinghome.com")
    with c2:
        st.image("https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEiR3HbnWkqiXgb2_WDrkNDkN8_aslJpFrAS_gHvbDvmHyU45egX_QfANzVBT9ZHL9-t-aXwHI6AWwkEVpkXeSxy6_EnookQ8VpOngg_aTpIhdVaAYedrU4WRjbRPovibO7d5o95mowMRDY/s1600/14_%25E0%25B9%2580%25E0%25B8%259A%25E0%25B8%25B2%25E0%25B8%25AB%25E0%25B8%25A7%25E0%25B8%25B2%25E0%25B8%2599-3.jpg", caption="จาก www.blogger.googleusercontent.com")

    st.markdown("---")

    # --- ส่วนที่ 3: สัญญาณเตือนและอาหาร (ใช้ Expander เหมือนเดิมแต่เพิ่ม Emoji) ---
    st.subheader("💡 ข้อมูลที่ควรรู้เพื่อการดูแลตัวเอง")
    
    tab1, tab2 = st.tabs(["⚠️ สัญญาณเตือน", "🥗 การรับประทานอาหาร"])
    
    with tab1:
        col_t1, col_t2 = st.columns([1, 2])
        with col_t1:
            st.image("https://cdn-icons-png.flaticon.com/512/2864/2864357.png", width=150)
        with col_t2:
            st.markdown("""
            * 🚽 **ปัสสาวะบ่อย** โดยเฉพาะตอนกลางคืน
            * 💧 **กระหายน้ำบ่อย** คอแห้งตลอดเวลา
            * 📉 **น้ำหนักลด** โดยไม่ทราบสาเหตุ
            * 😴 **อ่อนเพลีย** เหนื่อยง่ายแม้พักผ่อนพอ
            * 👓 **สายตาพร่ามัว** มองไม่ชัดเจน
            """)

    with tab2:
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.error("❌ อาหารที่ควรหลีกเลี่ยง")
            st.markdown("- น้ำอัดลม น้ำหวาน ชงต่างๆ\n- ขนมหวาน เบเกอรี่\n- ข้าวขัดขาว ขนมปังขาว\n- ของทอด ไขมันสูง")
        with col_f2:
            st.success("✅ อาหารที่ควรทาน")
            st.markdown("- ข้าวกล้อง ธัญพืชไม่ขัดสี\n- ผักใบเขียว ผลไม้หวานน้อย (แก้วมังกร, ฝรั่ง)\n- ปลา อกไก่\n- ดื่มน้ำเปล่าให้เพียงพอ")

    # --- ส่วนที่ 4: คำถามที่พบบ่อย (FAQ) ---
    with st.expander("🧠 คำถามที่พบบ่อย (FAQ)"):
        st.info("""
        **Q: โรคเบาหวานรักษาหายไหม?** **A:** ปัจจุบันยังไม่สามารถรักษาให้หายขาดได้ แต่สามารถ "สงบโรค" (Remission) ได้ด้วยการคุมอาหารและออกกำลังกาย จนไม่ต้องใช้ยา
        
        **Q: ถ้ามีอาการแผลหายช้าหมายความว่าอย่างไร?** **A:** เป็นหนึ่งในสัญญาณของน้ำตาลในเลือดสูง ทำให้เลือดไหลเวียนไม่ดีและเม็ดเลือดขาวทำงานได้น้อยลง
        """)

    # --- ส่วนที่ 5: สื่อมัลติมีเดียและแหล่งอ้างอิง ---
    st.markdown("### 🎥 วิดีโอความรู้เพิ่มเติม")
    st.video("https://youtu.be/Y0rkx5M-hsg")

    st.markdown("""

    ---

    ### 🔗 แหล่งข้อมูลเพิ่มเติม

    - [กรมควบคุมโรค - เบาหวาน](https://ddc.moph.go.th/)

    - [สมาคมโรคเบาหวานไทย](https://www.dmthai.org/)

    - [World Health Organization (WHO)](https://www.who.int/news-room/fact-sheets/detail/diabetes)

    """)

# 20.เริ่มต้นแอป
# 🔐 เช็ก login
if not st.session_state['logged_in']:
    auth_page()
    st.stop()

# ✅ ดึงข้อมูลผู้ใช้
user_profile = get_current_user_profile()

with st.sidebar:
    # ดึงชื่อและนามสกุลมาต่อกัน (ถ้าไม่มีให้ขึ้นว่า "ยังไม่ระบุชื่อ")
    full_name = f"{user_profile.get('name', 'ยังไม่ระบุชื่อ')} {user_profile.get('lastname', '')}".strip()
    
    # แสดงข้อมูลโปรไฟล์แบบ Card สวยงาม
    st.markdown(f"""
    <div class="profile-card">
        <small style="color: #666;">ผู้ใช้งานปัจจุบัน:</small><br>
        <strong style="font-size: 1.1rem; color: #1e3c72;">{full_name}</strong><br>
        <span style="font-size: 0.85rem; color: #555;">{st.session_state.get("user","")}</span><br>
        <div style="margin-top: 8px;">
            <span style="background: #eef2f7; padding: 2px 8px; border-radius: 5px; font-size: 0.75rem; color: #2a5298; border: 1px solid #d0dbe9;">
                สิทธิ์: {user_profile.get("role","user")}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ปุ่มออกจากระบบแบบเต็มความกว้าง
    logout_button() 
    
    #st.markdown("---") # เส้นคั่นเพื่อความสวยงาม

    # 3. แสดงสถานะเพิ่มเติม
    if user_profile and not user_profile.get("name"):
        st.warning("⚠ กรุณากรอกข้อมูลส่วนตัวให้ครบ")
        
    if user_profile and user_profile.get("role") == "admin":
        st.success("🛠️ ผู้ดูแลระบบ")

# --- หลังจากนี้คือส่วนของเมนู (อยู่นอก sidebar block หรือใช้ st.sidebar ก็ได้) ---
# 2. กำหนดรายการเมนูตามสิทธิ์
if user_profile.get("role") == "admin":
    # ลำดับเมนูสำหรับแอดมินตามที่คุณต้องการ
    menu = [
        "ระบบค้นหาประวัติคนไข้",
        "ระบบแอดมิน",
        "Dashboard",
        "โปรไฟล์ของฉัน"
    ]
else:
    # ลำดับเมนูสำหรับผู้ใช้ทั่วไป
    menu = [
        "วินิจฉัยโรคเบาหวาน",
        "ผลย้อนหลัง",
        "เกี่ยวกับโรคเบาหวาน",
        "โปรไฟล์ของฉัน"
    ]

# --- ส่วนการแสดงเนื้อหาหลัก (ย้ายมาไว้ด้านบนสุดของเนื้อหา) ---
tabs = st.tabs([f" {m}" for m in menu]) # สร้างแท็บตามรายการ menu

for i, tab in enumerate(tabs):
    with tab:
        current_menu = menu[i]
        # เรียกฟังก์ชันหน้าต่างๆ ตามชื่อเมนู
        if current_menu == "วินิจฉัยโรคเบาหวาน":
            diabetes_page()
        elif current_menu == "ผลย้อนหลัง":
            history_page()
        elif current_menu == "โปรไฟล์ของฉัน":
            profile_page()
        elif current_menu == "เกี่ยวกับโรคเบาหวาน":
            about_page()
        elif current_menu == "Dashboard":
            dashboard_page()
        elif current_menu == "ระบบแอดมิน":
            admin_page()
        elif current_menu == "ระบบค้นหาประวัติคนไข้":
            admin_results_page()

