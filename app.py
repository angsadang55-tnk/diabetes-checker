import firebase_admin
from firebase_admin import credentials, firestore
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import plotly.express as px
# --- 1. การเชื่อมต่อ Firebase ---
def init_firebase():
    if not firebase_admin._apps:
        try:
            cred_info = dict(st.secrets["firebase"])
            cred = credentials.Certificate(cred_info)
            firebase_admin.initialize_app(cred)
            st.success("🔥 Firebase connected successfully")
        except Exception as e:
            st.error(f"❌ Firebase init failed: {e}")
            st.stop()
    return firestore.client()

db = init_firebase()

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

# --- 3. โหลดโมเดล ---
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

def save_result(result, user_input):
    db.collection("results").add({
        "user": st.session_state["user"],
        "datetime": datetime.now(),
        "result": result,
        **user_input
    })
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

            try:
                from firebase_admin import auth
                auth.create_user(email=email, password=password)
                st.success("สมัครสมาชิกสำเร็จ! กรุณาเข้าสู่ระบบ")
                st.session_state.auth_mode = "login"
                st.rerun()
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาด: {e}")

        if st.button("มีบัญชีแล้ว? กลับเข้าสู่ระบบ"):
            st.session_state.auth_mode = "login"
            st.rerun()

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
            glucose, blood_pressure, skin_thickness,
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

        # 4. ตารางข้อมูลดิบ
        st.subheader("📄 ตารางข้อมูลดิบ")

        display_df = filtered_df[
            ["datetime", "result", "glucose", "blood_pressure", "bmi", "age"]
        ].rename(columns={
            "datetime": "วันที่/เวลา",
            "result": "ผลทำนาย",
            "glucose": "น้ำตาลในเลือด",
            "blood_pressure": "ความดันโลหิต",
            "bmi": "BMI",
            "age": "อายุ"
        })

        st.dataframe(
            display_df.sort_values(by="วันที่/เวลา", ascending=False),
            use_container_width=True
        )

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาด: {e}")

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

if not st.session_state['logged_in']:
    auth_page()
    st.stop()

# แสดงปุ่มออกจากระบบทุกหน้า
logout_button()

page = st.sidebar.selectbox("เมนู", ["วินิจฉัยโรคเบาหวาน", "ผลย้อนหลัง", "เกี่ยวกับโรคเบาหวาน"])

if page == "วินิจฉัยโรคเบาหวาน":
    diabetes_page()
elif page == "ผลย้อนหลัง":
    history_page()
elif page == "เกี่ยวกับโรคเบาหวาน":
    about_page()

