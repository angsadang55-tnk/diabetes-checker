import firebase_admin
from firebase_admin import credentials, firestore, auth
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import plotly.express as px

# ===== Session State Init (ต้องอยู่บนสุดก่อนใช้งาน) =====
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user" not in st.session_state:
    st.session_state.user = None

# --- 1. การเชื่อมต่อ Firebase ---
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

from datetime import datetime

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

def diabetes_page():
    st.title("ระบบวินิจฉัยโรคเบาหวานด้วย Machine Learning")
    st.markdown("กรอกข้อมูลสุขภาพของคุณเพื่อประเมินความเสี่ยงเป็นโรคเบาหวาน")

    with st.form(key='diabetes_form'):
        pregnancies = st.number_input("จำนวนครั้งที่ตั้งครรภ์", min_value=0, max_value=20, step=1)
        glucose = st.number_input("ระดับน้ำตาลในเลือด (Glucose)", min_value=0)
        blood_pressure = st.number_input("ความดันโลหิต (BloodPressure)", min_value=0)
        skin_thickness = 20
        insulin = st.number_input("ระดับอินซูลิน (Insulin)", min_value=0)
        weight = st.number_input("น้ำหนัก (กิโลกรัม)", min_value=1.0, step=0.1, format="%.1f")
        height_cm = st.number_input("ส่วนสูง (เซนติเมตร)", min_value=30.0, step=0.1, format="%.1f")

        if weight > 0 and height_cm > 0:
            height_m = height_cm / 100
            bmi = weight / (height_m ** 2)
            st.write(f"ค่า BMI ของคุณคือ: **{bmi:.2f}**")
        else:
            st.info("กรุณากรอกน้ำหนักและส่วนสูงเพื่อคำนวณ BMI")

        family_count = st.number_input(
            "จำนวนสมาชิกในครอบครัวที่เป็นโรคเบาหวาน",
            min_value=0,
            max_value=10,
            step=1,
            help="นับเฉพาะพ่อ แม่ พี่ น้อง ปู่ ย่า ตา ยาย"
        )

        if family_count == 0:
            pedigree_percent = 10
            diabetes_pedigree = 0.1
        elif family_count == 1:
            pedigree_percent = 35
            diabetes_pedigree = 0.4
        else:
            pedigree_percent = 65
            diabetes_pedigree = 0.8

        st.info(f"📌 ความเสี่ยงจากพันธุกรรมประมาณ **{pedigree_percent}%**")

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
        
def profile_page():
    st.subheader("👤 โปรไฟล์ผู้ใช้งาน")

    email = st.session_state.get("user")
    user_ref = db.collection("users").document(email)

    doc = user_ref.get()
    if doc.exists:
        user_data = doc.to_dict()
    else:
        user_data = {
            "email": email,
            "name": "",
            "age": 25,
            "gender": "หญิง",
            "role": "user"
        }
        user_ref.set(user_data)

    name = st.text_input("ชื่อ-นามสกุล", user_data.get("name", ""))
    age = st.number_input("อายุ", 1, 120, user_data.get("age", 25))
    gender = st.selectbox(
        "เพศ",
        ["หญิง", "ชาย", "อื่นๆ"],
        index=["หญิง", "ชาย", "อื่นๆ"].index(user_data.get("gender", "หญิง"))
    )

    if st.button("💾 บันทึกข้อมูล"):
        user_ref.update({
            "name": name,
            "age": age,
            "gender": gender
        })
        st.success("✅ บันทึกข้อมูลเรียบร้อย")
        
# แก้ไขฟังก์ชัน delete_user
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
def admin_results_page():
    if user_profile.get("role") != "admin":
        st.error("⛔ ไม่มีสิทธิ์")
        st.stop()

    st.subheader("📊 ผลทำนายของผู้ใช้ทั้งหมด")
    
    records = db.collection("results").stream()
    data = [r.to_dict() for r in records]

    if not data:
        st.info("ยังไม่มีข้อมูล")
        return

    df = pd.DataFrame(data)

    # 🔹 โหลด users
    users_docs = db.collection("users").stream()
    users_map = {
        u.to_dict().get("email"): u.to_dict().get("name", "")
        for u in users_docs
    }

    df["name"] = df["user"].map(users_map)
    # 🔹 ย้าย name ไปเป็นคอลัมน์แรก
    cols_order = [
        "name", "user", "result", "datetime",
        "bmi", "glucose", "insulin", "blood_pressure",
        "weight", "height_cm", "age"
    ]
    df = df[[c for c in cols_order if c in df.columns]]

    df["datetime"] = pd.to_datetime(df["datetime"])

    # ✅ กันข้อมูลเก่าที่ไม่มี field
    if "name" not in df.columns:
        df["name"] = ""

    if "user" not in df.columns:
        df["user"] = ""

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])

    # ----------------------------
    st.subheader("🔍 ค้นหาข้อมูลผู้ใช้")
    keyword = st.text_input("ค้นหาชื่อหรืออีเมล")

    if keyword:
        df = df[
            df["name"].str.contains(keyword, case=False, na=False) |
            df["user"].str.contains(keyword, case=False, na=False)
        ]

    #st.dataframe(
    #    df.sort_values("datetime", ascending=False),
     #   use_container_width=True
    #)

    # 🔒 บังคับโครงสร้างคอลัมน์ให้ตรง
    columns_order = [
        "name",
        "user",
        "result",
        "datetime",
        "bmi",
        "glucose",
        "insulin",
        "blood_pressure",
        "weight",
        "height_cm",
        "age"
    ]

    # เติมคอลัมน์ที่ขาด (กัน KeyError)
    for col in columns_order:
        if col not in df.columns:
            df[col] = ""

    df = df[columns_order]

    st.dataframe(df, use_container_width=True)

    st.download_button(
        "📥 ดาวน์โหลดผลทำนายทั้งหมด (CSV)",
        df.to_csv(index=False).encode("utf-8-sig"),
        file_name="all_results_clean.csv",
        mime="text/csv"
    )

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
# 🔐 เช็ก login
if not st.session_state['logged_in']:
    auth_page()
    st.stop()

# ✅ ดึงข้อมูลผู้ใช้
user_profile = get_current_user_profile()

with st.sidebar:
    # 1. แสดงข้อมูลโปรไฟล์ (โชว์ทุกคน)
    st.markdown(f"""
    <div style="
        background:#f0f6ff;
        padding:12px;
        border-radius:10px;
        margin-bottom:10px;
    ">
    👤 <span style="color:#000;"><b>{user_profile.get("name","ยังไม่ระบุชื่อ")}</b></span><br>
    <span style="color:#555;"><small>{st.session_state.get("user","")}</small></span><br>
    <span style="color:#555;">สิทธิ์: {user_profile.get("role","user")}</span>
    </div>
    """, unsafe_allow_html=True)

    # 2. วางปุ่มออกจากระบบตรงนี้ (ย้ายออกมาข้างนอกเพื่อให้โชว์ทุกคน)
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
        "ผลทำนายทั้งหมด",
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

page = st.sidebar.selectbox("เมนูหลัก", menu)
if page == "วินิจฉัยโรคเบาหวาน":
    diabetes_page()

elif page == "ผลย้อนหลัง":
    history_page()

elif page == "โปรไฟล์ของฉัน":
    profile_page()

elif page == "เกี่ยวกับโรคเบาหวาน":
    about_page()

elif page == "Dashboard":
    dashboard_page()

elif page == "ระบบแอดมิน":
    admin_page()

elif page == "ผลทำนายทั้งหมด":
    admin_results_page()



