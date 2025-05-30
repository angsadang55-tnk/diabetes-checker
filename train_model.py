import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# โหลดข้อมูล
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, names=columns)

# แยก features กับ label
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# แบ่ง train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างโมเดล
model = RandomForestClassifier()
model.fit(X_train, y_train)

# ประเมินผล
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# บันทึกโมเดล
joblib.dump(model, "diabetes_model.pkl")
print("✅ บันทึกโมเดลเป็นไฟล์ 'diabetes_model.pkl' สำเร็จแล้ว")
