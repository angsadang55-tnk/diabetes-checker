import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# โหลดข้อมูล
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, names=columns)

# สรุปข้อมูลเบื้องต้น
print(df.info())
print(df.describe())
print(df['Outcome'].value_counts())

# แสดงความสัมพันธ์แบบ Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Boxplot แสดงการกระจายของ Glucose ตามกลุ่ม Outcome
plt.figure(figsize=(8,5))
sns.boxplot(x='Outcome', y='Glucose', data=df)
plt.title('Glucose levels by Diabetes Outcome')
plt.show()

# Histogram แสดงการแจกแจงของอายุในกลุ่มผู้ป่วยเบาหวาน
plt.figure(figsize=(8,5))
sns.histplot(data=df[df['Outcome'] == 1], x='Age', bins=20, kde=True)
plt.title('Age Distribution in Diabetic Patients')
plt.show()
