import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# โหลดข้อมูล
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, names=columns)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างโมเดลและพารามิเตอร์ที่ต้องการทดลอง
models = {
    "RandomForest": RandomForestClassifier(),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "SVM": SVC()
}

params = {
    "RandomForest": {
        "n_estimators": [50, 100],
        "max_depth": [5, 10, None]
    },
    "LogisticRegression": {
        "C": [0.1, 1, 10]
    },
    "SVM": {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"]
    }
}

best_models = {}

for name in models:
    print(f"Training and tuning {name}...")
    grid = GridSearchCV(models[name], params[name], cv=5, scoring="accuracy")
    grid.fit(X_train, y_train)
    print(f"Best parameters for {name}: {grid.best_params_}")
    y_pred = grid.predict(X_test)
    print(classification_report(y_test, y_pred))
    best_models[name] = grid.best_estimator_

# บันทึกโมเดลที่ดีที่สุด (RandomForest สมมติเป็นดีที่สุด)
import joblib
joblib.dump(best_models["RandomForest"], "optimized_diabetes_model.pkl")
print("✅ บันทึกโมเดล optimized_diabetes_model.pkl เรียบร้อยแล้ว")
