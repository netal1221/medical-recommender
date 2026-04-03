import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

df = pd.read_csv('C:\Users\Admin\Desktop\medical-recommender\model\train_model.py')
df.fillna(0, inplace=True)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

os.makedirs('../model', exist_ok=True)
joblib.dump(model, '../model/model.pkl')
joblib.dump(le, '../model/label_encoder.pkl')
joblib.dump(list(X.columns), '../model/symptoms_list.pkl')
print("Model saved successfully!")