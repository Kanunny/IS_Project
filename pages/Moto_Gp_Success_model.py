import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import streamlit as st

# โหลดข้อมูล
data = pd.read_csv('data/RidersSummary.csv')

# ตรวจสอบข้อมูล
st.write("### Dataset Preview")
st.write(data.head())

# เข้ารหัสข้อมูลที่เป็นข้อความเป็นตัวเลข
label_encoders = {}
for column in ['class', 'motorcycle', 'team', 'home_country', 'rider_name']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# แยกข้อมูลสำหรับ Train และ Test
train_data = data[data['season'] == 2024]  # ใช้ปี 2024 เทรน
test_data = data[data['season'] == 2025]   # ใช้ปี 2025 ทำนาย

# กำหนด Features และ Target
features = ['class', 'motorcycle', 'team', 'races_participated', 'season']
X_train = train_data[features]
X_test = test_data[features]
y_train_wins = train_data['wins']
y_test_wins = test_data['wins']
y_train_podium = train_data['podium']
y_test_podium = test_data['podium']

# มาตรฐานข้อมูล (Standardization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Linear Regression Model ---
linear_model_wins = LinearRegression()
linear_model_wins.fit(X_train, y_train_wins)
predictions_wins_lr = linear_model_wins.predict(X_test)

linear_model_podium = LinearRegression()
linear_model_podium.fit(X_train, y_train_podium)
predictions_podium_lr = linear_model_podium.predict(X_test)

# --- Random Forest Regressor Model ---
rf_model_wins = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model_wins.fit(X_train, y_train_wins)
predictions_wins_rf = rf_model_wins.predict(X_test)

rf_model_podium = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model_podium.fit(X_train, y_train_podium)
predictions_podium_rf = rf_model_podium.predict(X_test)

# เพิ่มชื่อของนักแข่งที่ทำนายได้
rider_names = test_data['rider_name'].values

# สร้าง DataFrame แสดงผลการพยากรณ์
prediction_df = pd.DataFrame({
    'Rider': [label_encoders['rider_name'].inverse_transform([name])[0] for name in rider_names],
    'Actual Wins': y_test_wins,
    'Predicted Wins (Linear Regression)': predictions_wins_lr,
    'Predicted Wins (Random Forest)': predictions_wins_rf,
    'Actual Podium': y_test_podium,
    'Predicted Podium (Linear Regression)': predictions_podium_lr,
    'Predicted Podium (Random Forest)': predictions_podium_rf
})

# แสดงตารางผลลัพธ์
st.write("### Prediction Results")
st.write(prediction_df)

# --- ตัวเลือกให้เลือกโมเดลที่ต้องการดู ---
model_choice = st.selectbox("เลือกโมเดลที่ต้องการดู:", ["Linear Regression", "Random Forest"])

# --- กำหนดค่าพยากรณ์ตามโมเดลที่เลือก ---
if model_choice == "Linear Regression":
    wins_col = "Predicted Wins (Linear Regression)"
    podium_col = "Predicted Podium (Linear Regression)"
else:
    wins_col = "Predicted Wins (Random Forest)"
    podium_col = "Predicted Podium (Random Forest)"

# --- Plot กราฟเปรียบเทียบผลลัพธ์ ---
st.write(f"### Comparison of Actual vs {model_choice} Predicted Results (2025)")

fig, axes = plt.subplots(1, 2, figsize=(24, 8))  # ขยายขนาดกราฟให้ใหญ่ขึ้น

# กราฟ Wins
prediction_df.set_index('Rider')[['Actual Wins', wins_col]].plot(
    kind='bar', ax=axes[0], width=0.7, cmap="coolwarm", edgecolor="black")
axes[0].set_title(f'Comparison of Actual vs {model_choice} Predicted Wins (2025)', fontsize=16)
axes[0].set_ylabel('Wins', fontsize=14)
axes[0].set_xlabel('Rider', fontsize=14)
axes[0].grid(axis='y', linestyle='--', alpha=0.7)
axes[0].tick_params(axis='x', rotation=45, labelsize=12)

# กราฟ Podium
prediction_df.set_index('Rider')[['Actual Podium', podium_col]].plot(
    kind='bar', ax=axes[1], width=0.7, cmap="viridis", edgecolor="black")
axes[1].set_title(f'Comparison of Actual vs {model_choice} Predicted Podium (2025)', fontsize=16)
axes[1].set_ylabel('Podium Finishes', fontsize=14)
axes[1].set_xlabel('Rider', fontsize=14)
axes[1].grid(axis='y', linestyle='--', alpha=0.7)
axes[1].tick_params(axis='x', rotation=45, labelsize=12)

plt.tight_layout()
st.pyplot(fig)
