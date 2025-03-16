import streamlit as st


st.set_page_config(page_title="อธิบายการทำงานของแอป", layout="centered")
st.header("ที่มาของ Dataset")
st.write("""
จาก [Kaggle](https://www.kaggle.com/datasets/mayuravartak/motogp-riders-2024-performance-dataset) การแข่งขันมอเตอร์ไซค์ที่มีข้อมูลเกี่ยวกับนักแข่ง ฤดูกาล ทีม และผลการแข่งขัน เช่น จำนวนชัยชนะ (Wins) และการขึ้นโพเดียม (Podium)
""")


st.header("ทฤษฎีเบื้องหลังของ Linear Regression และ Random Forest")
st.header("1. Linear Regression")
st.write("""
Linear Regression เป็นเทคนิคของ Supervised Learning ที่ใช้ในการทำนายค่าต่อเนื่อง 
โดยใช้สมการเชิงเส้นในการอธิบายความสัมพันธ์ระหว่างตัวแปรอิสระและตัวแปรตาม

สมการของ Linear Regression มีรูปแบบดังนี้:
\[ y = w_1 x_1 + w_2 x_2 + ... + w_n x_n + b \]

โดยที่:
- \( y \) คือค่าที่ต้องการทำนาย
- \( x_1, x_2, ..., x_n \) คือคุณลักษณะของข้อมูล
- \( w_1, w_2, ..., w_n \) คือค่าสัมประสิทธิ์ของแต่ละตัวแปร
- \( b \) คือค่า bias หรือค่าคงที่

การทำงานของ Linear Regression:
1. คำนวณค่าสัมประสิทธิ์ \( w \) โดยใช้ Least Squares Method
2. ใช้ค่าคงที่ \( b \) และค่าสัมประสิทธิ์ \( w \) ที่หาได้มาทำนายค่าผลลัพธ์
3. วัดความแม่นยำของโมเดลโดยใช้ค่าความคลาดเคลื่อน เช่น Mean Squared Error (MSE)
""")

st.header("2. Random Forest")
st.write("""
Random Forest เป็นโมเดล Ensemble Learning ที่ใช้ Decision Trees หลายต้นร่วมกัน 
เพื่อให้การพยากรณ์มีความแม่นยำมากขึ้น โดยใช้เทคนิค Bagging (Bootstrap Aggregating)

หลักการทำงานของ Random Forest:
1. สุ่มชุดข้อมูล (Bootstrap Sampling) โดยสุ่มตัวอย่างข้อมูลจาก Training Set หลายชุด
2. สร้าง Decision Trees หลายต้น โดยแต่ละต้นได้รับข้อมูลที่แตกต่างกัน
3. เลือกคุณลักษณะแบบสุ่ม สำหรับแต่ละต้นไม้เพื่อลดความสัมพันธ์ระหว่างต้นไม้
4. รวมผลลัพธ์จากทุกต้นไม้:
   - สำหรับ Regression: ใช้ค่าเฉลี่ยของผลลัพธ์จากต้นไม้ทั้งหมด
   - สำหรับ Classification: ใช้เสียงข้างมากของผลลัพธ์จากแต่ละต้นไม้""")


st.header("การเตรียมข้อมูลและการพยากรณ์ผลการแข่งขัน")
st.write("""
แอปนี้ใช้ **Machine Learning** เพื่อพยากรณ์ผลการแข่งขัน โดยอาศัยข้อมูลจากฤดูกาล 2024 ในการเทรนโมเดล และใช้ข้อมูลฤดูกาล 2025 ทำนายผลลัพธ์
""")

st.title("อธิบายการทำงานของโค้ด")

st.header("1️ โหลดและเตรียมข้อมูล")
st.code("""
# โหลดข้อมูลจากไฟล์ CSV
data = pd.read_csv('data/RidersSummary.csv')

# ตรวจสอบข้อมูล
st.write(data.head())

# แปลงข้อมูลที่เป็นข้อความให้เป็นตัวเลข
label_encoders = {}
for column in ['class', 'motorcycle', 'team', 'home_country', 'rider_name']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le
""", language="python")
st.write("""
- โหลดข้อมูลจาก **RidersSummary.csv** และแสดงตัวอย่างข้อมูล
- แปลงข้อมูล **categorical** (เช่น ทีม นักแข่ง) เป็นตัวเลขโดยใช้ `LabelEncoder`
""")

st.header("2️ แยกข้อมูลสำหรับการ Train และ Test")
st.code("""
train_data = data[data['season'] == 2024]
test_data = data[data['season'] == 2025]

features = ['class', 'motorcycle', 'team', 'races_participated', 'season']
X_train = train_data[features]
X_test = test_data[features]
y_train_wins = train_data['wins']
y_test_wins = test_data['wins']
y_train_podium = train_data['podium']
y_test_podium = test_data['podium']
""", language="python")
st.write("""
- ใช้ข้อมูลจาก **ฤดูกาล 2024** เป็นชุดฝึก และ **ฤดูกาล 2025** เป็นชุดทดสอบ
- เลือก **features** ที่สำคัญสำหรับการทำนาย เช่น ทีม รถแข่ง และจำนวนการแข่งขัน
""")

st.header("3️ การฝึกโมเดล (Training Models)")
st.code("""
# Standardize ข้อมูล
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Linear Regression
linear_model_wins = LinearRegression()
linear_model_wins.fit(X_train, y_train_wins)
predictions_wins_lr = linear_model_wins.predict(X_test)

# Random Forest
rf_model_wins = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model_wins.fit(X_train, y_train_wins)
predictions_wins_rf = rf_model_wins.predict(X_test)
""", language="python")
st.write("""
- **Standardize ข้อมูล** โดยใช้ `StandardScaler()` เพื่อให้โมเดลเรียนรู้ได้ดีขึ้น
- เทรนโมเดล **Linear Regression** และ **Random Forest** เพื่อนำไปใช้ทำนายผล
""")

st.header("4️ ทำนายผลและแสดงผลลัพธ์")
st.code("""
# แสดงผลลัพธ์ในตาราง
prediction_df = pd.DataFrame({
    'Rider': [label_encoders['rider_name'].inverse_transform([name])[0] for name in test_data['rider_name'].values],
    'Actual Wins': y_test_wins,
    'Predicted Wins (Linear Regression)': predictions_wins_lr,
    'Predicted Wins (Random Forest)': predictions_wins_rf
})

st.write(prediction_df)
""", language="python")
st.write("""
- ทำนายผลการแข่งขัน และแสดง **ตารางเปรียบเทียบระหว่างค่าจริงกับค่าที่พยากรณ์ได้**
- รองรับการแสดงผลลัพธ์จากทั้ง **Linear Regression และ Random Forest**
""")

st.header("5️ การแสดงผลด้วยกราฟ")
st.code("""
fig, axes = plt.subplots(1, 2, figsize=(24, 8))
prediction_df.set_index('Rider')[['Actual Wins', 'Predicted Wins (Random Forest)']].plot(
    kind='bar', ax=axes[0], width=0.7, cmap="coolwarm", edgecolor="black")
st.pyplot(fig)
""", language="python")
st.write("""
- ใช้ **Matplotlib** และ **Streamlit** แสดงผลลัพธ์เป็น **กราฟแท่ง**
- เปรียบเทียบค่าจริงกับค่าที่โมเดลพยากรณ์ได้
""")
