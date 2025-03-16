import streamlit as st


st.set_page_config(page_title="อธิบายการทำงานของแอป", layout="centered")
st.header("ที่มาของ Dataset")
st.write("""
Dataset จาก [Kaggle](https://www.kaggle.com/datasets/ahemateja19bec1025/traffic-sign-dataset-classification?resource=download&SSORegistrationToken=CfDJ8OdV9jvuBrlOizyz5DdMpkzCIO6CiGdv8qxKiAukQet9xYyVEbI3vDX0_yHZn4Rj5TXAt2WQ6P7Netv3S6_3wGQbSjvndE1f77XtGY8kP-ncP9qjP8yTd_9_I1haWXBiTR8xaTX56Ipg8Awa2nTqY0BbobBtFiqowxQN5aunoeRsrfNtSmGWzaK7k-6jwPwFuyOGcYW2mSWcQLJNGDrgxtsHWGpha1sIA3Wekd9WJ_bdxSERXqYMNEYwNWwnmKsRvKDZL7cmL8zeAO7Gi6ALCuWvbV8GXC-nVXy6Y_glhzNOJKdEa60kABg1fqu3y37tH8-COBLyqVjfkNUSX6PMZED3LDBe7KAF8YJp_UIbxD0&DisplayName=NONTHACHAI%20KRATUDNGOEN) ประกอบด้วยภาพ "ป้ายจราจร" ประมาณ 3000 ภาพสำหรับทดสอบ""")
st.header("ทฤษฎีเบื้องหลัง CNN")
st.write("""
Convolutional Neural Network (CNN) เป็นโครงข่ายประสาทเทียมที่ออกแบบมาสำหรับการประมวลผลภาพโดยเฉพาะ ซึ่ง CNN ใช้หลักการของ Convolution (การพับภาพ) และ Pooling (การลดขนาดภาพ) เพื่อดึงคุณลักษณะเด่นของภาพโดยอัตโนมัติ""")


st.title("อธิบายการฝึกและการทำนายของ Traffic Sign Classification App")

st.header("1️ การฝึกโมเดล (Training)")
st.code("""
def train_model():
    model = create_model()
    datagen = ImageDataGenerator(
        rescale=1.0/255, validation_split=0.2,
        rotation_range=20, width_shift_range=0.2,
        height_shift_range=0.2, shear_range=0.2,
        zoom_range=0.2, horizontal_flip=True, fill_mode='nearest'
    )
    train_gen = datagen.flow_from_directory(UPLOAD_FOLDER, target_size=(64, 64), batch_size=32, class_mode='sparse', subset='training')
    val_gen = datagen.flow_from_directory(UPLOAD_FOLDER, target_size=(64, 64), batch_size=32, class_mode='sparse', subset='validation')

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
    ]

    model.fit(train_gen, validation_data=val_gen, epochs=20, callbacks=callbacks)
    model.save(MODEL_PATH)
    print("โมเดลได้ถูกฝึกและบันทึกแล้ว!")
""", language="python")
st.write("""
- **สร้างโมเดล** ด้วยฟังก์ชัน `create_model()`
- **ใช้ ImageDataGenerator** เพื่อเพิ่มความหลากหลายของข้อมูลและแบ่งเป็นชุดฝึก (training) และชุดทดสอบ (validation)
- **ใช้ callback** เช่น EarlyStopping, ModelCheckpoint และ ReduceLROnPlateau เพื่อป้องกัน overfitting
- **โมเดลถูกบันทึก** ไว้ที่ `MODEL_PATH`
""")

st.header("2️ การทำนาย (Prediction)")
st.code("""
def predict_image(model, image):
    img = image.convert('RGB').resize((64, 64))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction[0])
    return CLASS_NAMES[predicted_class_index], prediction[0][predicted_class_index] * 100
""", language="python")
st.write("""
- **โหลดและแปลงภาพ** ให้เป็นขนาดที่โมเดลรองรับ (64x64 pixels)
- **ทำการคำนวณผลลัพธ์** โดยใช้ `model.predict()`
- **เลือกคลาสที่มีค่าความมั่นใจสูงสุด** (`np.argmax()`)
- **คืนค่าผลลัพธ์** เป็นชื่อป้ายจราจรและค่าความมั่นใจ (%)
""")

st.header(" สรุปการทำงาน")
st.write("""
1️ **ฝึกโมเดลด้วยชุดข้อมูลที่มี** และใช้เทคนิคลด overfitting
2️ **โหลดภาพที่ผู้ใช้อัปโหลด** และแปลงให้อยู่ในรูปแบบที่โมเดลสามารถอ่านได้
3️ **ใช้โมเดลในการทำนายผล** และแสดงประเภทของป้ายจราจร
""")
st.success("ระบบพร้อมใช้งานสำหรับการตรวจจับป้ายจราจรแล้ว!")