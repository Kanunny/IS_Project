import os
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# กำหนดค่าพื้นฐาน
MODEL_PATH = "models/traffic_sign_model.h5"
UPLOAD_FOLDER = "uploaded_images"
CLASS_NAMES = ['ป้ายบังคับ', 'ป้ายเตือน', 'ป้ายแนะนำ']

# สร้างโมเดล CNN
def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(CLASS_NAMES), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# ฝึกโมเดล
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

# โหลดโมเดลที่มีอยู่
def load_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    else:
        st.warning("ไม่มีโมเดลที่ฝึกแล้ว กรุณาฝึกโมเดลใหม่!")
        return None

# ทำนายประเภทของป้าย
def predict_image(model, image):
    img = image.convert('RGB').resize((64, 64))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction[0])
    return CLASS_NAMES[predicted_class_index], prediction[0][predicted_class_index] * 100

# เรียกใช้ Streamlit
st.set_page_config(page_title="Traffic Sign Classification App", layout="centered")
st.title("🚦 Traffic Sign Classification App")
st.write("อัปโหลดรูปภาพป้ายจราจรเพื่อทำการทำนายประเภทของป้าย")

uploaded_file = st.file_uploader("อัปโหลดรูปภาพ (สำหรับทำนาย)", type=['png', 'jpg', 'jpeg'])
model = load_model()

if model is not None and uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='รูปภาพที่อัปโหลด', use_container_width=True)
    predicted_class, confidence = predict_image(model, image)
    st.write(f"**ประเภทของป้าย:** {predicted_class} - ความมั่นใจ: {confidence:.2f}%")
