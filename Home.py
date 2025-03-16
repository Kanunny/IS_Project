import streamlit as st

# ตั้งค่าหน้า
st.set_page_config(page_title="Home", page_icon="🏠")

# โหลดไฟล์ CSS
with open("assets/styles.css", "r", encoding="utf-8") as css_file:
    st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)

# แสดงชื่อของคุณ
st.markdown('<h1 class="title">Welcome to the Project</h1>', unsafe_allow_html=True)
st.markdown('<h2 style="text-align: center; color: #444;">นนทชัย กระตุดเงิน 6604062610101 S.3  </h2>', unsafe_allow_html=True)

