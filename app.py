import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os

# --- 1. ตั้งค่าหน้าเว็บ ---
st.set_page_config(
    page_title="Chili Doctor AI",
    page_icon="🌶️",
    layout="centered"
)

# --- 2. ฟังก์ชันโหลดโมเดล ---
@st.cache_resource
def load_model():
    filename = 'efficientnetb4_model.h5'
    
    if not os.path.exists(filename):
        file_id = '1tURhAR8mXLAgnuU3EULswpcFGxnalWAV'
        url = f'https://drive.google.com/uc?id={file_id}'
        
        with st.status("⏳ กำลังดาวน์โหลดโมเดลจาก Cloud... (ครั้งแรกเท่านั้น)", expanded=True) as status:
            try:
                import gdown
                gdown.download(url, filename, quiet=False)
                if os.path.exists(filename):
                    status.update(label="✅ ดาวน์โหลดสำเร็จ!", state="complete", expanded=False)
                else:
                    status.update(label="❌ ดาวน์โหลดไม่สำเร็จ", state="error")
                    return None
            except Exception as e:
                status.update(label=f"❌ Error: {e}", state="error")
                return None

    try:
        model = tf.keras.models.load_model(filename)
        return model
    except Exception as e:
        st.error(f"❌ ไฟล์โมเดลมีปัญหา: {e}")
        return None

# --- 3. ฟังก์ชันเตรียมรูป ---
def import_and_predict(image_data, model):
    size = (300, 300)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    img_array = img_array.astype(np.float32) 
    
    data = np.ndarray(shape=(1, 300, 300, 3), dtype=np.float32)
    data[0] = img_array
    
    prediction = model.predict(data)
    return prediction

# --- 4. ส่วนแสดงผล (UI) ---

# โหลดโมเดล
model = load_model()

if model is None:
    st.stop()

class_names = ['healthy', 'leaf curl', 'leaf spot', 'whitefly', 'yellow']

# --- ⭐ สร้างกรอบขาว (Card) สำหรับส่วน Input ⭐ ---
with st.container(border=True):
    # ส่วนหัว (Icon + Title)
    st.markdown("""
        <div class="custom-header">
            <div class="app-icon">🌶️</div>
            <div class="subtitle">AI Expert System</div>
            <h1 style="margin-top: 0;">Chili Doctor AI</h1>
        </div>
    """, unsafe_allow_html=True)

    # คำอธิบาย
    st.markdown("""
    <p style="text-align: center; margin-bottom: 30px;">
        ระบบผู้เชี่ยวชาญปัญญาประดิษฐ์เพื่อวินิจฉัยโรคของพริกจากใบ <br>
        <span style="font-size: 0.95rem; color: #FF4B2B; font-weight: 500;">(กรุณาอัปโหลดรูปภาพที่เห็นใบพริกชัดเจน)</span>
    </p>
    """, unsafe_allow_html=True)

    # ช่องอัปโหลดไฟล์
    file = st.file_uploader("", type=["jpg", "png", "jpeg"])
    
    # ข้อความแนะนำ
    if file is None:
        st.markdown("""
            <div style="text-align: center; color: #999; margin-top: 15px; font-size: 0.9rem;">
                <i class="fas fa-arrow-up"></i> คลิก หรือ ลากไฟล์มาวางในกรอบเส้นประด้านบน
            </div>
        """, unsafe_allow_html=True)

# --- ส่วนแสดงผลลัพธ์ (อยู่นอกกรอบ Card หลัก) ---
if file is not None:
    image = Image.open(file)
    
    # สร้างกรอบใหม่สำหรับผลลัพธ์โดยเฉพาะ
    with st.container(border=True):
        st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div><br>', unsafe_allow_html=True)
        
        if st.button("🔍 วิเคราะห์โรค"):
            with st.spinner('🤖 AI กำลังวิเคราะห์ข้อมูล...'):
                predictions = import_and_predict(image, model)
                class_index = np.argmax(predictions)
                result_class = class_names[class_index]
                confidence = np.max(predictions) * 100

            # เส้นคั่น
            st.markdown("<hr style='border-top: 2px solid #f0f0f0; margin: 30px 0;'>", unsafe_allow_html=True)
            
            # การ์ดแสดงชื่อโรค
            st.markdown(f"""
                <div style="background-color: #f8fff9; border: 2px solid #28a745; padding: 25px; border-radius: 20px; text-align: center; margin-bottom: 20px;">
                    <div style="color: #28a745; font-size: 1rem; font-weight: 600; margin-bottom: 5px;">ผลการวิเคราะห์</div>
                    <h2 style="margin:0; color: #1e7e34; font-size: 2rem;">{result_class}</h2>
                </div>
                <p style="text-align: center; color: #6c757d; font-size: 1rem;">ความมั่นใจ (Confidence): <b>{confidence:.2f}%</b></p>
            """, unsafe_allow_html=True)

            # คำแนะนำการรักษา
            treatment_text = ""
            treatment_bg = "#fff3cd"
            treatment_border = "#ffecb5"
            text_color = "#856404"
            icon = "⚠️"

            if result_class == 'healthy':
                treatment_text = "✅ **ต้นพริกแข็งแรงดี!** ไม่พบร่องรอยโรค หมั่นดูแลรดน้ำและใส่ปุ๋ยตามปกติ"
                treatment_bg = "#d4edda"
                treatment_border = "#c3e6cb"
                text_color = "#155724"
                icon = "🌿"
            elif result_class == 'leaf curl':
                treatment_text = "⚠️ **คำแนะนำ:** โรคใบหงิกมักเกิดจากแมลงหวี่ขาว ให้กำจัดวัชพืชและใช้สารสกัดสะเดา หรือเชื้อราเมตาไรเซียมฉีดพ่น"
            elif result_class == 'leaf spot':
                treatment_text = "⚠️ **คำแนะนำ:** โรคใบจุดตากบ เกิดจากเชื้อรา ให้ตัดแต่งใบที่เป็นโรคเผาทำลาย และฉีดพ่นสารป้องกันเชื้อรา"
            elif result_class == 'whitefly':
                 treatment_text = "⚠️ **คำแนะนำ:** พบแมลงหวี่ขาว ให้ใช้กับดักกาวเหนียวสีเหลือง หรือฉีดพ่นน้ำหมักสมุนไพร"
            elif result_class == 'yellow':
                 treatment_text = "⚠️ **คำแนะนำ:** อาการใบเหลือง อาจเกิดจากการขาดสารอาหาร หรือไวรัส ควรตรวจสอบดินและใส่ปุ๋ยบำรุง"
                 
            st.markdown(f"""
                <div style="background-color: {treatment_bg}; color: {text_color}; padding: 25px; border-radius: 15px; border: 1px solid {treatment_border}; line-height: 1.8; font-size: 1.1rem;">
                    <strong>{icon} คำแนะนำจาก AI:</strong><br>
                    {treatment_text}
                </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 60px; color: #ffffff; font-size: 0.9rem; opacity: 0.8;">
    โครงงานวิจัยทางคอมพิวเตอร์ • มหาวิทยาลัยราชภัฏอุบลราชธานี<br>
    <span style="font-size: 0.8rem;">พัฒนาโดย: แมวสีขาวเทา และผองเพื่อน</span>
</div>
""", unsafe_allow_html=True)