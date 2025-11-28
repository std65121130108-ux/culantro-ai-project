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

# --- 🎨 ส่วนตกแต่ง CSS (ธีมสีสดใส เหมือนหน้า Portal) ---
st.markdown("""
<style>
    /* นำเข้าฟอนต์ Prompt */
    @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;600&display=swap');
    
    /* 1. พื้นหลัง: สีส้มแดงสดใส (ตามหน้า HTML Portal) */
    .stApp {
        background: linear-gradient(135deg, #FF416C 0%, #FF4B2B 100%);
        font-family: 'Prompt', sans-serif;
        color: #333333;
    }

    /* 2. กล่องหลัก: สไตล์กระจก (Glassmorphism) เหมือน .glass-card */
    .main .block-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 24px; /* ความโค้งเท่ากับ HTML */
        padding: 2.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2); /* เงาเท่ากับ HTML */
        max-width: 700px;
        margin-top: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }

    /* หัวข้อ */
    h1 {
        color: #333; /* สีเทาเข้ม */
        font-weight: 600;
        text-align: center;
        padding-bottom: 0.5rem;
    }
    
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* 3. ปุ่มกด: สีส้มแดง */
    div.stButton > button {
        background: linear-gradient(90deg, #FF416C 0%, #FF4B2B 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.6rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 65, 108, 0.4);
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 65, 108, 0.6);
        background: linear-gradient(90deg, #ff5b7a 0%, #ff6b4b 100%);
        color: white;
    }
    
    /* 4. File Uploader: เส้นขอบสีแดง */
    .stFileUploader {
        border: 2px dashed #FF4B2B;
        border-radius: 15px;
        padding: 15px;
        background: rgba(255, 255, 255, 0.5);
        transition: border-color 0.3s;
    }
    .stFileUploader:hover {
        border-color: #c0392b;
        background-color: rgba(255, 255, 255, 0.8);
    }
    
    /* Custom Header Style */
    .custom-header {
        text-align: center;
        margin-bottom: 35px;
    }
    .app-icon {
        width: 100px;
        height: 100px;
        background: linear-gradient(45deg, #ff9a9e 0%, #fad0c4 99%, #fad0c4 100%);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 50px;
        margin: 0 auto 20px;
        box-shadow: 0 4px 15px rgba(255, 75, 43, 0.3);
        animation: pulse 2s infinite;
    }
    .subtitle {
        color: #d32f2f;
        font-weight: 500;
        font-size: 0.9rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    
    .stAlert {
        border-radius: 12px;
        border: none;
        box-shadow: 0 2px 10px rgba(0,0,0,0.03);
    }
    
    @keyframes pulse {
        0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(255, 75, 43, 0.4); }
        70% { transform: scale(1.05); box-shadow: 0 0 0 10px rgba(255, 75, 43, 0); }
        100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(255, 75, 43, 0); }
    }
</style>
""", unsafe_allow_html=True)

# --- 2. ฟังก์ชันโหลดโมเดล ---
@st.cache_resource
def load_model():
    filename = 'efficientnetb4_model.h5'
    
    if not os.path.exists(filename):
        file_id = '1tURhAR8mXLAgnuU3EULswpcFGxnalWAV'
        url = f'https://drive.google.com/uc?id={file_id}'
        
        # ใช้ container เปล่าเพื่อแสดงข้อความโหลดแบบสวยๆ
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

# สร้างส่วนหัวแบบ Custom HTML เพื่อให้เหมือนหน้า Portal
st.markdown("""
    <div class="custom-header">
        <div class="app-icon">🌶️</div>
        <div class="subtitle">AI Expert System</div>
        <h1 style="margin-top: 0; color: #333;">Chili Doctor AI</h1>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
<p style="text-align: center; color: #555; margin-bottom: 30px; line-height: 1.6;">
    ระบบผู้เชี่ยวชาญปัญญาประดิษฐ์เพื่อวินิจฉัยโรคของพริกจากใบ <br>
    <span style="font-size: 0.9rem; color: #888;">(กรุณาอัปโหลดรูปภาพที่เห็นใบพริกชัดเจน)</span>
</p>
""", unsafe_allow_html=True)

# โหลดโมเดล
model = load_model()

if model is None:
    st.stop()

class_names = ['healthy', 'leaf curl', 'leaf spot', 'whitefly', 'yellow']

# ส่วนอัปโหลด
file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if file is None:
    st.info("👆 กรุณาเลือกรูปภาพ (.jpg, .png) จากเครื่องของคุณ")
else:
    image = Image.open(file)
    # แสดงรูปภาพแบบจัดกึ่งกลางและมีมุมมน
    st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
    st.image(image, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # เว้นวรรคนิดหน่อย
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🔍 วิเคราะห์โรค"):
        with st.spinner('AI กำลังวิเคราะห์ข้อมูล...'):
            predictions = import_and_predict(image, model)
            class_index = np.argmax(predictions)
            result_class = class_names[class_index]
            confidence = np.max(predictions) * 100

        # แสดงผลลัพธ์แบบการ์ด Alert สไตล์มินิมอล
        st.markdown("<hr style='border-top: 1px solid #eee; margin: 30px 0;'>", unsafe_allow_html=True)
        st.markdown(f"""
            <div style="background-color: #f0fff4; border: 1px solid #c3e6cb; padding: 20px; border-radius: 15px; text-align: center; margin-bottom: 15px;">
                <h4 style="margin:0; color: #155724; font-weight: 600;">ผลการวิเคราะห์: <span style="font-size: 1.4rem;">{result_class}</span></h4>
            </div>
            <p style="text-align: center; color: #6c757d; font-size: 0.9rem;">ความมั่นใจ (Confidence): <b>{confidence:.2f}%</b></p>
        """, unsafe_allow_html=True)

        # คำแนะนำ
        treatment_text = ""
        treatment_bg = "#fff8e1" # สีเหลืองอ่อนมากๆ
        treatment_border = "#ffeeba"
        text_color = "#856404"

        if result_class == 'healthy':
            treatment_text = "✅ **ต้นพริกแข็งแรงดี!** ไม่พบร่องรอยโรค หมั่นดูแลรดน้ำตามปกติ"
            treatment_bg = "#d4edda" # เขียวอ่อน
            treatment_border = "#c3e6cb"
            text_color = "#155724"
        elif result_class == 'leaf curl':
            treatment_text = "⚠️ **คำแนะนำ:** โรคใบหงิกมักเกิดจากแมลงหวี่ขาว ให้กำจัดวัชพืชและใช้สารสกัดสะเดา หรือเชื้อราเมตาไรเซียมฉีดพ่น"
        elif result_class == 'leaf spot':
            treatment_text = "⚠️ **คำแนะนำ:** โรคใบจุดตากบ เกิดจากเชื้อรา ให้ตัดแต่งใบที่เป็นโรคเผาทำลาย และฉีดพ่นสารป้องกันเชื้อรา"
        elif result_class == 'whitefly':
             treatment_text = "⚠️ **คำแนะนำ:** พบแมลงหวี่ขาว ให้ใช้กับดักกาวเหนียวสีเหลือง หรือฉีดพ่นน้ำหมักสมุนไพร"
        elif result_class == 'yellow':
             treatment_text = "⚠️ **คำแนะนำ:** อาการใบเหลือง อาจเกิดจากการขาดสารอาหาร หรือไวรัส ควรตรวจสอบดินและใส่ปุ๋ยบำรุง"
             
        # แสดงคำแนะนำในกล่องที่ดูสะอาดตา
        st.markdown(f"""
            <div style="background-color: {treatment_bg}; color: {text_color}; padding: 18px; border-radius: 12px; border: 1px solid {treatment_border}; line-height: 1.6;">
                {treatment_text}
            </div>
        """, unsafe_allow_html=True)

# Footer สวยๆ
st.markdown("""
<div style="text-align: center; margin-top: 60px; color: #e0e0e0; font-size: 0.8rem; border-top: 1px solid rgba(255,255,255,0.3); padding-top: 20px;">
    โครงงานวิจัยทางคอมพิวเตอร์ • มหาวิทยาลัยราชภัฏอุบลราชธานี<br>
    <span style="font-size: 0.75rem;">พัฒนาโดย: แมวสีขาวเทา และผองเพื่อน</span>
</div>
""", unsafe_allow_html=True)