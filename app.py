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

# --- 3. โหลดโมเดล ---
@st.cache_resource
def load_model():
    filename = 'efficientnetb4_model.h5'
    if not os.path.exists(filename):
        file_id = '1tURhAR8mXLAgnuU3EULswpcFGxnalWAV'
        url = f'https://drive.google.com/uc?id={file_id}'
        with st.status("⏳ กำลังดาวน์โหลดโมเดล...", expanded=True) as status:
            try:
                import gdown
                gdown.download(url, filename, quiet=False)
                if os.path.exists(filename):
                    status.update(label="✅ เสร็จสิ้น!", state="complete", expanded=False)
                else:
                    return None
            except:
                return None
    try:
        return tf.keras.models.load_model(filename)
    except:
        return None

# ฟังก์ชันทำนาย
def import_and_predict(image_data, model):
    size = (300, 300)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image).astype(np.float32)
    data = np.ndarray(shape=(1, 300, 300, 3), dtype=np.float32)
    data[0] = img_array
    return model.predict(data)

# --- 4. ส่วนแสดงผล (UI) ---

model = load_model()

# --- สร้างกรอบขาว (Card) ---
with st.container(border=True):
    
    # 1. ส่วนหัว
    st.markdown("""
        <div class="card-header-custom">
            <div class="app-icon">🌶️</div>
            <div class="subtitle">AI Expert System</div>
            <h1>Chili Doctor AI</h1>
        </div>
        
        <p class="description">
            ระบบผู้เชี่ยวชาญปัญญาประดิษฐ์สำหรับวินิจฉัยโรคพริกจากใบ <br>
            ด้วยเทคโนโลยี <strong>Deep Learning (EfficientNetB4)</strong>
        </p>
    """, unsafe_allow_html=True)

    # 2. ส่วนอัปโหลด
    file = st.file_uploader("", type=["jpg", "png", "jpeg"])
    
    if file is None:
        st.markdown("""
            <div style="text-align: center; margin-top: 10px;">
                <small style="color: #bbb;">รองรับไฟล์ JPG, PNG, JPEG</small>
            </div>
        """, unsafe_allow_html=True)

    # 3. ส่วนแสดงผล
    if file is not None:
        image = Image.open(file)
        
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, use_container_width=True)
            
        if st.button("🚀 เริ่มต้นวิเคราะห์โรค"):
            if model is None:
                st.error("❌ ไม่สามารถโหลดโมเดลได้")
            else:
                with st.spinner('AI กำลังประมวลผล...'):
                    predictions = import_and_predict(image, model)
                    class_names = ['healthy', 'leaf curl', 'leaf spot', 'whitefly', 'yellow']
                    class_index = np.argmax(predictions)
                    result_class = class_names[class_index]
                    confidence = np.max(predictions) * 100

                # เส้นคั่น
                st.markdown("<hr style='margin: 30px 0; border-top: 1px solid #eee;'>", unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div style="text-align: center;">
                        <h3 style="color: #333; margin-bottom: 5px;">ผลการวิเคราะห์</h3>
                        <h1 style="color: #ff4b4b; font-size: 2.2rem; margin: 0;">{result_class}</h1>
                        <p style="color: #777;">ความมั่นใจ: <b>{confidence:.2f}%</b></p>
                    </div>
                """, unsafe_allow_html=True)

                # --- คำแนะนำ ---
                treatment_text = ""
                bg_color = "#fff3cd"
                text_color = "#856404"
                border_color = "#ffecb5"
                icon = "⚠️"
                
                if result_class == 'healthy':
                    treatment_text = "ต้นพริกแข็งแรงดี! ไม่พบร่องรอยโรค หมั่นดูแลรดน้ำและใส่ปุ๋ยตามปกติ"
                    bg_color = "#d4edda"
                    text_color = "#155724"
                    border_color = "#c3e6cb"
                    icon = "🌿"
                elif result_class == 'leaf curl':
                    treatment_text = "โรคใบหงิกมักเกิดจากแมลงหวี่ขาว ให้กำจัดวัชพืชและใช้สารสกัดสะเดา หรือเชื้อราเมตาไรเซียมฉีดพ่น"
                    icon = "🌀"
                elif result_class == 'leaf spot':
                    treatment_text = "โรคใบจุดตากบ เกิดจากเชื้อรา ให้ตัดแต่งใบที่เป็นโรคเผาทำลาย และฉีดพ่นสารป้องกันเชื้อรา"
                    icon = "🍂"
                elif result_class == 'whitefly':
                      treatment_text = "พบแมลงหวี่ขาว ให้ใช้กับดักกาวเหนียวสีเหลือง หรือฉีดพ่นน้ำหมักสมุนไพร"
                      icon = "🪰"
                elif result_class == 'yellow':
                      treatment_text = "อาการใบเหลือง อาจเกิดจากการขาดสารอาหาร หรือไวรัส ควรตรวจสอบดินและใส่ปุ๋ยบำรุง"
                      icon = "🟡"
                
                st.markdown(f"""
                    <div style="background-color: {bg_color}; color: {text_color}; border: 1px solid {border_color}; padding: 20px; border-radius: 12px; margin-top: 15px; font-size: 0.95rem;">
                        <div style="display: flex; align-items: start;">
                            <div style="font-size: 1.8rem; margin-right: 15px;">{icon}</div>
                            <div>
                                <strong style="display: block; margin-bottom: 5px;">คำแนะนำ:</strong>
                                {treatment_text}
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

# 4. Footer - อยู่นอกการ์ด
st.markdown("""
    <div class="footer-credit">
        โครงงานวิจัยทางคอมพิวเตอร์ <br>
        <strong>มหาวิทยาลัยราชภัฏอุบลราชธานี</strong> <br>
        <span class="badge-custom">v.1.0 Final</span>
        <div style="margin-top: 10px; font-size: 0.75rem; color: #aaa;">
            พัฒนาโดย: แมวสีขาวเทา และผองเพื่อน
        </div>
    </div>
""", unsafe_allow_html=True)