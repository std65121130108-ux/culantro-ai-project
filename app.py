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

# --- 🎨 ส่วนตกแต่ง CSS (เวอร์ชันกรอบขาวเด่นชัด) ---
st.markdown("""
<style>
    /* นำเข้าฟอนต์ Prompt */
    @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;600&display=swap');
    
    /* บังคับฟอนต์ทั้งหน้า */
    html, body, [class*="css"] {
        font-family: 'Prompt', sans-serif;
    }
    
    /* 1. พื้นหลังหลัก: สีส้มแดงสดใส */
    .stApp, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #FF416C 0%, #FF4B2B 100%) !important;
    }

    /* 2. ลบพื้นหลังกล่องใหญ่เดิมออก (เพื่อให้กรอบเล็กข้างในเด่น) */
    [data-testid="block-container"] {
        background: transparent !important; /* เปลี่ยนเป็นใส */
        box-shadow: none !important;
        padding: 2rem 1rem !important;
        max-width: 700px;
    }

    /* หัวข้อ h1 */
    h1 {
        color: #333 !important;
        font-weight: 700 !important;
        text-align: center;
    }
    
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* 3. ปรับแต่งปุ่มกด (Button) */
    div.stButton > button {
        background: linear-gradient(90deg, #FF416C 0%, #FF4B2B 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 0.6rem 2rem !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        box-shadow: 0 4px 15px rgba(255, 65, 108, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 65, 108, 0.6) !important;
        color: white !important;
    }
    
    /* 4. ⭐⭐⭐ พระเอกของเรา: กรอบสีขาว (Card) ⭐⭐⭐ */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important; /* พื้นหลังสีขาวทึบ */
        border: 1px solid rgba(0,0,0,0.1) !important;
        border-radius: 25px !important;
        padding: 30px !important;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2) !important; /* เงาลอยเด่น */
        margin-bottom: 20px !important;
    }
    
    /* File Uploader Area */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #f8f9fa !important;
        border: 2px dashed #FF4B2B !important; /* เส้นประสีส้ม */
        border-radius: 15px !important;
        padding: 20px !important;
    }
    div[data-testid="stFileUploaderDropzone"] div {
        color: #555 !important;
    }

    /* Custom Header Style */
    .custom-header {
        text-align: center;
        margin-bottom: 20px;
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
        margin: 0 auto 15px;
        box-shadow: 0 4px 15px rgba(255, 75, 43, 0.3);
        animation: pulse 2s infinite;
        border: 4px solid white;
    }
    .subtitle {
        color: #d32f2f;
        font-weight: 500;
        font-size: 0.9rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(255, 75, 43, 0.4); }
        70% { transform: scale(1.05); box-shadow: 0 0 0 10px rgba(255, 75, 43, 0); }
        100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(255, 75, 43, 0); }
    }
    
    /* จัดการข้อความ info ให้สวยงาม */
    .stAlert {
        border-radius: 10px;
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

# --- ⭐ สร้างกรอบขาว (Card) ครอบทั้งหมดตามที่ต้องการ ⭐ ---
with st.container(border=True):
    # 1. ส่วนหัว (Icon + Title)
    st.markdown("""
        <div class="custom-header">
            <div class="app-icon">🌶️</div>
            <div class="subtitle">AI Expert System</div>
            <h1 style="margin-top: 0; color: #333;">Chili Doctor AI</h1>
        </div>
    """, unsafe_allow_html=True)

    # 2. คำอธิบาย
    st.markdown("""
    <p style="text-align: center; color: #555; margin-bottom: 20px; line-height: 1.6;">
        ระบบผู้เชี่ยวชาญปัญญาประดิษฐ์เพื่อวินิจฉัยโรคของพริกจากใบ <br>
        <span style="font-size: 0.9rem; color: #888;">(กรุณาอัปโหลดรูปภาพที่เห็นใบพริกชัดเจน)</span>
    </p>
    """, unsafe_allow_html=True)

    # 3. ช่องอัปโหลดไฟล์
    file = st.file_uploader("", type=["jpg", "png", "jpeg"])
    
    # 4. ข้อความแนะนำ (ใส่ไว้ในกรอบตามที่ขอ)
    if file is None:
        st.markdown("""
            <div style="text-align: center; color: #888; margin-top: 10px; font-size: 0.9rem;">
                👆 กรุณาเลือกรูปภาพ (.jpg, .png) จากเครื่องของคุณ
            </div>
        """, unsafe_allow_html=True)

# --- ส่วนแสดงผลลัพธ์ (อยู่นอกกรอบ Card หลัก จะได้ดูแยกส่วนกัน) ---
if file is not None:
    image = Image.open(file)
    
    # กรอบแสดงผลลัพธ์
    with st.container(border=True):
        st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div><br>', unsafe_allow_html=True)
        
        if st.button("🔍 วิเคราะห์โรค"):
            with st.spinner('AI กำลังวิเคราะห์ข้อมูล...'):
                predictions = import_and_predict(image, model)
                class_index = np.argmax(predictions)
                result_class = class_names[class_index]
                confidence = np.max(predictions) * 100

            # การ์ดผลลัพธ์
            st.markdown(f"""
                <div style="background-color: #f0fff4; border: 1px solid #c3e6cb; padding: 20px; border-radius: 15px; text-align: center; margin-bottom: 15px;">
                    <h4 style="margin:0; color: #155724; font-weight: 600;">ผลการวิเคราะห์: <span style="font-size: 1.4rem;">{result_class}</span></h4>
                </div>
                <p style="text-align: center; color: #6c757d; font-size: 0.9rem;">ความมั่นใจ (Confidence): <b>{confidence:.2f}%</b></p>
            """, unsafe_allow_html=True)

            # คำแนะนำ
            treatment_text = ""
            treatment_bg = "#fff8e1"
            treatment_border = "#ffeeba"
            text_color = "#856404"

            if result_class == 'healthy':
                treatment_text = "✅ **ต้นพริกแข็งแรงดี!** ไม่พบร่องรอยโรค หมั่นดูแลรดน้ำตามปกติ"
                treatment_bg = "#d4edda"
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
                 
            st.markdown(f"""
                <div style="background-color: {treatment_bg}; color: {text_color}; padding: 18px; border-radius: 12px; border: 1px solid {treatment_border}; line-height: 1.6;">
                    {treatment_text}
                </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 60px; color: #e0e0e0; font-size: 0.8rem; border-top: 1px solid rgba(255,255,255,0.3); padding-top: 20px;">
    โครงงานวิจัยทางคอมพิวเตอร์ • มหาวิทยาลัยราชภัฏอุบลราชธานี<br>
    <span style="font-size: 0.75rem;">พัฒนาโดย: แมวสีขาวเทา และผองเพื่อน</span>
</div>
""", unsafe_allow_html=True)