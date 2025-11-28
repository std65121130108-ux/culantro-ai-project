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

# --- 2. 🎨 CSS ตกแต่ง (Clean White Theme - ไม่มีกรอบ) ---
st.markdown("""
<style>
    /* นำเข้าฟอนต์ Prompt */
    @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');
    
    /* บังคับฟอนต์ทั้งหน้า */
    html, body, [class*="css"] {
        font-family: 'Prompt', sans-serif;
        color: #333;
    }
    
    /* 1. พื้นหลัง: สีขาวสะอาดตา (White Background) */
    .stApp {
        background-color: #ffffff;
        background-image: radial-gradient(#ff4b2b 0.5px, transparent 0.5px);
        background-size: 20px 20px; /* ลายจุดจางๆ สีแดง ให้ดูไม่โล่งเกินไป */
        opacity: 1;
    }

    /* 2. จัดการ Layout ให้ดูโปร่ง */
    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 700px;
    }
    
    /* 3. หัวข้อ (Header) */
    .header-container {
        text-align: center;
        margin-bottom: 40px;
    }
    .app-icon {
        font-size: 60px;
        margin-bottom: 10px;
        display: inline-block;
        animation: float 3s ease-in-out infinite;
    }
    .app-title {
        font-weight: 800;
        font-size: 2.5rem;
        background: linear-gradient(90deg, #FF416C 0%, #FF4B2B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        padding: 0;
        letter-spacing: -1px;
    }
    .app-subtitle {
        color: #666;
        font-weight: 500;
        font-size: 1.1rem;
        margin-top: 10px;
    }
    
    /* 4. ช่องอัปโหลดไฟล์ (File Uploader) - แบบเรียบ */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #FAFAFA !important;
        border: 2px dashed #FF4B2B !important;
        border-radius: 20px !important;
        padding: 30px !important;
        transition: all 0.3s;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        background-color: #FFF0F0 !important;
        transform: scale(1.01);
    }
    [data-testid="stFileUploaderDropzone"] div div::before {
        content: "📸 อัปโหลดรูปภาพใบพริกที่นี่";
        color: #555;
        font-weight: 600;
        font-size: 1rem;
    }
    
    /* 5. ปุ่มกด (Button) */
    div.stButton > button {
        background: linear-gradient(90deg, #FF416C 0%, #FF4B2B 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 15px 30px !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        box-shadow: 0 10px 20px rgba(255, 75, 43, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    div.stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 30px rgba(255, 75, 43, 0.5) !important;
    }
    
    /* 6. Footer */
    .footer {
        text-align: center;
        margin-top: 50px;
        color: #999;
        font-size: 0.8rem;
        border-top: 1px solid #eee;
        padding-top: 20px;
    }
    
    /* Animation */
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    
    /* ซ่อน Header/Footer เดิม */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

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

# 1. ส่วนหัว (Header) - วางกลางจอ ไม่มีกรอบ
st.markdown("""
    <div class="header-container">
        <div class="app-icon">🌶️</div>
        <h1 class="app-title">Chili Doctor AI</h1>
        <p class="app-subtitle">
            ระบบผู้เชี่ยวชาญตรวจวินิจฉัยโรคพริกอัจฉริยะ<br>
            <span style="font-size: 0.9rem; color: #999;">Deep Learning Technology (EfficientNetB4)</span>
        </p>
    </div>
""", unsafe_allow_html=True)

# 2. ส่วนอัปโหลด (File Uploader) - วางโล่งๆ
file = st.file_uploader("", type=["jpg", "png", "jpeg"])

# ข้อความเตือนเล็กๆ เมื่อยังไม่เลือกไฟล์
if file is None:
    st.markdown("""
        <div style="text-align: center; color: #bbb; margin-top: 10px; font-size: 0.9rem;">
            รองรับไฟล์ JPG, PNG (ขนาดไม่เกิน 200MB)
        </div>
    """, unsafe_allow_html=True)

# 3. ส่วนผลลัพธ์ (Result)
if file is not None:
    image = Image.open(file)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # แสดงรูปภาพ (จัดกึ่งกลาง)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, use_container_width=True)
        
    # ปุ่มกด
    if st.button("🔍 วิเคราะห์โรค"):
        if model is None:
            st.error("❌ ไม่สามารถโหลดโมเดลได้")
        else:
            with st.spinner('🤖 AI กำลังประมวลผล...'):
                predictions = import_and_predict(image, model)
                class_names = ['healthy', 'leaf curl', 'leaf spot', 'whitefly', 'yellow']
                class_index = np.argmax(predictions)
                result_class = class_names[class_index]
                confidence = np.max(predictions) * 100

            st.markdown("<hr style='margin: 30px 0; border-top: 1px solid #eee;'>", unsafe_allow_html=True)
            
            # ผลลัพธ์
            st.markdown(f"""
                <div style="text-align: center;">
                    <h3 style="color: #555; margin: 0; font-size: 1.2rem;">ผลการวิเคราะห์</h3>
                    <h1 style="background: linear-gradient(90deg, #FF416C 0%, #FF4B2B 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 3rem; margin: 10px 0; font-weight: 800;">{result_class}</h1>
                    <div style="display: inline-block; background: #f0f0f0; padding: 5px 15px; border-radius: 20px; color: #555; font-size: 0.9rem;">
                        ความมั่นใจ: <b>{confidence:.2f}%</b>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # คำแนะนำ
            treatment_text = ""
            bg_color = "#FFF8E1" # สีเหลืองอ่อน
            text_color = "#FF6F00"
            border_color = "#FFECB3"
            icon = "💡"
            
            if result_class == 'healthy':
                treatment_text = "ต้นพริกแข็งแรงดี! ไม่พบร่องรอยโรค หมั่นดูแลรดน้ำและใส่ปุ๋ยตามปกติ"
                bg_color = "#E8F5E9"
                text_color = "#2E7D32"
                border_color = "#C8E6C9"
                icon = "🌿"
            elif result_class == 'leaf curl':
                treatment_text = "โรคใบหงิกมักเกิดจากแมลงหวี่ขาว ให้กำจัดวัชพืชและใช้สารสกัดสะเดา หรือเชื้อราเมตาไรเซียมฉีดพ่น"
            elif result_class == 'leaf spot':
                treatment_text = "โรคใบจุดตากบ เกิดจากเชื้อรา ให้ตัดแต่งใบที่เป็นโรคเผาทำลาย และฉีดพ่นสารป้องกันเชื้อรา"
            elif result_class == 'whitefly':
                 treatment_text = "พบแมลงหวี่ขาว ให้ใช้กับดักกาวเหนียวสีเหลือง หรือฉีดพ่นน้ำหมักสมุนไพร"
            elif result_class == 'yellow':
                 treatment_text = "อาการใบเหลือง อาจเกิดจากการขาดสารอาหาร หรือไวรัส ควรตรวจสอบดินและใส่ปุ๋ยบำรุง"
            
            st.markdown(f"""
                <div style="background-color: {bg_color}; color: {text_color}; padding: 25px; border-radius: 20px; margin-top: 25px; border: 1px solid {border_color}; line-height: 1.6; text-align: left;">
                    <strong style="display: block; margin-bottom: 5px; font-size: 1.1rem;">{icon} คำแนะนำ:</strong>
                    {treatment_text}
                </div>
            """, unsafe_allow_html=True)

# 4. Footer
st.markdown("""
    <div class="footer">
        โครงงานวิจัยทางคอมพิวเตอร์ • มหาวิทยาลัยราชภัฏอุบลราชธานี<br>
        พัฒนาโดย: แมวสีขาวเทา และผองเพื่อน
    </div>
""", unsafe_allow_html=True)