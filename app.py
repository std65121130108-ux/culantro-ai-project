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

# --- 2. 🎨 CSS ตกแต่ง (Premium & Clean Design) ---
st.markdown("""
<style>
    /* นำเข้าฟอนต์ Prompt */
    @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');
    
    /* บังคับฟอนต์ทั้งหน้า */
    html, body, [class*="css"] {
        font-family: 'Prompt', sans-serif;
    }
    
    /* 1. พื้นหลังหลัก (Background): สีส้มแดงไล่เฉด */
    .stApp {
        background: linear-gradient(135deg, #FF416C 0%, #FF4B2B 100%);
    }

    /* 2. ปรับแต่ง "กรอบสีขาว" (Card) ให้สวยหรู */
    /* เป้าหมายคือ st.container(border=True) */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF; /* สีขาวทึบ */
        border-radius: 24px;
        padding: 40px;
        box-shadow: 0 20px 50px rgba(0,0,0,0.15); /* เงานุ่มๆ */
        border: none;
        margin-bottom: 20px;
    }
    
    /* 3. จัดการข้อความ (Typography) */
    .app-title {
        color: #333333;
        font-weight: 700;
        font-size: 2.2rem;
        margin: 0;
        padding: 0;
        text-align: center;
        letter-spacing: -0.5px;
    }
    .app-subtitle {
        color: #FF4B2B;
        font-weight: 600;
        font-size: 0.9rem;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        text-align: center;
        margin-bottom: 8px;
    }
    .app-desc {
        color: #666666;
        text-align: center;
        font-size: 1.1rem;
        margin-top: 15px;
        line-height: 1.6;
    }
    .app-note {
        color: #FF4B2B;
        font-size: 0.9rem;
        text-align: center;
        font-weight: 500;
        margin-bottom: 30px;
    }
    
    /* 4. ไอคอนพริกตรงกลาง */
    .icon-wrapper {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }
    .main-icon {
        font-size: 60px;
        background: #fff5f5;
        border-radius: 50%;
        width: 100px;
        height: 100px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 10px 25px rgba(255, 75, 43, 0.15);
    }

    /* 5. ปรับช่องอัปโหลดไฟล์ (File Uploader) ให้ดูดี */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #FAFAFA;
        border: 2px dashed #E0E0E0;
        border-radius: 16px;
        padding: 30px 20px;
        transition: all 0.3s;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #FF4B2B;
        background-color: #FFF0F0;
    }
    /* ซ่อนข้อความเล็กๆ เดิมของ Streamlit แล้วใส่ข้อความใหม่ */
    [data-testid="stFileUploaderDropzone"] div div::before {
        content: "📂 คลิก หรือ ลากไฟล์รูปภาพมาวางที่นี่";
        font-size: 1.1rem;
        color: #555;
        font-weight: 500;
        display: block;
        margin-bottom: 8px;
    }
    [data-testid="stFileUploaderDropzone"] div div small {
        display: none;
    }
    
    /* 6. ปุ่มกด (Button) */
    div.stButton > button {
        background: linear-gradient(90deg, #FF416C 0%, #FF4B2B 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 12px 30px;
        font-size: 1.2rem;
        font-weight: 600;
        width: 100%;
        box-shadow: 0 10px 25px rgba(255, 75, 43, 0.3);
        margin-top: 10px;
        transition: all 0.3s;
    }
    div.stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(255, 75, 43, 0.5);
    }
    
    /* ซ่อน Header/Footer */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* จัดการการ์ดผลลัพธ์ */
    .result-card {
        background-color: #F0FFF4;
        border: 1px solid #C3E6CB;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. โหลดโมเดล (ฟังก์ชันเดิม) ---
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

# ==========================================
# ⬜ ส่วน Input (การ์ดใบที่ 1)
# ==========================================
with st.container(border=True): # สร้างกรอบขาว
    
    # ไอคอน + หัวข้อ
    st.markdown("""
        <div class="icon-wrapper">
            <div class="main-icon">🌶️</div>
        </div>
        <div class="app-subtitle">AI Expert System</div>
        <div class="app-title">Chili Doctor AI</div>
        
        <div class="app-desc">
            ระบบผู้เชี่ยวชาญปัญญาประดิษฐ์เพื่อวินิจฉัยโรคของพริกจากใบ
        </div>
        <div class="app-note">
            (กรุณาอัปโหลดรูปภาพที่เห็นใบพริกชัดเจน)
        </div>
    """, unsafe_allow_html=True)

    # ช่องอัปโหลดไฟล์ (อยู่ในกรอบขาวเดียวกัน)
    file = st.file_uploader("", type=["jpg", "png", "jpeg"])

# ==========================================
# ⬜ ส่วน Result (การ์ดใบที่ 2 - แสดงเมื่อมีไฟล์)
# ==========================================
if file is not None:
    # สร้างกรอบขาวอีกอันแยกออกมา
    with st.container(border=True):
        image = Image.open(file)
        
        # จัดรูปให้อยู่ตรงกลาง
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, use_container_width=True)
        
        # ปุ่มกดวิเคราะห์
        if st.button("🔍 วิเคราะห์โรคเดี๋ยวนี้"):
            if model is None:
                st.error("❌ ไม่สามารถโหลดโมเดลได้")
            else:
                with st.spinner('🤖 AI กำลังประมวลผล...'):
                    predictions = import_and_predict(image, model)
                    class_names = ['healthy', 'leaf curl', 'leaf spot', 'whitefly', 'yellow']
                    class_index = np.argmax(predictions)
                    result_class = class_names[class_index]
                    confidence = np.max(predictions) * 100

                # แสดงเส้นคั่น
                st.markdown("<hr style='margin: 30px 0; border-top: 1px solid #eee;'>", unsafe_allow_html=True)
                
                # แสดงผลลัพธ์
                st.markdown(f"""
                    <div style="text-align: center; margin-bottom: 20px;">
                        <h3 style="color: #555; margin: 0; font-size: 1.2rem;">ผลการวิเคราะห์</h3>
                        <h1 style="color: #28a745; font-size: 3rem; margin: 10px 0; font-weight: 800;">{result_class}</h1>
                        <p style="color: #888; font-size: 1.1rem;">ความมั่นใจ: <b>{confidence:.2f}%</b></p>
                    </div>
                """, unsafe_allow_html=True)

                # คำแนะนำ (Card ย่อย)
                treatment_text = ""
                bg_color = "#fff3cd"
                text_color = "#856404"
                icon = "⚠️"

                if result_class == 'healthy':
                    treatment_text = "ต้นพริกแข็งแรงดี! ไม่พบร่องรอยโรค หมั่นดูแลรดน้ำและใส่ปุ๋ยตามปกติ"
                    bg_color = "#d4edda"
                    text_color = "#155724"
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
                    <div style="background-color: {bg_color}; color: {text_color}; padding: 25px; border-radius: 16px; text-align: left; font-size: 1.1rem; line-height: 1.6;">
                        <strong style="display:block; margin-bottom:10px; font-size:1.2rem;">{icon} คำแนะนำ:</strong>
                        {treatment_text}
                    </div>
                """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style="text-align: center; margin-top: 40px; color: rgba(255,255,255,0.8); font-size: 0.9rem;">
        โครงงานวิจัยทางคอมพิวเตอร์ • มหาวิทยาลัยราชภัฏอุบลราชธานี<br>
        พัฒนาโดย: แมวสีขาวเทา และผองเพื่อน
    </div>
""", unsafe_allow_html=True)