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

# --- 2. 🎨 CSS ตกแต่ง (White Card Theme - Single Card Layout) ---
st.markdown("""
<style>
    /* นำเข้าฟอนต์ Prompt */
    @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');
    
    /* บังคับฟอนต์ทั้งหน้า */
    html, body, [class*="css"] {
        font-family: 'Prompt', sans-serif;
    }
    
    /* 1. พื้นหลังหลัก (Background): Gradient สีส้มแดง */
    .stApp, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #FF416C 0%, #FF4B2B 100%) !important;
    }

    /* 2. ปรับแต่ง "กรอบ/การ์ด" (Container) ให้เป็นสีขาวทึบ */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
        border-radius: 24px !important;
        border: none !important;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2) !important;
        padding: 40px 30px !important;
        margin-bottom: 20px;
    }
    
    /* ป้องกันสีพื้นหลังซ้อนทับ */
    [data-testid="stVerticalBlockBorderWrapper"] > div {
        background-color: transparent !important;
    }
    
    /* ซ่อน Header/Footer เดิม */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* 3. ส่วนหัว (Icon & Titles) */
    .card-header-custom {
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
        margin: 0 auto 20px;
        box-shadow: 0 4px 15px rgba(255, 75, 43, 0.3);
        animation: pulse 2s infinite;
        border: none;
    }
    
    .subtitle {
        color: #d32f2f;
        font-weight: 500;
        font-size: 0.9rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    
    h1 {
        color: #333 !important;
        font-weight: 600 !important;
        font-size: 1.8rem !important;
        margin: 0 !important;
        padding: 0 !important;
        text-align: center;
    }
    
    /* 4. คำอธิบาย (Description) */
    .description {
        color: #666;
        font-size: 0.95rem;
        line-height: 1.6;
        text-align: center;
        margin-bottom: 30px;
    }

    /* 5. ปุ่มกด (Button) */
    div.stButton > button {
        background: linear-gradient(90deg, #FF416C 0%, #FF4B2B 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 12px 40px !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(255, 65, 108, 0.4) !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
    }
    div.stButton > button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 6px 20px rgba(255, 65, 108, 0.6) !important;
        color: white !important;
    }
    
    /* 6. File Uploader */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #f9f9f9 !important;
        border: 2px dashed #FF4B2B !important;
        border-radius: 15px !important;
    }
    
    /* 7. Footer */
    .footer-credit {
        font-size: 0.8rem;
        color: #fff;
        margin-top: 30px;
        padding-top: 15px;
        text-align: center;
        opacity: 0.8;
    }
    .badge-custom {
        background-color: rgba(255,255,255,0.2);
        color: #fff;
        padding: 0.35em 0.65em;
        font-size: 0.75em;
        font-weight: 700;
        border-radius: 0.25rem;
        display: inline-block;
        margin-top: 10px;
    }

    /* Animation Keyframes */
    @keyframes pulse {
        0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(255, 75, 43, 0.4); }
        70% { transform: scale(1.05); box-shadow: 0 0 0 10px rgba(255, 75, 43, 0); }
        100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(255, 75, 43, 0); }
    }
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

# --- ⭐ สร้างกรอบขาว (Card) เดียวครอบทั้งหมด ⭐ ---
# สังเกตว่าเราเปิด with st.container() แค่ครั้งเดียวตรงนี้ แล้วเอาทุกอย่างใส่เข้าไป
with st.container(border=True):
    
    # 1. ส่วนหัว (Icon + Titles)
    st.markdown("""
        <div class="card-header-custom">
            <div class="app-icon">🌶️</div>
            <div class="subtitle">AI Expert System</div>
            <h1>Chili Doctor AI</h1>
        </div>
        
        <p class="description">
            ระบบผู้เชี่ยวชาญปัญญาประดิษฐ์สำหรับวินิจฉัยโรคพริกจากใบ <br>
            ด้วยเทคโนโลยี <strong>Deep Learning (EfficientNetB4)</strong> <br>
            ความแม่นยำสูง รวดเร็ว และใช้งานง่าย
        </p>
    """, unsafe_allow_html=True)

    # 2. ส่วนอัปโหลด
    file = st.file_uploader("", type=["jpg", "png", "jpeg"])
    
    if file is None:
        st.markdown("""
            <div style="text-align: center; margin-top: 10px;">
                <small style="color: #999;">*แนะนำให้เปิดผ่าน Google Chrome หรือ Safari</small>
            </div>
        """, unsafe_allow_html=True)

    # 3. ส่วนแสดงผล (ย้ายเข้ามาอยู่ใน Indent ของ container แล้ว! ตอนนี้จะอยู่ในกรอบขาวเดียวกัน)
    if file is not None:
        image = Image.open(file)
        
        st.markdown("<br>", unsafe_allow_html=True)
        # จัดรูปให้อยู่ตรงกลางสวยๆ
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

                # เส้นคั่นภายในการ์ด
                st.markdown("<hr style='margin: 30px 0; border-top: 1px solid #eee;'>", unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div style="text-align: center;">
                        <h3 style="color: #333; margin-bottom: 5px;">ผลการวิเคราะห์</h3>
                        <h1 style="color: #FF4B2B; font-size: 2.2rem; margin: 0;">{result_class}</h1>
                        <p style="color: #777;">ความมั่นใจ: <b>{confidence:.2f}%</b></p>
                    </div>
                """, unsafe_allow_html=True)

                # --- จัดการ Icon และคำแนะนำตามโรค ---
                treatment_text = ""
                bg_color = "#fff3cd"
                text_color = "#856404"
                border_color = "#ffecb5"
                icon = "⚠️" # ไอคอนเริ่มต้น
                
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

# 4. Footer (Credit) - อยู่นอกการ์ด เพื่อความสวยงาม
st.markdown("""
    <div class="footer-credit">
        โครงงานวิจัยทางคอมพิวเตอร์ <br>
        <strong>มหาวิทยาลัยราชภัฏอุบลราชธานี</strong> <br>
        <span class="badge-custom">v.1.0 (Final Release)</span> <br>
        <div style="margin-top: 10px; font-size: 0.75rem; color: #eee;">
            พัฒนาโดย: แมวสีขาวเทา และผองเพื่อน
        </div>
    </div>
""", unsafe_allow_html=True)