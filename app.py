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

# --- 2. 🎨 CSS ตกแต่ง (Update: รองรับ Tabs และ Camera) ---
def local_css():
    st.markdown("""
    <style>
        /* นำเข้าฟอนต์ Prompt */
        @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');
        
        /* 1. Global Settings */
        html, body, [class*="css"] {
            font-family: 'Prompt', sans-serif;
        }

        /* 2. พื้นหลัง Gradient */
        .stApp {
            background: linear-gradient(135deg, #FF416C 0%, #FF4B2B 100%) !important;
            background-attachment: fixed !important;
            background-size: cover !important;
        }

        /* 3. Block Container (การ์ดขาวใบใหญ่) */
        div.block-container {
            background-color: rgba(255, 255, 255, 0.95) !important;
            border-radius: 25px !important;
            padding: 3rem 2rem !important;
            margin-top: 2rem !important;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3) !important;
            max-width: 700px !important;
        }

        /* ตัวอักษรสีเข้ม */
        div.block-container h1, div.block-container h2, div.block-container h3, 
        div.block-container p, div.block-container span, div.block-container div, 
        div.block-container label, div.block-container small {
             color: #333333 !important;
        }
        
        /* ยกเว้น Text ในปุ่มกด */
        div.stButton > button p { color: white !important; }

        /* ซ่อน Header/Footer เดิม */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}

        /* 4. Custom Elements */
        .app-icon {
            width: 100px;
            height: 100px;
            background: linear-gradient(45deg, #ff9a9e 0%, #fad0c4 99%, #fad0c4 100%) !important;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 50px;
            margin: 0 auto 20px;
            box-shadow: 0 6px 20px rgba(255, 75, 43, 0.4) !important;
            cursor: default;
        }
        
        .subtitle {
            color: #d32f2f !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
            letter-spacing: 2px !important;
            text-transform: uppercase !important;
            text-align: center !important;
            margin-bottom: 5px !important;
        }
        
        h1 {
            font-weight: 800 !important;
            font-size: 2.2rem !important;
            margin: 0 !important;
            padding: 0 !important;
            text-align: center !important;
        }

        .description {
            font-size: 1rem !important;
            line-height: 1.6 !important;
            text-align: center !important;
            margin: 20px 0 30px 0 !important;
        }

        /* 5. ปุ่มกด */
        div.stButton > button {
            background: linear-gradient(90deg, #FF416C 0%, #FF4B2B 100%) !important;
            border: none !important;
            color: white !important;
            padding: 15px 40px !important;
            border-radius: 50px !important;
            font-size: 1.1rem !important;
            font-weight: 600 !important;
            box-shadow: 0 5px 15px rgba(255, 65, 108, 0.4) !important;
            width: 100% !important;
            transition: all 0.3s ease !important;
        }
        div.stButton > button:hover {
            transform: scale(1.02) !important;
            box-shadow: 0 8px 25px rgba(255, 65, 108, 0.5) !important;
        }

        /* 6. File Uploader */
        [data-testid="stFileUploaderDropzone"] {
            background-color: rgba(240, 240, 240, 0.5) !important;
            border: 2px dashed #FF4B2B !important;
            border-radius: 15px !important;
            padding: 20px !important;
        }
        [data-testid="stFileUploaderDropzone"] button {
             border: none !important;
             background: #FF4B2B !important;
             color: white !important;
        }

        /* 7. Tabs Styling (ตกแต่งแท็บ) */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background-color: transparent;
            margin-bottom: 20px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 45px;
            background-color: #f0f0f0;
            border-radius: 20px;
            padding: 0px 20px;
            color: #666;
            font-weight: 600;
            border: none;
        }
        .stTabs [aria-selected="true"] {
            background-color: #ffe5e5 !important;
            color: #FF4B2B !important;
            border: 1px solid #FF4B2B !important;
        }

        /* 8. Footer */
        .footer-credit {
            font-size: 0.8rem !important;
            color: #888 !important;
            margin-top: 30px !important;
            padding-top: 20px !important;
            text-align: center !important;
            border-top: 1px solid rgba(0,0,0,0.1) !important;
        }
        .badge-custom {
            background-color: #f0f0f0 !important;
            color: #333 !important;
            padding: 0.35em 0.8em !important;
            font-size: 0.75em !important;
            font-weight: 700 !important;
            border-radius: 20px !important;
            display: inline-block !important;
            margin-top: 10px !important;
        }
    </style>
    """, unsafe_allow_html=True)

local_css()

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

# 1. ส่วนหัว
st.markdown("""
    <div style="text-align: center;">
        <div class="app-icon">🌶️</div>
        <div class="subtitle">AI Expert System</div>
        <h1>Chili Doctor AI</h1>
        <p class="description">
            ระบบผู้เชี่ยวชาญปัญญาประดิษฐ์สำหรับวินิจฉัยโรคพริกจากใบ <br>
            ด้วยเทคโนโลยี <strong>Deep Learning (EfficientNetB4)</strong> <br>
            ความแม่นยำสูง รวดเร็ว และใช้งานง่าย
        </p>
    </div>
""", unsafe_allow_html=True)

# 2. ส่วนรับข้อมูล (Tabs: ถ่ายภาพ / อัปโหลด)
tab_cam, tab_up = st.tabs(["📸 ถ่ายภาพใบพริก", "📂 อัปโหลดไฟล์รูป"])

img_file_buffer = None

with tab_cam:
    st.markdown("<div style='text-align: center; color: #666; margin-bottom: 10px;'>กดปุ่มด้านล่างเพื่อถ่ายรูป</div>", unsafe_allow_html=True)
    camera_image = st.camera_input("กล้องถ่ายรูป", label_visibility="hidden")
    if camera_image is not None:
        img_file_buffer = camera_image

with tab_up:
    uploaded_file = st.file_uploader("เลือกรูปภาพจากเครื่อง", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        img_file_buffer = uploaded_file

# 3. ส่วนแสดงผลและทำนาย
if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div style="border-radius: 15px; overflow: hidden; box-shadow: 0 5px 15px rgba(0,0,0,0.1); border: 3px solid rgba(255,255,255,0.8);">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    if st.button("🚀 วินิจฉัยโรคทันที"):
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
            st.markdown("<div style='height: 1px; background-color: rgba(0,0,0,0.1); margin: 30px 0;'></div>", unsafe_allow_html=True)
            
            # ผลลัพธ์
            st.markdown(f"""
                <div style="text-align: center;">
                    <h3 style="color: #666; font-size: 1rem; margin-bottom: 5px;">ผลการวิเคราะห์</h3>
                    <h1 style="color: #FF4B2B !important; font-size: 2.5rem; margin: 0;">{result_class.upper()}</h1>
                    <div style="background: #fff0f0; color: #FF4B2B; display: inline-block; padding: 5px 15px; border-radius: 20px; font-weight: bold; font-size: 0.9rem; margin-top: 10px;">
                        ความมั่นใจ: {confidence:.2f}%
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Icon & Treatment Logic
            treatment_text = ""
            bg_color = "rgba(255, 248, 225, 0.8)"
            icon_bg = "#ffecb3"
            icon = "⚠️"
            
            if result_class == 'healthy':
                treatment_text = "ต้นพริกแข็งแรงดี! ไม่พบร่องรอยโรค หมั่นดูแลรดน้ำและใส่ปุ๋ยตามปกติ"
                bg_color = "rgba(232, 245, 233, 0.8)"
                icon_bg = "#c8e6c9"
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
            
            # กล่องคำแนะนำ
            st.markdown(f"""
                <div style="background-color: {bg_color}; padding: 25px; border-radius: 15px; margin-top: 25px; text-align: left; border: 1px solid rgba(0,0,0,0.05);">
                    <div style="display: flex; align-items: start;">
                        <div style="background: white; width: 45px; height: 45px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.4rem; margin-right: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); flex-shrink: 0;">
                            {icon}
                        </div>
                        <div>
                            <strong style="display: block; margin-bottom: 5px; color: #333; font-size: 1rem;">คำแนะนำการดูแลรักษา</strong>
                            <span style="color: #555; line-height: 1.5; font-size: 0.9rem;">{treatment_text}</span>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

# 4. Footer
st.markdown("""
    <div class="footer-credit">
        โครงงานวิจัยทางคอมพิวเตอร์ <br>
        <strong>มหาวิทยาลัยราชภัฏอุบลราชธานี</strong> <br>
        <span class="badge-custom">v.1.0 (Final Release)</span> <br>
        <div style="margin-top: 10px; font-size: 0.75rem; color: #aaa;">
            พัฒนาโดย: แมวสีขาวเทา และผองเพื่อน
        </div>
    </div>
""", unsafe_allow_html=True)