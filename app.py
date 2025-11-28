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

# --- 2. 🎨 CSS ดีไซน์ใหม่ (Modern Minimalist Tech) ---
st.markdown("""
<style>
    /* นำเข้าฟอนต์ Prompt */
    @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');
    
    /* บังคับฟอนต์ทั้งหน้า */
    html, body, [class*="css"] {
        font-family: 'Prompt', sans-serif;
        color: #1a1a1a;
    }
    
    /* 1. พื้นหลัง: ลายจุด Modern Tech Pattern */
    .stApp {
        background-color: #f8f9fa;
        background-image: radial-gradient(#e0e0e0 1px, transparent 1px);
        background-size: 20px 20px;
    }

    /* 2. การ์ดหลัก (Main Card) */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important;
        border-radius: 24px !important;
        padding: 40px !important;
        box-shadow: 0 20px 40px -10px rgba(0,0,0,0.08) !important; /* เงาฟุ้งๆ นุ่มๆ */
        border: 1px solid rgba(0,0,0,0.05) !important;
        margin-bottom: 20px;
    }
    
    /* 3. Typography (จัดตัวหนังสือ) */
    .badge {
        background-color: #fff0f0;
        color: #ff4b4b;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        display: inline-block;
        margin-bottom: 15px;
    }
    
    h1 {
        color: #111;
        font-weight: 800 !important;
        font-size: 2.8rem !important;
        letter-spacing: -1.5px;
        margin-bottom: 10px !important;
    }
    
    .desc {
        color: #666;
        font-size: 1.1rem;
        font-weight: 400;
        line-height: 1.6;
        margin-bottom: 30px;
    }
    
    /* 4. ช่องอัปโหลด (Minimal Style) */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #fafafa !important;
        border: 2px dashed #e0e0e0 !important;
        border-radius: 16px !important;
        padding: 40px 20px !important;
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #ff4b4b !important;
        background-color: #fffbfb !important;
        transform: scale(1.01);
    }
    /* ข้อความใน Dropzone */
    [data-testid="stFileUploaderDropzone"] div div::before {
        content: "รูปภาพใบพริก";
        font-size: 1.1rem;
        font-weight: 600;
        color: #333;
        display: block;
        margin-bottom: 5px;
    }
    
    /* 5. ปุ่มกด (Black Button Style - เท่ๆ) */
    div.stButton > button {
        background-color: #111 !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 16px 32px !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
    }
    div.stButton > button:hover {
        background-color: #ff4b4b !important; /* เปลี่ยนเป็นสีแดงเมื่อชี้ */
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(255, 75, 75, 0.25) !important;
    }
    
    /* ซ่อน Header/Footer */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Result Styling */
    .result-box {
        background: #f8fff9;
        border-left: 6px solid #00c853;
        padding: 20px;
        border-radius: 8px;
        margin-top: 20px;
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
        with st.status("🚀 Initializing System...", expanded=True) as status:
            try:
                import gdown
                gdown.download(url, filename, quiet=False)
                if os.path.exists(filename):
                    status.update(label="System Ready!", state="complete", expanded=False)
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
# ⬜ ส่วน Input (Clean Card)
# ==========================================
with st.container(border=True):
    # Header Section
    st.markdown("""
        <div style="text-align: center;">
            <div class="badge">✨ AI Powered System</div>
            <h1>Chili Doctor</h1>
            <p class="desc">
                ระบบผู้เชี่ยวชาญตรวจวินิจฉัยโรคพริกอัจฉริยะ<br>
                <span style="color: #999; font-size: 0.9rem;">Upload a photo to start diagnosis</span>
            </p>
        </div>
    """, unsafe_allow_html=True)

    # File Uploader
    file = st.file_uploader("", type=["jpg", "png", "jpeg"])

# ==========================================
# ⬜ ส่วน Result (แสดงเมื่อมีไฟล์)
# ==========================================
if file is not None:
    with st.container(border=True):
        image = Image.open(file)
        
        # Grid Layout for Image
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, use_container_width=True)
            st.markdown('<p style="text-align: center; color: #999; font-size: 0.8rem; margin-top: 5px;">Image Preview</p>', unsafe_allow_html=True)
        
        # Analyze Button
        if st.button("Start Analysis ⚡"):
            if model is None:
                st.error("Error: Model not found.")
            else:
                with st.spinner('Processing image data...'):
                    predictions = import_and_predict(image, model)
                    class_names = ['healthy', 'leaf curl', 'leaf spot', 'whitefly', 'yellow']
                    class_index = np.argmax(predictions)
                    result_class = class_names[class_index]
                    confidence = np.max(predictions) * 100

                # Result Design
                st.markdown("<hr style='border-top: 1px solid #eee; margin: 30px 0;'>", unsafe_allow_html=True)
                
                # Main Result
                st.markdown(f"""
                    <div style="text-align: center;">
                        <h2 style="color: #333; margin: 0; font-size: 1.5rem;">Analysis Result</h2>
                        <h1 style="color: #00c853; font-size: 3.5rem; font-weight: 800; margin: 10px 0; letter-spacing: -2px;">{result_class}</h1>
                        <div style="display: inline-block; background: #eee; padding: 5px 15px; border-radius: 15px; font-size: 0.9rem; font-weight: 600; color: #555;">
                            Confidence Score: {confidence:.2f}%
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                # Recommendation Card
                rec_title = ""
                rec_detail = ""
                rec_color = "#333"
                
                if result_class == 'healthy':
                    rec_title = "Healthy Plant"
                    rec_detail = "ต้นพริกของคุณแข็งแรงดีครับ! ไม่พบร่องรอยของโรคใดๆ แนะนำให้ดูแลรดน้ำและใส่ปุ๋ยตามปกติ"
                    rec_color = "#00c853"
                elif result_class == 'leaf curl':
                    rec_title = "Leaf Curl Disease"
                    rec_detail = "โรคใบหงิก: มักเกิดจากแมลงหวี่ขาวเป็นพาหะ แนะนำให้กำจัดวัชพืชรอบแปลง และใช้สารสกัดสะเดาฉีดพ่น"
                    rec_color = "#ff9800"
                elif result_class == 'leaf spot':
                    rec_title = "Leaf Spot Disease"
                    rec_detail = "โรคใบจุดตากบ: เกิดจากเชื้อรา แนะนำให้ตัดแต่งใบที่เป็นโรคไปเผาทำลาย และฉีดพ่นสารป้องกันเชื้อรา"
                    rec_color = "#ff5722"
                elif result_class == 'whitefly':
                    rec_title = "Whitefly Infestation"
                    rec_detail = "พบแมลงหวี่ขาว: ภัยเงียบของสวนพริก แนะนำให้ใช้กับดักกาวเหนียวสีเหลือง หรือฉีดพ่นน้ำหมักสมุนไพรไล่แมลง"
                    rec_color = "#2196f3"
                elif result_class == 'yellow':
                    rec_title = "Yellow Leaf Disease"
                    rec_detail = "อาการใบเหลือง: อาจเกิดจากไวรัสหรือขาดสารอาหาร ตรวจสอบสภาพดินและใส่ปุ๋ยบำรุงด่วน"
                    rec_color = "#ffeb3b"

                st.markdown(f"""
                    <div style="background-color: #fafafa; border-radius: 16px; padding: 25px; margin-top: 30px; border: 1px solid #eee;">
                        <h4 style="margin-top: 0; color: {rec_color}; display: flex; align-items: center; gap: 10px;">
                            💡 Recommendation
                        </h4>
                        <p style="color: #333; margin-bottom: 5px; font-weight: 600; font-size: 1.1rem;">{rec_title}</p>
                        <p style="color: #666; font-weight: 300; margin: 0; font-size: 1rem;">{rec_detail}</p>
                    </div>
                """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 50px; color: #bbb; font-size: 0.8rem; font-weight: 300;">
    Computer Research Project • UBRU<br>
    Designed by WhiteCat Team
</div>
""", unsafe_allow_html=True)