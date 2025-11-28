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

# --- 2. 🎨 CSS ตกแต่ง (Design: Solid White Card 100%) ---
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
    /* บังคับฟอนต์ Prompt */
    html, body, [class*="css"], [class*="st-"] {
        font-family: 'Prompt', sans-serif !important;
    }
    
    /* 1. Background: Gradient เต็มจอ */
    .stApp {
        background: linear-gradient(135deg, #FF416C 0%, #FF4B2B 100%) !important;
        background-attachment: fixed !important;
    }

    /* 2. Main White Card (แก้ให้เป็นสีขาวทึบ 100% ห้ามใส) */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important; /* สีขาวทึบ */
        background: #ffffff !important;       /* ย้ำว่าเป็นสีขาว */
        border-radius: 30px !important;
        border: none !important;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.15) !important;
        padding: 40px 30px !important;
        max-width: 550px;
        margin: auto;
        
        /* แก้ปัญหาพื้นหลังใส */
        opacity: 1 !important;
        backdrop-filter: none !important;
    }

    /* 3. Typography: ปรับสีตัวหนังสือในกรอบให้ชัดเจน */
    div[data-testid="stVerticalBlockBorderWrapper"] h1 {
        color: #FF4B2B !important; /* หัวข้อสีแดง */
        font-weight: 700 !important;
        font-size: 2.5rem !important;
        margin-bottom: 5px !important;
        text-align: center;
        text-shadow: none !important; /* เอาเงาออกเพื่อให้คมชัดบนพื้นขาว */
    }
    
    .subtitle {
        color: #666 !important;
        font-size: 1.1rem !important;
        font-weight: 400;
        margin-bottom: 20px;
        text-align: center;
    }
    
    .tech-badge {
        background: #ffebee;
        color: #c62828;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }

    /* 4. Upload Area */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #f8f9fa !important;
        border: 2px dashed #FF4B2B !important;
        border-radius: 20px !important;
        padding: 30px !important;
    }
    [data-testid="stFileUploaderDropzone"] div div::before {
        content: "Drag & Drop Image Here";
        color: #555;
        font-weight: 600;
        font-size: 1rem;
    }
    [data-testid="stFileUploaderDropzone"] small {
        color: #888 !important;
    }

    /* 5. Button */
    div.stButton > button {
        background: linear-gradient(90deg, #FF416C 0%, #FF4B2B 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 15px 30px !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        box-shadow: 0 10px 20px rgba(255, 75, 43, 0.3) !important;
        width: 100%;
        margin-top: 20px;
        transition: transform 0.2s;
    }
    div.stButton > button:hover {
        transform: scale(1.02);
    }
    div.stButton > button p {
        color: white !important;
    }

    /* Result Styling */
    .result-container {
        text-align: center;
        margin-top: 20px;
        padding-top: 20px;
        border-top: 1px solid #eee;
    }
    .result-title {
        color: #FF4B2B;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 10px 0;
    }
    .recommendation-box {
        background-color: #fff8e1;
        border-left: 6px solid #ffc107;
        padding: 20px;
        border-radius: 10px;
        text-align: left;
        margin-top: 20px;
        display: flex;
        align-items: start;
    }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 40px;
        color: rgba(255,255,255,0.8);
        font-size: 0.8rem;
    }

    #MainMenu, header, footer {visibility: hidden;}
    
</style>
""", unsafe_allow_html=True)

# --- 3. โหลดโมเดล ---
@st.cache_resource
def load_model():
    filename = 'efficientnetb4_model.h5'
    if not os.path.exists(filename):
        pass 
    try:
        return tf.keras.models.load_model(filename)
    except:
        return None

def import_and_predict(image_data, model):
    size = (300, 300)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image).astype(np.float32)
    data = np.ndarray(shape=(1, 300, 300, 3), dtype=np.float32)
    data[0] = img_array
    return model.predict(data)

# --- 4. ส่วนแสดงผล UI ---

model = load_model()

# สร้าง Container (กรอบขาว)
with st.container(border=True):
    
    # Header
    st.markdown("""
        <div style="text-align: center;">
            <div style="font-size: 4rem; margin-bottom: 5px;">🌶️</div>
            <h1>Chili Doctor AI</h1>
            <div class="subtitle">ระบบผู้เชี่ยวชาญตรวจวินิจฉัยโรคพริกอัจฉริยะ</div>
            <span class="tech-badge">Deep Learning (EfficientNetB4)</span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # พื้นที่อัปโหลด
    file = st.file_uploader("", type=["jpg", "png", "jpeg"])

    if file is not None:
        image = Image.open(file)
        
        # แสดงรูปภาพ
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 4, 1])
        with col2:
            st.image(image, use_container_width=True)
        
        # ปุ่มกด
        if st.button("🚀 วินิจฉัยโรค (Start Diagnosis)"):
            if model is None:
                st.error("⚠️ Model file not found.")
            else:
                with st.spinner('กำลังวิเคราะห์...'):
                    predictions = import_and_predict(image, model)
                    class_names = ['Healthy', 'Leaf Curl', 'Leaf Spot', 'Whitefly', 'Yellow']
                    class_index = np.argmax(predictions)
                    result_class = class_names[class_index]
                    confidence = np.max(predictions) * 100

                # --- ส่วนแสดงผลลัพธ์ (อยู่ในกรอบขาว) ---
                treatment_text = ""
                icon = ""
                box_color = "#f8f9fa"
                border_color = "#ccc"
                
                if result_class == 'Healthy':
                    treatment_text = "ต้นพริกแข็งแรงดีมาก! แนะนำให้ดูแลรดน้ำและใส่ปุ๋ยบำรุงตามปกติ"
                    icon = "🌿"
                    box_color = "#e8f5e9" # เขียวอ่อน
                    border_color = "#4caf50"
                elif result_class == 'Leaf Curl':
                    treatment_text = "โรคใบหงิก: ระวังแมลงพาหะ (เช่น แมลงหวี่ขาว) กำจัดวัชพืช และใช้สารสกัดสะเดาฉีดพ่น"
                    icon = "🍂"
                    box_color = "#fff3e0" # ส้มอ่อน
                    border_color = "#ff9800"
                elif result_class == 'Leaf Spot':
                    treatment_text = "โรคใบจุด: เกิดจากเชื้อรา ให้ตัดแต่งใบที่เป็นโรคเผาทำลาย และฉีดพ่นสารป้องกันกำจัดเชื้อรา"
                    icon = "🌑"
                    box_color = "#ffebee" # แดงอ่อน
                    border_color = "#f44336"
                elif result_class == 'Whitefly':
                    treatment_text = "แมลงหวี่ขาว: เป็นพาหะนำโรค ให้ใช้กับดักกาวเหนียวสีเหลือง หรือฉีดพ่นน้ำหมักสมุนไพรไล่แมลง"
                    icon = "🪰"
                    box_color = "#e3f2fd" # ฟ้าอ่อน
                    border_color = "#2196f3"
                elif result_class == 'Yellow':
                    treatment_text = "อาการใบเหลือง: อาจเกิดจากการขาดธาตุอาหาร ตรวจสอบสภาพดินและใส่ปุ๋ยบำรุง"
                    icon = "🟡"
                    box_color = "#fffde7" # เหลืองอ่อน
                    border_color = "#ffeb3b"

                st.markdown(f"""
                    <div class="result-container">
                        <div style="color: #888; font-size: 0.9rem;">ผลการวิเคราะห์</div>
                        <div class="result-title">{result_class.upper()}</div>
                        <span style="background: #FF4B2B; color: white; padding: 5px 15px; border-radius: 20px; font-weight: 600;">
                            ความแม่นยำ: {confidence:.2f}%
                        </span>
                        
                        <div style="background-color: {box_color}; border-left: 5px solid {border_color}; padding: 20px; border-radius: 10px; text-align: left; margin-top: 25px; display: flex; align-items: start;">
                            <div style="font-size: 2rem; margin-right: 15px;">{icon}</div>
                            <div>
                                <h4 style="margin: 0 0 5px 0; color: #333;">คำแนะนำ</h4>
                                <p style="color: #444; margin: 0; line-height: 1.5;">{treatment_text}</p>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

# Footer นอกกรอบขาว
st.markdown("""
    <div class="footer">
        Computer Research Project • UBRU<br>
        Designed by WhiteCat Team
    </div>
""", unsafe_allow_html=True)