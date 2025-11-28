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

# --- 2. 🎨 CSS ตกแต่ง (ฉบับแก้ไข: การ์ดสีขาว + ตัวหนังสือชัด) ---
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
    /* บังคับฟอนต์ Prompt */
    html, body, [class*="css"], [class*="st-"] {
        font-family: 'Prompt', sans-serif !important;
    }
    
    /* 1. พื้นหลังหลัก (Gradient สีส้มแดง) */
    .stApp {
        background: linear-gradient(135deg, #FF416C 0%, #FF4B2B 100%) !important;
        background-attachment: fixed !important;
    }

    /* 2. บังคับการ์ดให้เป็นสีขาวขุ่นและมีเงาชัดเจน */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: rgba(255, 255, 255, 0.95) !important; /* สีขาว 95% */
        backdrop-filter: blur(12px) !important;
        border-radius: 24px !important;
        border: 1px solid rgba(255, 255, 255, 0.8) !important;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2) !important;
        padding: 30px !important;
        max-width: 500px;
        margin: auto;
    }

    /* 3. บังคับตัวหนังสือข้างในกรอบให้เป็นสีเข้ม (แก้ปัญหาตัวหนังสือขาวกลืนพื้นหลัง) */
    div[data-testid="stVerticalBlockBorderWrapper"] h1,
    div[data-testid="stVerticalBlockBorderWrapper"] h2,
    div[data-testid="stVerticalBlockBorderWrapper"] h3,
    div[data-testid="stVerticalBlockBorderWrapper"] p,
    div[data-testid="stVerticalBlockBorderWrapper"] div,
    div[data-testid="stVerticalBlockBorderWrapper"] span,
    div[data-testid="stVerticalBlockBorderWrapper"] label,
    div[data-testid="stVerticalBlockBorderWrapper"] small {
        color: #333333 !important;
    }

    /* หัวข้อใหญ่สีแดง */
    div[data-testid="stVerticalBlockBorderWrapper"] h1 {
        color: #FF4B2B !important;
        text-shadow: none !important;
        font-size: 2rem !important;
        margin-bottom: 5px !important;
    }
    
    /* Subtitle */
    .subtitle {
        color: #666 !important;
        font-size: 1rem !important;
        text-align: center;
        margin-bottom: 10px;
    }

    /* 4. ปรับช่อง Upload ให้พื้นหลังเทาอ่อน ตัดกับสีขาว */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #f1f3f4 !important;
        border: 2px dashed #FF4B2B !important;
        border-radius: 16px !important;
        padding: 20px !important;
    }
    [data-testid="stFileUploaderDropzone"] div div::before {
        color: #555 !important;
        content: "Drag and drop file here";
        font-weight: 600;
    }
    [data-testid="stFileUploaderDropzone"] small {
        color: #888 !important;
    }

    /* 5. ปุ่มกด */
    div.stButton > button {
        background: linear-gradient(90deg, #FF416C 0%, #FF4B2B 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        box-shadow: 0 5px 15px rgba(255, 65, 108, 0.4) !important;
        font-weight: 600 !important;
        transition: transform 0.2s;
        margin-top: 10px;
    }
    div.stButton > button:hover {
        transform: scale(1.03);
    }
    /* บังคับตัวหนังสือในปุ่มเป็นสีขาวเสมอ */
    div.stButton > button p {
        color: white !important;
    }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 30px;
        color: rgba(255,255,255,0.8);
        font-size: 0.8rem;
    }
    
    /* ซ่อน Header/Footer เดิม */
    #MainMenu, header, footer {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

# --- 3. โหลดโมเดล (ใช้ Cache เพื่อความเร็ว) ---
@st.cache_resource
def load_model():
    filename = 'efficientnetb4_model.h5'
    # หากไม่มีไฟล์ ให้ข้ามไปก่อนเพื่อป้องกัน Error หน้าเว็บ
    if not os.path.exists(filename):
        # ใส่โค้ด gdown ของคุณที่นี่ถ้าจำเป็น
        pass 
        
    try:
        return tf.keras.models.load_model(filename)
    except:
        return None

def import_and_predict(image_data, model):
    size = (300, 300) # ปรับขนาดตามที่โมเดลต้องการ
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image).astype(np.float32)
    data = np.ndarray(shape=(1, 300, 300, 3), dtype=np.float32)
    data[0] = img_array
    return model.predict(data)

# --- 4. ส่วนแสดงผล (UI) ---

model = load_model()

# ใช้ Container แบบมีขอบ (CSS จะทำงานที่ตัวนี้)
with st.container(border=True):
    
    # Header
    st.markdown("""
        <div style="text-align: center;">
            <div style="font-size: 4rem; margin-bottom: 5px;">🌶️</div>
            <h1>Chili Doctor AI</h1>
            <div class="subtitle">ระบบผู้เชี่ยวชาญตรวจวินิจฉัยโรคพริกอัจฉริยะ</div>
            <span style="background: #ffebee; color: #c62828 !important; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; display: inline-block; margin-bottom: 20px;">
                Deep Learning Technology (EfficientNetB4)
            </span>
        </div>
    """, unsafe_allow_html=True)

    # พื้นที่อัปโหลด
    file = st.file_uploader("", type=["jpg", "png", "jpeg"])
    
    if file is not None:
        image = Image.open(file)
        
        # แสดงรูปภาพ
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 4, 1])
        with col2:
            st.image(image, use_container_width=True)
        
        # รายละเอียดไฟล์
        size_kb = file.size / 1024
        st.markdown(f"""
            <div style="text-align: center; margin-top: 10px; font-size: 0.85rem; color: #666 !important;">
                📄 {file.name} ({size_kb:.1f} KB)
            </div>
        """, unsafe_allow_html=True)
            
        # ปุ่ม Analyze
        if st.button("🔍 Analyze Image"):
            if model is None:
                st.error("⚠️ ไม่พบไฟล์โมเดล (efficientnetb4_model.h5)")
                st.info("กรุณาตรวจสอบว่าไฟล์โมเดลอยู่ในโฟลเดอร์เดียวกับโค้ด")
            else:
                with st.spinner('กำลังวิเคราะห์...'):
                    predictions = import_and_predict(image, model)
                    class_names = ['healthy', 'leaf curl', 'leaf spot', 'whitefly', 'yellow']
                    class_index = np.argmax(predictions)
                    result_class = class_names[class_index]
                    confidence = np.max(predictions) * 100

                st.markdown("<hr style='margin: 20px 0; border-top: 1px solid #eee;'>", unsafe_allow_html=True)
                
                # แสดงผลลัพธ์
                st.markdown(f"""
                    <div style="text-align: center;">
                        <div style="color: #888 !important; font-size: 0.9rem;">ผลการวินิจฉัย</div>
                        <h2 style="color: #d32f2f !important; margin: 5px 0; font-size: 1.8rem;">{result_class.upper()}</h2>
                        <div style="background: #f1f1f1; padding: 5px 15px; border-radius: 15px; display: inline-block; font-size: 0.85rem; color: #555 !important;">
                            ความมั่นใจ: {confidence:.2f}%
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                # Logic คำแนะนำ (Treatment)
                treatment_text = ""
                bg_color = "#fff3cd"
                text_color = "#856404"
                
                if result_class == 'healthy':
                    treatment_text = "🌿 <b>ต้นพริกแข็งแรงดี!</b> ไม่พบร่องรอยของโรค หมั่นดูแลรดน้ำตามปกติ"
                    bg_color = "#d4edda"
                    text_color = "#155724"
                elif result_class == 'leaf curl':
                    treatment_text = "🍂 <b>โรคใบหงิก:</b> ระวังแมลงพาหะ ให้กำจัดวัชพืชรอบแปลงและใช้น้ำหมักชีวภาพหรือสารสกัดสะเดา"
                elif result_class == 'leaf spot':
                    treatment_text = "🌑 <b>โรคใบจุดตากบ:</b> เกิดจากเชื้อรา ให้ตัดแต่งใบที่เป็นโรคไปเผาทำลาย และฉีดพ่นสารป้องกันกำจัดเชื้อรา"
                elif result_class == 'whitefly':
                    treatment_text = "🪰 <b>แมลงหวี่ขาว:</b> เป็นพาหะนำโรค ให้ใช้กับดักกาวเหนียวสีเหลือง หรือฉีดพ่นน้ำหมักสมุนไพรไล่แมลง"
                elif result_class == 'yellow':
                    treatment_text = "🟡 <b>อาการใบเหลือง:</b> อาจขาดธาตุอาหาร ให้ตรวจสอบสภาพดิน ปรับปรุงดิน และใส่ปุ๋ยบำรุงให้เหมาะสม"
                
                # แสดงกล่องคำแนะนำ (ใส่ !important ใน inline style เพื่อกัน CSS หลักทับสี)
                st.markdown(f"""
                    <div style="background-color: {bg_color}; color: {text_color} !important; padding: 20px; border-radius: 16px; margin-top: 20px; font-size: 0.95rem; text-align: left; line-height: 1.5; border: 1px solid rgba(0,0,0,0.05);">
                        {treatment_text}
                    </div>
                """, unsafe_allow_html=True)

# Footer นอกการ์ด
st.markdown("""
    <div class="footer">
        Computer Research Project • UBRU<br>
        Designed by WhiteCat Team
    </div>
""", unsafe_allow_html=True)