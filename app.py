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

# --- 2. 🎨 CSS ตกแต่ง (Design: Premium Glassmorphism) ---
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;700&display=swap" rel="stylesheet">
<style>
    /* Global Font Settings */
    html, body, [class*="css"], [class*="st-"] {
        font-family: 'Prompt', sans-serif !important;
    }
    
    /* 1. Background: สีส้มแดงไล่เฉด สดใส */
    .stApp {
        background: linear-gradient(135deg, #FF416C 0%, #FF4B2B 100%) !important;
        background-attachment: fixed !important;
    }

    /* 2. Glass Card Container (หัวใจหลักของความสวย) */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.85) !important; /* โปร่งแสงนิดๆ */
        backdrop-filter: blur(20px) !important; /* เบลอฉากหลังให้เนียน */
        -webkit-backdrop-filter: blur(20px) !important;
        border-radius: 30px !important;
        border: 1px solid rgba(255, 255, 255, 0.6) !important; /* ขอบขาวจางๆ */
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2) !important; /* เงาฟุ้งลึก */
        padding: 40px 30px !important;
        max-width: 480px;
        margin: auto;
    }

    /* 3. Typography Cleanup */
    h1 {
        color: #d32f2f !important;
        font-weight: 700 !important;
        font-size: 2.2rem !important;
        margin-bottom: 5px !important;
        letter-spacing: -0.5px;
    }
    .subtitle {
        color: #555 !important;
        font-size: 1rem !important;
        font-weight: 400;
        margin-bottom: 15px;
    }
    .badge {
        background: #ffebee;
        color: #c62828;
        padding: 5px 15px;
        border-radius: 50px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        border: 1px solid rgba(198, 40, 40, 0.1);
    }

    /* 4. Upload Area - ให้ดูสะอาดตา */
    [data-testid="stFileUploaderDropzone"] {
        background-color: white !important;
        border: 2px dashed #FF8A80 !important; /* เส้นประสีพีช */
        border-radius: 20px !important;
        padding: 30px 20px !important;
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #d32f2f !important;
        background-color: #fffaf9 !important;
    }
    [data-testid="stFileUploaderDropzone"] div div::before {
        content: "Drag & Drop Image Here";
        color: #333;
        font-weight: 600;
        font-size: 1rem;
    }
    [data-testid="stFileUploaderDropzone"] small {
        color: #999 !important;
        margin-top: 5px;
    }

    /* 5. Modern Button */
    div.stButton > button {
        background: linear-gradient(90deg, #FF416C 0%, #FF4B2B 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 15px !important;
        padding: 15px 25px !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        box-shadow: 0 5px 15px rgba(255, 75, 43, 0.3) !important;
        width: 100%;
        transition: all 0.3s ease;
        margin-top: 15px;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(255, 75, 43, 0.5) !important;
    }
    div.stButton > button p {
        color: white !important;
    }

    /* 6. Animations */
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    .floating-icon {
        animation: float 3s ease-in-out infinite;
        font-size: 4.5rem;
        display: inline-block;
        filter: drop-shadow(0 10px 10px rgba(0,0,0,0.1));
    }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 40px;
        color: rgba(255,255,255,0.9);
        font-size: 0.8rem;
        font-weight: 300;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Utilities */
    .stImage > img {
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    #MainMenu, header, footer {visibility: hidden;}
    
</style>
""", unsafe_allow_html=True)

# --- 3. โหลดโมเดล ---
@st.cache_resource
def load_model():
    filename = 'efficientnetb4_model.h5'
    if not os.path.exists(filename):
        pass # Handle download here if needed
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

# เริ่มสร้าง Card Container
with st.container(border=True):
    
    # Header Section
    st.markdown("""
        <div style="text-align: center; padding-top: 10px;">
            <div class="floating-icon">🌶️</div>
            <h1>Chili Doctor AI</h1>
            <div class="subtitle">ระบบผู้เชี่ยวชาญตรวจวินิจฉัยโรคพริกอัจฉริยะ</div>
            <span class="badge">EfficiencyNetB4 Model</span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Upload Section
    file = st.file_uploader("", type=["jpg", "png", "jpeg"])
    
    if file is not None:
        image = Image.open(file)
        
        # Display Image (Centered & Styled)
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([0.5, 5, 0.5])
        with col2:
            st.image(image, use_container_width=True)
        
        # File Info
        size_kb = file.size / 1024
        st.markdown(f"""
            <div style="text-align: center; margin-top: 15px; font-size: 0.85rem; color: #888;">
                <span style="background: #f5f5f5; padding: 4px 10px; border-radius: 10px;">
                    📎 {file.name} • {size_kb:.1f} KB
                </span>
            </div>
        """, unsafe_allow_html=True)
            
        # Button
        if st.button("🔍 START DIAGNOSIS"):
            if model is None:
                st.error("⚠️ Model file not found.")
            else:
                with st.spinner('AI is analyzing...'):
                    predictions = import_and_predict(image, model)
                    class_names = ['Healthy', 'Leaf Curl', 'Leaf Spot', 'Whitefly', 'Yellow']
                    class_index = np.argmax(predictions)
                    result_class = class_names[class_index]
                    confidence = np.max(predictions) * 100

                st.markdown("<hr style='margin: 25px 0; border: 0; border-top: 1px dashed #ddd;'>", unsafe_allow_html=True)
                
                # Result Display (สวยงามขึ้น)
                st.markdown(f"""
                    <div style="text-align: center;">
                        <div style="color: #999; font-size: 0.85rem; letter-spacing: 1px; text-transform: uppercase;">DIAGNOSIS RESULT</div>
                        <h2 style="color: #d32f2f; margin: 10px 0; font-size: 2rem; font-weight: 700;">{result_class.upper()}</h2>
                        <div style="margin-top: 5px;">
                            <span style="background: #e8f5e9; color: #2e7d32; padding: 5px 15px; border-radius: 20px; font-weight: 600; font-size: 0.9rem;">
                                Confidence: {confidence:.2f}%
                            </span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                # Treatment Logic
                treatment_title = "Recommendation"
                treatment_text = ""
                bg_color = "#fff8e1" # สีเหลืองอ่อนพาสเทล
                icon = "💡"
                
                if result_class == 'Healthy':
                    treatment_text = "ต้นพริกแข็งแรงดีมาก! แนะนำให้ดูแลรดน้ำและใส่ปุ๋ยตามปกติต่อไป"
                    bg_color = "#e8f5e9" # เขียวพาสเทล
                    icon = "🌿"
                elif result_class == 'Leaf Curl':
                    treatment_text = "พบโรคใบหงิก แนะนำให้กำจัดวัชพืชรอบแปลง และใช้สารสกัดสะเดาฉีดพ่นเพื่อไล่แมลงพาหะ"
                    icon = "🍂"
                elif result_class == 'Leaf Spot':
                    treatment_text = "พบโรคใบจุด ตัดแต่งใบที่เป็นโรคไปเผาทำลายทันที และฉีดพ่นสารป้องกันเชื้อรา"
                    icon = "🌑"
                elif result_class == 'Whitefly':
                    treatment_text = "พบแมลงหวี่ขาว ใช้กับดักกาวเหนียวสีเหลือง หรือฉีดพ่นน้ำหมักสมุนไพร"
                    icon = "🪰"
                elif result_class == 'Yellow':
                    treatment_text = "พบอาการใบเหลือง อาจขาดธาตุอาหาร ตรวจสอบสภาพดินและเติมปุ๋ยบำรุง"
                    icon = "🟡"
                
                # Treatment Box (Modern Style)
                st.markdown(f"""
                    <div style="background-color: {bg_color}; padding: 25px; border-radius: 20px; margin-top: 25px; text-align: left; border-left: 5px solid rgba(0,0,0,0.1);">
                        <div style="font-weight: 600; color: #333; margin-bottom: 8px; font-size: 1.1rem;">
                            {icon} {treatment_title}
                        </div>
                        <div style="color: #444; font-size: 0.95rem; line-height: 1.6;">
                            {treatment_text}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        Computer Research Project • UBRU<br>
        <span style="opacity: 0.7; font-size: 0.7rem;">Designed by WhiteCat Team</span>
    </div>
""", unsafe_allow_html=True)