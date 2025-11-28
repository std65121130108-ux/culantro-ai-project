import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os

# -------------------------------------------------------
# 1) PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="Chili Doctor AI",
    page_icon="🌶️",
    layout="centered"
)

# -------------------------------------------------------
# 2) GLOBAL CSS — ชุดดีไซน์ให้เหมือนหน้า HTML
# -------------------------------------------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;600&display=swap" rel="stylesheet">

<style>
/* Global Font */
html, body, [class*="css"], [class*="st-"] {
    font-family: 'Prompt', sans-serif !important;
}

/* Background ไล่สีแดงชมพู */
.stApp {
    background: linear-gradient(135deg, #FF416C 0%, #FF4B2B 100%) !important;
    background-attachment: fixed !important;
}

/* Glass Card เหมือนหน้า HTML */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255, 255, 255, 0.95) !important;
    backdrop-filter: blur(15px) !important;
    -webkit-backdrop-filter: blur(15px) !important;
    border-radius: 24px !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    box-shadow: 0 10px 35px rgba(0,0,0,0.25) !important;
    padding: 40px 25px !important;
    max-width: 480px;
    margin: auto;
    animation: fadeUp 0.8s ease-out;
}

/* Animations */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(40px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse {
    0%   { transform: scale(1); box-shadow: 0 0 0 0 rgba(255,75,43,0.3); }
    70%  { transform: scale(1.05); box-shadow: 0 0 0 20px rgba(255,75,43,0); }
    100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(255,75,43,0); }
}

/* Title / Subtitle */
h1 {
    color: #333 !important;
    font-weight: 600 !important;
    margin-bottom: 0px !important;
}
.subtitle {
    color: #d32f2f !important;
    text-transform: uppercase;
    font-weight: 500;
    letter-spacing: 1px;
}

/* Floating Chili Icon */
.app-icon {
    width: 100px;
    height: 100px;
    background: linear-gradient(45deg, #ff9a9e 0%, #fad0c4 99%);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 55px;
    margin: 0 auto 20px;
    animation: pulse 2s infinite;
    box-shadow: 0 4px 15px rgba(255,75,43,0.3);
}

/* Upload zone */
[data-testid="stFileUploaderDropzone"] {
    background-color: white !important;
    border: 2px dashed #FF8A80 !important;
    border-radius: 20px !important;
    padding: 35px 20px !important;
    transition: 0.25s ease;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #d32f2f !important;
    background-color: #fff6f5 !important;
}

/* Modern Button */
div.stButton > button {
    background: linear-gradient(90deg, #FF416C 0%, #FF4B2B 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 15px 25px !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    box-shadow: 0 5px 15px rgba(255,75,43,0.3) !important;
    width: 100%;
    transition: 0.25s ease;
    margin-top: 10px;
}
div.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 7px 20px rgba(255,75,43,0.55) !important;
}

/* Hide streamlit default elements */
#MainMenu, header, footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------
# 3) LOAD MODEL
# -------------------------------------------------------
@st.cache_resource
def load_model():
    filename = "efficientnetb4_model.h5"
    if not os.path.exists(filename):
        return None
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


model = load_model()

# -------------------------------------------------------
# 4) UI — GLASS CARD STYLE
# -------------------------------------------------------

with st.container(border=True):

    st.markdown("""
        <div class="app-icon">🌶️</div>
        <div class="subtitle">AI Expert System</div>
        <h1 style="text-align:center;">Chili Doctor AI</h1>
        <p style="text-align:center; color:#666; margin-top:10px;">
            ระบบผู้เชี่ยวชาญวินิจฉัยโรคพริกด้วย Deep Learning (EfficientNetB4)
        </p>
    """, unsafe_allow_html=True)

    file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if file is not None:
        image = Image.open(file)

        st.image(image, use_container_width=True)

        size_kb = file.size / 1024
        st.markdown(
            f"<p style='text-align:center; font-size:0.85rem; color:#999;'>📎 {file.name} • {size_kb:.1f} KB</p>",
            unsafe_allow_html=True
        )

        if st.button("🚀 เริ่มต้นวินิจฉัย"):
            if model is None:
                st.error("ไม่พบไฟล์โมเดล")
            else:
                with st.spinner("AI กำลังวิเคราะห์..."):
                    predictions = import_and_predict(image, model)
                    class_names = ['Healthy', 'Leaf Curl', 'Leaf Spot', 'Whitefly', 'Yellow']
                    idx = np.argmax(predictions)
                    result_class = class_names[idx]
                    confidence = np.max(predictions) * 100

                st.markdown("<hr>", unsafe_allow_html=True)

                st.markdown(f"""
                    <h2 style='text-align:center; color:#d32f2f;'>{result_class.upper()}</h2>
                    <p style='text-align:center;'>ความมั่นใจ {confidence:.2f}%</p>
                """, unsafe_allow_html=True)

                # Recommendation mapping
                suggestions = {
                    "Healthy": ("🌿", "ต้นพริกแข็งแรงดี ดูแลต่อเนื่อง"),
                    "Leaf Curl": ("🍂", "พบโรคใบหงิก ใช้สารสกัดสะเดาหรือกำจัดวัชพืช"),
                    "Leaf Spot": ("🌑", "พบโรคใบจุด ตัดใบเสียและใช้สารป้องกันเชื้อรา"),
                    "Whitefly": ("🪰", "พบแมลงหวี่ขาว ใช้แผ่นกาวเหนียวหรือฉีดพ่นสมุนไพร"),
                    "Yellow": ("🟡", "ใบเหลืองจากขาดธาตุอาหาร ปรับปรุงดินและให้ปุ๋ย")
                }

                icon, text = suggestions[result_class]

                st.markdown(f"""
                    <div style="background:#fff8e9; padding:20px; border-radius:20px; margin-top:20px;">
                        <h4>{icon} คำแนะนำ</h4>
                        <p style="font-size:1rem; color:#444;">{text}</p>
                    </div>
                """, unsafe_allow_html=True)

# -------------------------------------------------------
# FOOTER
# -------------------------------------------------------
st.markdown("""
<div style='text-align:center; margin-top:35px; color:white; font-size:0.8rem; opacity:0.9;'>
    โครงงานวิจัยทางคอมพิวเตอร์ • UBRU<br>
    <span style='font-size:0.7rem; opacity:0.7;'>Developed by WhiteCat Team</span>
</div>
""", unsafe_allow_html=True)
