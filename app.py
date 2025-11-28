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
<link href="https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600&display=swap" rel="stylesheet">

<style>

    * {
        font-family: 'Prompt', sans-serif !important;
    }

    /* Background */
    .stApp {
        background: #f6f6f6 !important;
        display: flex;
        justify-content: center;
        padding-top: 40px;
        padding-bottom: 40px;
    }

    /* Main card */
    .main-card {
        background: rgba(255,255,255,0.75);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 22px;
        padding: 40px 35px;
        width: 420px;
        margin: auto;
        box-shadow: 0 10px 28px rgba(0,0,0,0.08);
        animation: fadeIn 0.9s ease;
    }

    /* Icon */
    .icon-circle {
        width: 75px;
        height: 75px;
        border-radius: 50%;
        background: #111;
        color: white;
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 34px;
        margin: 0 auto 20px;
        font-weight: 500;
    }

    /* Title */
    h1 {
        text-align: center;
        margin-bottom: 0;
        font-weight: 600;
        color: #111;
        letter-spacing: -0.5px;
        font-size: 1.9rem;
    }

    .subtitle {
        text-align: center;
        color: #777;
        font-size: 0.85rem;
        margin-top: 3px;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    /* Upload box */
    [data-testid="stFileUploaderDropzone"] {
        background: #ffffff;
        border-radius: 14px;
        border: 1px dashed #d0d0d0 !important;
        padding: 25px;
    }

    /* Button */
    div.stButton > button {
        background: #111 !important;
        color: white !important;
        border-radius: 40px;
        padding: 12px 0;
        width: 100%;
        font-size: 1rem;
        font-weight: 500;
        border: none;
        transition: 0.25s ease;
    }

    div.stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 18px rgba(0,0,0,0.2);
    }

    /* Fade animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(25px); }
        to   { opacity: 1; transform: translateY(0); }
    }

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
