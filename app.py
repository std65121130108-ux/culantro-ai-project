import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import time
import os
import mysql.connector
import io
import gdown
import urllib.parse

# --- [ส่วนสำคัญ 1] สร้างไฟล์ Config บังคับ Light Mode ---
config_dir = ".streamlit"
config_path = os.path.join(config_dir, "config.toml")

if not os.path.exists(config_dir):
    os.makedirs(config_dir)

with open(config_path, "w") as f:
    f.write('[theme]\nbase="light"\nprimaryColor="#F9A825"\nbackgroundColor="#FFFFFF"\nsecondaryBackgroundColor="#FFF8E1"\ntextColor="#333333"\n')

# --- 1. ตั้งค่าหน้าเว็บ ---
st.set_page_config(
    page_title="Corn Doctor AI",
    page_icon="🌽",
    layout="centered"
)

# --- 2. CSS ตกแต่ง (ปรับธีมเป็น เหลือง-เขียว ข้าวโพด) ---
def local_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;600;800&display=swap');
        
        html, body, [class*="css"], [data-testid="stAppViewContainer"] {
            font-family: 'Prompt', sans-serif !important;
            color: #333333 !important;
            font-weight: 400 !important;
        }

        .stApp {
            background: linear-gradient(135deg, #a8ff78 0%, #78ffd6 100%) !important;
            background-attachment: fixed !important;
            background-size: cover !important;
        }

        header[data-testid="stHeader"] { background-color: transparent !important; }
        div[data-testid="stDecoration"] { display: none; }

        ul[data-testid="main-menu-list"] {
            background-color: #ffffff !important;
            border-radius: 10px !important;
            padding: 10px !important;
        }
        ul[data-testid="main-menu-list"] * {
            color: #333333 !important;
            background-color: #ffffff !important;
        }
        button[kind="header"] { color: #ffffff !important; }

        div.block-container {
            background-color: rgba(255, 255, 255, 0.95) !important;
            border-radius: 30px !important;
            padding: 2rem 2rem 4rem 2rem !important; 
            margin-top: 2rem !important;
            box-shadow: 0 15px 50px rgba(0,0,0,0.3) !important;
            min-height: auto !important;
        }

        .app-header-icon {
            font-size: 80px !important;
            background: radial-gradient(circle, #fff176 0%, #fbc02d 100%) !important;
            width: 140px !important;
            height: 140px !important;
            border-radius: 50% !important;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 15px auto !important;
            box-shadow: 0 10px 25px rgba(255, 193, 7, 0.4) !important;
            border: 5px solid #ffffff !important;
        }

        div[role="radiogroup"] {
            display: flex !important;
            flex-direction: row !important;
            gap: 10px !important;
            justify-content: center !important;
            flex-wrap: wrap !important;
        }
        div[role="radiogroup"] label {
            background: linear-gradient(135deg, #fbc02d 0%, #f57f17 100%) !important;
            border: none !important;
            padding: 10px 20px !important;
            border-radius: 25px !important;
            cursor: pointer !important;
            transition: all 0.2s !important;
            margin: 0 !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2) !important;
            color: #ffffff !important; 
        }
        div[role="radiogroup"] label p {
            color: #ffffff !important;
            font-weight: 400 !important;
            font-size: 1rem !important;
        }
        div[role="radiogroup"] label:hover {
            filter: brightness(1.1) !important;
            transform: translateY(-2px) !important;
        }
        .stRadio > label {
            color: #e65100 !important;
            font-weight: 800 !important;
            font-size: 1.3rem !important;
            margin-bottom: 15px !important;
            display: block;
            text-align: center;
        }

        /* ปุ่มกดปกติของ Streamlit */
        div.stButton > button {
            background: linear-gradient(135deg, #fbc02d 0%, #f57f17 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 15px !important;
            font-weight: 400 !important;
            transition: all 0.3s ease !important;
            width: 100% !important;
            padding: 0.8rem !important;
            font-size: 1rem !important;
            box-shadow: 0 4px 10px rgba(0,0,0,0.2) !important;
        }
        div.stButton > button:hover {
            filter: brightness(1.1) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 15px rgba(0,0,0,0.3) !important;
            color: #ffffff !important;
        }
        div[data-testid="column"] button {
             background: linear-gradient(135deg, #fbc02d 0%, #f57f17 100%) !important;
             color: #ffffff !important;
             border: none !important;
        }

        div[data-testid="stImage"] > img {
            border-radius: 20px;
            max-height: 350px;
            width: auto;
            max-width: 100%;
            margin: 0 auto;
            display: block;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .footer-credit {
            font-size: 0.8rem; color: #888; text-align: center; margin-top: 10px;
        }
        
        h1 { 
            text-align: center; color: #e65100 !important; 
            font-weight: 800 !important; font-size: 2.2rem !important;
            margin-bottom: 5px !important; text-shadow: 2px 2px 0px #fff8e1;
        }

        /* --- [CSS สำหรับปุ่มลิงก์ HTML ที่สร้างเอง] --- */
        .custom-home-btn {
            background: linear-gradient(135deg, #fbc02d 0%, #f57f17 100%);
            color: #ffffff !important;
            text-decoration: none;
            padding: 0.8rem 2rem;
            border-radius: 15px;
            font-weight: 400;
            font-family: 'Prompt', sans-serif;
            display: inline-block;
            box-shadow: 0 4px 10px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
            text-align: center;
            width: 100%;
            border: none;
        }
        .custom-home-btn:hover {
            filter: brightness(1.1);
            transform: translateY(-2px);
            box-shadow: 0 6px 15px rgba(0,0,0,0.3);
            color: #ffffff !important;
        }
    </style>
    """, unsafe_allow_html=True)

local_css()

# --- 3. ฟังก์ชัน Database ---
def init_connection():
    # ⚠️ อย่าลืมแก้ชื่อ Database ถ้าคุณสร้าง DB ใหม่สำหรับข้าวโพด ⚠️
    return mysql.connector.connect(
        host="www.cedubru.com",     
        user="cedubruc_corn_db",        
        password="zj2wRET8wxzCMQPKEb2N",
        database="cedubruc_corn_db" 
    )

def get_image_list(filter_mode):
    try:
        conn = init_connection()
        cursor = conn.cursor()
        # ⚠️ แก้ชื่อตารางเป็น corn_images ⚠️
        if "ยังไม่ตรวจ" in filter_mode:
            sql = "SELECT id, image_name, prediction_result FROM corn_images WHERE prediction_result IS NULL ORDER BY id ASC"
        elif "ตรวจแล้ว" in filter_mode:
            sql = "SELECT id, image_name, prediction_result FROM corn_images WHERE prediction_result IS NOT NULL ORDER BY id DESC"
        else:
            sql = "SELECT id, image_name, prediction_result FROM corn_images ORDER BY id DESC"
        cursor.execute(sql)
        data = cursor.fetchall()
        conn.close()
        return data
    except Exception as e:
        st.error(f"❌ DB Error: {e}")
        return []

def get_image_data(img_id):
    try:
        conn = init_connection()
        cursor = conn.cursor()
        # ⚠️ แก้ชื่อตารางเป็น corn_images ⚠️
        cursor.execute("SELECT image_data, prediction_result, confidence FROM corn_images WHERE id = %s", (img_id,))
        data = cursor.fetchone()
        conn.close()
        return data 
    except: return None

def update_database(img_id, result, confidence):
    try:
        conn = init_connection()
        cursor = conn.cursor()
        # ⚠️ แก้ชื่อตารางเป็น corn_images ⚠️
        sql = "UPDATE corn_images SET prediction_result = %s, confidence = %s WHERE id = %s"
        cursor.execute(sql, (result, float(confidence), img_id))
        conn.commit()
        conn.close()
        return True
    except: return False

# --- 4. Load Model (with gdown) ---
if hasattr(st, 'cache_resource'): cache_decorator = st.cache_resource
else: cache_decorator = st.experimental_singleton

@cache_decorator
def load_model():
    # ⚠️ ตั้งชื่อไฟล์โมเดลใหม่ ⚠️
    filename = 'corn_model_full_v1.h5'
    
    # -------------------------------------------------------------
    # ⚠️ [สำคัญมาก] ใส่ File ID โมเดลข้าวโพด จาก Google Drive ของคุณ ⚠️
    # -------------------------------------------------------------
    file_id = '1Wp-evSKo2eajsNqAg3s1jAeRjeUhtgag' 
    # ตัวอย่าง: file_id = '1A2b3C4d5E...'
    
    url = f'https://drive.google.com/uc?id={file_id}'

    # เช็คว่ามีไฟล์ไหม ถ้าไม่มีให้โหลด
    if not os.path.exists(filename):
        with st.spinner("⏳ กำลังดาวน์โหลดโมเดลข้าวโพด... (ครั้งแรกอาจนานหน่อย)"):
            try:
                gdown.download(url, filename, quiet=False)
            except Exception as e:
                st.error(f"❌ ดาวน์โหลดไม่สำเร็จ: {e}")
                return None

    # โหลดเข้า TensorFlow
    try:
        return tf.keras.models.load_model(filename)
    except Exception as e:
        st.error(f"❌ โมเดลเสียหาย: {e}")
        return None

def import_and_predict(image_data, model):
    # ⚠️ เช็คขนาดรูปที่โมเดลข้าวโพดต้องการ (ปกติ EfficientNet ใช้ 224, 300, 380)
    size = (380, 380) 
    try:
        image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    except AttributeError:
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img_array = np.asarray(image).astype(np.float32)
    data = np.ndarray(shape=(1, 380, 380, 3), dtype=np.float32)
    data[0] = img_array
    return model.predict(data)

# --- 5. Main UI ---
model = load_model()

st.markdown("""
    <div class='app-header-icon'>🌽</div>
    <h1>Corn Doctor AI</h1>
    <p style='text-align: center; color: #555; margin-bottom: 30px; font-size: 1.1rem;'>
        ระบบวินิจฉัยโรคใบข้าวโพดอัจฉริยะด้วย AI
    </p>
""", unsafe_allow_html=True)

# --- ตัวกรอง ---
c1, c2, c3 = st.columns([0.1, 3, 0.1])
with c2:
    filter_option = st.radio(
        "📂 ตัวกรองข้อมูล:", 
        ["ทั้งหมด (All)", "ตรวจแล้ว (Analyzed)", "ยังไม่ตรวจ (Pending)"], 
    )

# ดึงข้อมูลตามตัวกรอง
image_list = get_image_list(filter_option)

if len(image_list) > 0:
    id_list = [row[0] for row in image_list]
    
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    if st.session_state.current_index >= len(id_list):
        st.session_state.current_index = 0

    current_id = id_list[st.session_state.current_index]
    
    # --- แสดงผล ---
    st.markdown("---")
    st.markdown(f"<div style='text-align: center; color: #333; margin-bottom: 15px; font-weight: normal; font-size: 1.1rem; background: #fff8e1; padding: 10px; border-radius: 10px;'>📸 รูปที่ {st.session_state.current_index + 1} / {len(id_list)} (ID: {current_id})</div>", unsafe_allow_html=True)

    data_row = get_image_data(current_id)
    
    if data_row:
        blob_data, saved_result, saved_conf = data_row
        image = Image.open(io.BytesIO(blob_data))
        
        col_img, col_act = st.columns([1, 1])
        
        with col_img:
            st.image(image, use_column_width=True)
        
        with col_act:
            st.markdown("### ผลการวิเคราะห์")
            
            if saved_result:
                bg = "#d4edda" if 'Healthy' in saved_result or 'ปกติ' in saved_result else "#f8d7da"
                text_col = "#155724" if 'Healthy' in saved_result or 'ปกติ' in saved_result else "#721c24"
                
                st.markdown(f"""
                    <div style="background-color: {bg}; padding: 20px; border-radius: 15px; border: 2px solid {text_col}; margin-bottom: 20px; text-align: center;">
                        <h2 style="color: {text_col} !important; margin: 0; font-size: 1.6rem; font-weight: 400;">{saved_result}</h2>
                        <p style="margin-top: 10px; font-size: 1rem; color: #333;">ความมั่นใจ: <strong>{saved_conf:.2f}%</strong></p>
                    </div>
                """, unsafe_allow_html=True)
                
                if st.button("🔄 วินิจฉัยซ้ำ"):
                    update_database(current_id, None, 0)
                    st.experimental_rerun()
            
            else:
                st.info("⚠️ รูปนี้ยังไม่ได้รับการตรวจสอบ")
                # ปุ่มวิเคราะห์เดี่ยว
                if st.button("🚀 วินิจฉัยรูปนี้"):
                    if model:
                        with st.spinner("AI กำลังทำงาน..."):
                            preds = import_and_predict(image, model)
                            
                            # -------------------------------------------------------------
                            # ⚠️ [ส่วนสำคัญ 2] แก้ไข Class Name ให้ตรงกับ Model ข้าวโพด ⚠️
                            # -------------------------------------------------------------
                            # ตัวอย่าง 4 class มาตรฐาน (แก้ตามโมเดลจริงของคุณ)
                            class_names = ['Common_Rust', 'Gray_Leaf_Spot', 'Blight', 'Healthy']
                            
                            idx = np.argmax(preds)
                            res_eng = class_names[idx]
                            conf = np.max(preds) * 100
                            
                            # แปลเป็นไทย
                            th_dict = {
                                'Common_Rust': 'โรคราสนิม (Common Rust)',
                                'Gray_Leaf_Spot': 'โรคใบจุดสีเทา (Gray Leaf Spot)',
                                'Blight': 'โรคใบไหม้แผลใหญ่ (Blight)',
                                'Healthy': 'ปกติ (Healthy)'
                            }
                            final_res = th_dict.get(res_eng, res_eng)
                            
                            update_database(current_id, final_res, conf)
                            st.success("บันทึกเรียบร้อย!")
                            time.sleep(0.5)
                            st.experimental_rerun()
                    else:
                        st.error("โมเดลยังไม่โหลด")
                
                # --- [ปุ่ม Batch อยู่ด้านล่าง] ---
                if "ยังไม่ตรวจ" in filter_option:
                    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                    if st.button(f"⚡ วิเคราะห์ทั้งหมดที่เหลือ ({len(image_list)} รูป)"):
                        if model:
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            for i, (img_id, img_name, _) in enumerate(image_list):
                                status_text.text(f"⏳ กำลังวิเคราะห์... {i+1}/{len(image_list)}")
                                data_row = get_image_data(img_id)
                                if data_row:
                                    blob_data = data_row[0]
                                    image = Image.open(io.BytesIO(blob_data))
                                    preds = import_and_predict(image, model)
                                    
                                    # ⚠️ ใช้ Class Name เดียวกันกับข้างบน
                                    class_names = ['Common_Rust', 'Gray_Leaf_Spot', 'Blight', 'Healthy']
                                    
                                    idx = np.argmax(preds)
                                    res_eng = class_names[idx]
                                    conf = np.max(preds) * 100
                                    
                                    th_dict = {
                                        'Common_Rust': 'โรคราสนิม (Common Rust)',
                                        'Gray_Leaf_Spot': 'โรคใบจุดสีเทา (Gray Leaf Spot)',
                                        'Blight': 'โรคใบไหม้แผลใหญ่ (Blight)',
                                        'Healthy': 'ปกติ (Healthy)'
                                    }
                                    final_res = th_dict.get(res_eng, res_eng)
                                    update_database(img_id, final_res, conf)
                                progress_bar.progress((i + 1) / len(image_list))
                            
                            status_text.text("✅ เสร็จสิ้น!")
                            st.success("บันทึกข้อมูลเรียบร้อยแล้ว")
                            time.sleep(1)
                            st.experimental_rerun()

    # --- ปุ่มนำทาง ---
    st.markdown("<br>", unsafe_allow_html=True) 
    c_prev, c_empty, c_next = st.columns([1, 0.2, 1]) 
    
    with c_prev:
        is_first_image = st.session_state.current_index == 0
        if is_first_image:
            if st.button("⏮️ ไปรูปสุดท้าย"):
                st.session_state.current_index = len(id_list) - 1
                st.experimental_rerun()
        else:
            if st.button("◀️ ย้อนกลับ"):
                st.session_state.current_index -= 1
                st.experimental_rerun()
            
    with c_next:
        is_last_image = st.session_state.current_index >= len(id_list) - 1
        if is_last_image:
            if st.button("🔄 เริ่มต้นใหม่"):
                st.session_state.current_index = 0
                st.experimental_rerun()
        else:
            if st.button("ถัดไป ▶️"):
                st.session_state.current_index += 1
                st.experimental_rerun()

else:
    st.warning("ไม่พบข้อมูลในฐานข้อมูลตามตัวกรองที่เลือก")

# --- [ส่วนปุ่มลิงก์ HTML ที่เพิ่มใหม่] ---
# ⚠️ เปลี่ยนลิงก์ให้เป็น Path ของเว็บตรวจโรคข้าวโพดของคุณ ⚠️
base_url = "http://www.cedubru.com/ตรวจโรคข้าวโพด/"
path = "ตรวจโรคใบข้าวโพด/" # แก้ Path ให้ตรงกับโฟลเดอร์จริง
full_url = base_url + urllib.parse.quote(path)

st.markdown(f"""
    <div style="text-align: center; margin-top: 30px; margin-bottom: 20px;">
        <a href="{full_url}" target="_blank" class="custom-home-btn">
            🏠 คลิกเพื่อกลับสู่หน้าหลัก
        </a>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="footer-credit">
        <strong>ระบบวินิจฉัยโรคใบข้าวโพด V.1.0</strong>
    </div>
""", unsafe_allow_html=True)