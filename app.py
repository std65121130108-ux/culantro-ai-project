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
import requests

# --- ฟังก์ชันช่วย Rerun (แก้ปัญหา AttributeError) ---
def force_rerun():
    if hasattr(st, 'rerun'):
        st.rerun()
    else:
        st.experimental_rerun()

# --- [ส่วนสำคัญ 1] สร้างไฟล์ Config ---
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

# --- 2. CSS ตกแต่ง ---
def local_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;600;800&display=swap');
        
        html, body, [class*="css"], [data-testid="stAppViewContainer"] {
            font-family: 'Prompt', sans-serif !important;
            color: #333333 !important;
        }
        .stApp {
            background: linear-gradient(135deg, #a8ff78 0%, #78ffd6 100%) !important;
            background-attachment: fixed !important;
            background-size: cover !important;
        }
        div.block-container {
            background-color: rgba(255, 255, 255, 0.95) !important;
            border-radius: 30px !important;
            padding: 2rem !important;
            box-shadow: 0 15px 50px rgba(0,0,0,0.3) !important;
        }
        .app-header-icon {
            font-size: 80px !important;
            background: radial-gradient(circle, #fff176 0%, #fbc02d 100%) !important;
            width: 140px; height: 140px;
            border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            margin: 0 auto 15px auto;
            border: 5px solid #ffffff;
        }
        div[role="radiogroup"] label {
            background: linear-gradient(135deg, #fbc02d 0%, #f57f17 100%) !important;
            border: none; padding: 10px 20px; border-radius: 25px;
            color: white !important;
        }
        div.stButton > button {
            background: linear-gradient(135deg, #fbc02d 0%, #f57f17 100%) !important;
            color: #ffffff !important; border-radius: 15px; border: none;
        }
        h1 { text-align: center; color: #e65100 !important; font-weight: 800; }
        .custom-home-btn {
            background: linear-gradient(135deg, #fbc02d 0%, #f57f17 100%);
            color: #ffffff !important; padding: 0.8rem 2rem;
            border-radius: 15px; text-decoration: none; display: inline-block;
        }
    </style>
    """, unsafe_allow_html=True)

local_css()

# --- 3. ฟังก์ชัน Database ---
# URL รูปภาพ (จาก PHP ที่คุณอัปโหลด)
BASE_IMAGE_URL = "http://www.cedubru.com/corn/uploads/" 

def init_connection():
    return mysql.connector.connect(
        host="www.cedubru.com",
        user="cedubruc_corn_db_s",      
        password="bcbbDrypgCQXnSYu8Qrw", 
        database="cedubruc_corn_db_s"   
    )

def get_image_list(filter_mode):
    try:
        conn = init_connection()
        cursor = conn.cursor()
        base_sql = """
            SELECT p.case_id, m.file_path, p.ai_prediction 
            FROM plant_cases p
            JOIN media_files m ON p.case_id = m.case_id
            WHERE m.file_type = 'image'
        """
        if "ยังไม่ตรวจ" in filter_mode:
            sql = base_sql + " AND p.status = 'NEW' ORDER BY p.case_id ASC"
        elif "ตรวจแล้ว" in filter_mode:
            sql = base_sql + " AND p.status IN ('AI_ANALYZED', 'EXPERT_CONFIRMED') ORDER BY p.case_id DESC"
        else:
            sql = base_sql + " ORDER BY p.case_id DESC"
            
        cursor.execute(sql)
        data = cursor.fetchall()
        conn.close()
        return data
    except Exception as e:
        st.error(f"❌ DB Error: {e}")
        return []

def get_image_data(file_path, case_id):
    try:
        conn = init_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT ai_prediction, ai_confidence FROM plant_cases WHERE case_id = %s", (case_id,))
        result_data = cursor.fetchone()
        conn.close()
        
        saved_result = result_data[0] if result_data else None
        saved_conf = result_data[1] if result_data else 0

        img_url = BASE_IMAGE_URL + file_path
        try:
            response = requests.get(img_url, timeout=10)
            if response.status_code == 200:
                return response.content, saved_result, saved_conf
            else:
                st.warning(f"โหลดรูปไม่ได้ (HTTP {response.status_code}): {img_url}")
                return None
        except:
            return None
    except:
        return None

def update_database(case_id, result, confidence):
    try:
        conn = init_connection()
        cursor = conn.cursor()
        if result is None:
            sql = "UPDATE plant_cases SET ai_prediction=NULL, ai_confidence=0, status='NEW' WHERE case_id=%s"
            cursor.execute(sql, (case_id,))
        else:
            sql = "UPDATE plant_cases SET ai_prediction=%s, ai_confidence=%s, status = IF(status='EXPERT_CONFIRMED', status, 'AI_ANALYZED') WHERE case_id=%s"
            cursor.execute(sql, (result, float(confidence), case_id))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Update Error: {e}")
        return False

# --- 4. Load Model ---
if hasattr(st, 'cache_resource'): cache_decorator = st.cache_resource
else: cache_decorator = st.experimental_singleton

@cache_decorator
def load_model():
    filename = 'corn_model_full_v1.h5'
    file_id = '1Wp-evSKo2eajsNqAg3s1jAeRjeUhtgag'
    url = f'https://drive.google.com/uc?id={file_id}'

    if not os.path.exists(filename):
        with st.spinner("⏳ กำลังดาวน์โหลดโมเดล..."):
            try:
                gdown.download(url, filename, quiet=False)
            except: return None
    try:
        return tf.keras.models.load_model(filename)
    except: return None

def import_and_predict(image_data, model):
    size = (380, 380)
    try: image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    except AttributeError: image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img_array = np.asarray(image).astype(np.float32)
    data = np.ndarray(shape=(1, 380, 380, 3), dtype=np.float32)
    data[0] = img_array
    return model.predict(data)

# --- 5. Main UI ---
model = load_model()

st.markdown("""
    <div class='app-header-icon'>🌽</div>
    <h1>Corn Doctor AI</h1>
    <p style='text-align: center; color: #555;'>เชื่อมต่อฐานข้อมูล v2025</p>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns([0.1, 3, 0.1])
with c2:
    filter_option = st.radio("📂 ตัวกรองเคส:", ["ยังไม่ตรวจ (Pending)", "ตรวจแล้ว (Analyzed)", "ทั้งหมด (All)"], index=0)

image_list = get_image_list(filter_option)

if len(image_list) > 0:
    id_list = [row[0] for row in image_list]
    if 'current_index' not in st.session_state: st.session_state.current_index = 0
    if st.session_state.current_index >= len(id_list): st.session_state.current_index = 0

    current_idx = st.session_state.current_index
    current_case_id = image_list[current_idx][0]
    current_file_path = image_list[current_idx][1]
    
    st.markdown("---")
    st.markdown(f"<div style='text-align:center; background:#fff8e1; padding:10px; border-radius:10px;'>📸 รายการที่ {current_idx + 1} / {len(id_list)} (Case ID: {current_case_id})</div>", unsafe_allow_html=True)

    data_row = get_image_data(current_file_path, current_case_id)
    if data_row:
        blob_data, saved_result, saved_conf = data_row
        image = Image.open(io.BytesIO(blob_data))
        
        col_img, col_act = st.columns([1, 1])
        with col_img:
            st.image(image, use_column_width=True)
            st.caption(f"File: {current_file_path}")
        
        with col_act:
            st.markdown("### ผลการวิเคราะห์")
            if saved_result:
                bg = "#d4edda" if 'Healthy' in saved_result or 'ปกติ' in saved_result else "#f8d7da"
                text_col = "#155724" if 'Healthy' in saved_result or 'ปกติ' in saved_result else "#721c24"
                st.markdown(f"""
                    <div style="background-color: {bg}; padding: 20px; border-radius: 15px; border: 2px solid {text_col}; margin-bottom: 20px; text-align: center;">
                        <h2 style="color: {text_col} !important; margin: 0; font-size: 1.6rem;">{saved_result}</h2>
                        <p style="margin-top: 10px;">ความมั่นใจ: <strong>{saved_conf:.2f}%</strong></p>
                    </div>
                """, unsafe_allow_html=True)
                if st.button("🔄 วินิจฉัยซ้ำ"):
                    update_database(current_case_id, None, 0)
                    force_rerun() # <--- ใช้ฟังก์ชันใหม่ตรงนี้
            else:
                st.info("⚠️ เคสนี้ยังไม่ผ่าน AI")
                if st.button("🚀 วินิจฉัยรูปนี้"):
                    if model:
                        with st.spinner("AI กำลังทำงาน..."):
                            preds = import_and_predict(image, model)
                            class_names = ['Common_Rust', 'Gray_Leaf_Spot', 'Blight', 'Healthy']
                            idx = np.argmax(preds)
                            res_eng = class_names[idx]
                            conf = np.max(preds) * 100
                            th_dict = {'Common_Rust': 'โรคราสนิม (Common Rust)', 'Gray_Leaf_Spot': 'โรคใบจุดสีเทา (Gray Leaf Spot)', 'Blight': 'โรคใบไหม้แผลใหญ่ (Blight)', 'Healthy': 'ปกติ (Healthy)'}
                            final_res = th_dict.get(res_eng, res_eng)
                            update_database(current_case_id, final_res, conf)
                            st.success("บันทึกเรียบร้อย!")
                            time.sleep(0.5)
                            force_rerun() # <--- ใช้ฟังก์ชันใหม่ตรงนี้
                    else: st.error("Model Error")

                if "ยังไม่ตรวจ" in filter_option:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button(f"⚡ วิเคราะห์ทั้งหมด ({len(image_list)} เคส)"):
                        if model:
                            prog_bar = st.progress(0)
                            for i, (c_id, f_path, _) in enumerate(image_list):
                                d_row = get_image_data(f_path, c_id)
                                if d_row:
                                    img_b = d_row[0]
                                    img_pil = Image.open(io.BytesIO(img_b))
                                    p = import_and_predict(img_pil, model)
                                    class_names = ['Common_Rust', 'Gray_Leaf_Spot', 'Blight', 'Healthy']
                                    idx = np.argmax(p)
                                    res_eng = class_names[idx]
                                    conf = np.max(p) * 100
                                    th_dict = {'Common_Rust': 'โรคราสนิม (Common Rust)', 'Gray_Leaf_Spot': 'โรคใบจุดสีเทา (Gray Leaf Spot)', 'Blight': 'โรคใบไหม้แผลใหญ่ (Blight)', 'Healthy': 'ปกติ (Healthy)'}
                                    final_res = th_dict.get(res_eng, res_eng)
                                    update_database(c_id, final_res, conf)
                                prog_bar.progress((i + 1) / len(image_list))
                            st.success("เสร็จสิ้น!")
                            time.sleep(1)
                            force_rerun() # <--- ใช้ฟังก์ชันใหม่ตรงนี้

    st.markdown("<br>", unsafe_allow_html=True)
    c_prev, c_empty, c_next = st.columns([1, 0.2, 1])
    with c_prev:
        if st.button("◀️ ก่อนหน้า"):
            st.session_state.current_index = max(0, st.session_state.current_index - 1)
            force_rerun() # <--- ใช้ฟังก์ชันใหม่ตรงนี้
    with c_next:
        if st.button("ถัดไป ▶️"):
            st.session_state.current_index = min(len(id_list) - 1, st.session_state.current_index + 1)
            force_rerun() # <--- ใช้ฟังก์ชันใหม่ตรงนี้

else:
    st.warning("ไม่พบข้อมูลตามเงื่อนไข")

base_url = "http://www.cedubru.com/"
path = "corn/"
full_url = base_url + path
st.markdown(f"<div style='text-align:center; margin-top:30px;'><a href='{full_url}' target='_blank' class='custom-home-btn'>🏠 กลับหน้าหลัก</a></div>", unsafe_allow_html=True)