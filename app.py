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
            padding: 2rem 2rem 4rem 2rem !important; 
            box-shadow: 0 15px 50px rgba(0,0,0,0.3) !important;
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

        /* ปรับแต่งปุ่มและ Radio */
        div[role="radiogroup"] label {
            background: linear-gradient(135deg, #fbc02d 0%, #f57f17 100%) !important;
            border: none !important;
            padding: 10px 20px !important;
            border-radius: 25px !important;
            color: #ffffff !important; 
        }
        div.stButton > button {
            background: linear-gradient(135deg, #fbc02d 0%, #f57f17 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 15px !important;
            width: 100% !important;
        }
        h1 { 
            text-align: center; color: #e65100 !important; 
            font-weight: 800 !important; font-size: 2.2rem !important;
            text-shadow: 2px 2px 0px #fff8e1;
        }
        
        .custom-home-btn {
            background: linear-gradient(135deg, #fbc02d 0%, #f57f17 100%);
            color: #ffffff !important;
            text-decoration: none;
            padding: 0.8rem 2rem;
            border-radius: 15px;
            display: inline-block;
            text-align: center;
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True)

local_css()

# --- 3. ฟังก์ชัน Database (ปรับปรุงตาม SQL ใหม่) ---
# ⚠️ ระบุ URL โฟลเดอร์ที่เก็บรูปบน Server ของคุณ ⚠️
# เช่น http://www.cedubru.com/uploads/ หรือ path local ถ้า Run เครื่องเดียวกับ Web Server
IMAGE_BASE_URL = "http://www.cedubru.com/uploads/cases/" 
# หมายเหตุ: ถ้าไม่ทราบ URL ที่แน่นอน รูปอาจจะไม่ขึ้น ต้องแก้ตรงนี้ให้ถูก

def init_connection():
    return mysql.connector.connect(
        host="www.cedubru.com",     
        user="cedubruc_corn_db_s",        
        password="bcbbDrypgCQXnSYu8Qrw",
        database="cedubruc_corn_db_s"  # แก้ให้ตรงกับชื่อ DB ในไฟล์ SQL
    )

def get_image_list(filter_mode):
    try:
        conn = init_connection()
        cursor = conn.cursor()
        
        # เชื่อม plant_cases กับ media_files
        base_query = """
            SELECT p.case_id, m.file_path, p.ai_prediction 
            FROM plant_cases p 
            JOIN media_files m ON p.case_id = m.case_id 
        """
        
        if "ยังไม่ตรวจ" in filter_mode:
            # status NEW หรือ ai_prediction เป็น NULL
            sql = base_query + "WHERE p.status = 'NEW' OR p.ai_prediction IS NULL ORDER BY p.case_id ASC"
        elif "ตรวจแล้ว" in filter_mode:
            sql = base_query + "WHERE p.status != 'NEW' ORDER BY p.case_id DESC"
        else:
            sql = base_query + "ORDER BY p.case_id DESC"
            
        cursor.execute(sql)
        data = cursor.fetchall()
        conn.close()
        return data
    except Exception as e:
        st.error(f"❌ DB Error (List): {e}")
        return []

def get_image_data(case_id):
    try:
        conn = init_connection()
        cursor = conn.cursor()
        # ดึง file_path และผลทำนาย
        sql = """
            SELECT m.file_path, p.ai_prediction, p.ai_confidence 
            FROM plant_cases p 
            JOIN media_files m ON p.case_id = m.case_id 
            WHERE p.case_id = %s LIMIT 1
        """
        cursor.execute(sql, (case_id,))
        data = cursor.fetchone()
        conn.close()
        return data 
    except Exception as e:
        st.error(f"❌ DB Error (Data): {e}")
        return None

def update_database(case_id, result, confidence):
    try:
        conn = init_connection()
        cursor = conn.cursor()
        # อัปเดตลง table plant_cases
        sql = """
            UPDATE plant_cases 
            SET ai_prediction = %s, ai_confidence = %s, status = 'AI_ANALYZED', diagnosed_at = NOW() 
            WHERE case_id = %s
        """
        cursor.execute(sql, (result, float(confidence), case_id))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"❌ DB Update Error: {e}")
        return False

def load_image_from_path(file_path):
    # ฟังก์ชันโหลดรูปจาก URL หรือ Path
    try:
        # กรณี 1: ถ้าเป็น URL
        full_url = urllib.parse.urljoin(IMAGE_BASE_URL, file_path)
        response = requests.get(full_url, timeout=5)
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
        else:
            # กรณี 2: ลองโหลด local เผื่อรันบนเครื่องที่มีไฟล์
            if os.path.exists(file_path):
                return Image.open(file_path)
            # กรณีหาไม่เจอ
            return None
    except Exception as e:
        # st.error(f"Load Image Error: {e}")
        return None

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
            except Exception as e:
                st.error(f"❌ Download Error: {e}")
                return None

    try:
        return tf.keras.models.load_model(filename)
    except Exception as e:
        st.error(f"❌ Model Error: {e}")
        return None

def import_and_predict(image_data, model):
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
    <p style='text-align: center; color: #555; margin-bottom: 30px;'>
        ระบบวินิจฉัยโรคใบข้าวโพดอัจฉริยะ (Linked DB)
    </p>
""", unsafe_allow_html=True)

# --- ตัวกรอง ---
c1, c2, c3 = st.columns([0.1, 3, 0.1])
with c2:
    filter_option = st.radio(
        "📂 ตัวกรองข้อมูล:", 
        ["ทั้งหมด (All)", "ตรวจแล้ว (Analyzed)", "ยังไม่ตรวจ (Pending)"], 
        index=2 # Default เป็นยังไม่ตรวจ
    )

image_list = get_image_list(filter_option)

if len(image_list) > 0:
    id_list = [row[0] for row in image_list] # list ของ case_id
    
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    if st.session_state.current_index >= len(id_list):
        st.session_state.current_index = 0

    current_case_id = id_list[st.session_state.current_index]
    
    # --- แสดงผล ---
    st.markdown("---")
    st.markdown(f"<div style='text-align: center; background: #fff8e1; padding: 10px; border-radius: 10px;'>📸 Case ID: {current_case_id} ({st.session_state.current_index + 1}/{len(id_list)})</div>", unsafe_allow_html=True)

    data_row = get_image_data(current_case_id)
    
    if data_row:
        file_path, saved_result, saved_conf = data_row
        
        # โหลดรูปจาก Path หรือ URL
        image = load_image_from_path(file_path)
        
        col_img, col_act = st.columns([1, 1])
        
        with col_img:
            if image:
                st.image(image, use_column_width=True, caption=file_path)
            else:
                st.warning(f"⚠️ ไม่พบรูปภาพ: {file_path}")
                st.caption(f"ตรวจสอบ URL: {IMAGE_BASE_URL}{file_path}")
        
        with col_act:
            st.markdown("### ผลการวิเคราะห์")
            
            if saved_result and saved_result != "รอประมวลผล...":
                bg = "#d4edda" if 'Healthy' in saved_result or 'ปกติ' in saved_result else "#f8d7da"
                text_col = "#155724" if 'Healthy' in saved_result or 'ปกติ' in saved_result else "#721c24"
                
                st.markdown(f"""
                    <div style="background-color: {bg}; padding: 20px; border-radius: 15px; border: 2px solid {text_col}; margin-bottom: 20px; text-align: center;">
                        <h2 style="color: {text_col} !important; margin: 0; font-size: 1.6rem;">{saved_result}</h2>
                        <p style="margin-top: 10px;">ความมั่นใจ: <strong>{saved_conf:.2f}%</strong></p>
                    </div>
                """, unsafe_allow_html=True)
                
                if st.button("🔄 วินิจฉัยซ้ำ"):
                    # รีเซ็ตค่าเพื่อวินิจฉัยใหม่
                    update_database(current_case_id, None, 0)
                    st.experimental_rerun()
            
            else:
                st.info("⚠️ เคสนี้ยังไม่ได้รับการประมวลผลโดย AI")
                if image and st.button("🚀 วินิจฉัยรูปนี้"):
                    if model:
                        with st.spinner("AI กำลังทำงาน..."):
                            preds = import_and_predict(image, model)
                            
                            # ⚠️ แก้ไข Class Name ตาม Model จริงของคุณ ⚠️
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
                            
                            update_database(current_case_id, final_res, conf)
                            st.success("บันทึกเรียบร้อย!")
                            time.sleep(0.5)
                            st.experimental_rerun()
                    else:
                        st.error("Model Error")

                # ปุ่ม Batch Analysis
                if "ยังไม่ตรวจ" in filter_option and image:
                     st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                     if st.button(f"⚡ Auto-Run ที่เหลือ ({len(image_list)} รูป)"):
                        if model:
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # วนลูปเฉพาะที่อยู่ใน List (ยังไม่ตรวจ)
                            for i, (c_id, f_path, _) in enumerate(image_list):
                                status_text.text(f"⏳ Processing Case {c_id}...")
                                
                                # โหลดรูปใหม่แต่ละรอบ
                                img_data = get_image_data(c_id)
                                if img_data:
                                    f_p, _, _ = img_data
                                    img_obj = load_image_from_path(f_p)
                                    
                                    if img_obj:
                                        preds = import_and_predict(img_obj, model)
                                        idx = np.argmax(preds)
                                        res_eng = class_names[idx]
                                        conf = np.max(preds) * 100
                                        final_res = th_dict.get(res_eng, res_eng)
                                        
                                        update_database(c_id, final_res, conf)
                                    else:
                                        # ถ้ารูปเสีย/หาไม่เจอ ข้ามไป
                                        pass

                                progress_bar.progress((i + 1) / len(image_list))
                            
                            status_text.text("✅ เสร็จสิ้นทั้งหมด!")
                            time.sleep(1)
                            st.experimental_rerun()

    # --- ปุ่มนำทาง ---
    st.markdown("<br>", unsafe_allow_html=True) 
    c_prev, c_empty, c_next = st.columns([1, 0.2, 1]) 
    
    with c_prev:
        if st.session_state.current_index > 0:
            if st.button("◀️ ย้อนกลับ"):
                st.session_state.current_index -= 1
                st.experimental_rerun()
            
    with c_next:
        if st.session_state.current_index < len(id_list) - 1:
            if st.button("ถัดไป ▶️"):
                st.session_state.current_index += 1
                st.experimental_rerun()

else:
    st.warning("ไม่พบข้อมูลตามตัวกรองที่เลือก")

# --- Footer Link ---
base_url = "http://www.cedubru.com/"
path = "ตรวจโรคใบข้าวโพด/" 
full_url = base_url + urllib.parse.quote(path)

st.markdown(f"""
    <div style="text-align: center; margin-top: 30px;">
        <a href="{full_url}" target="_blank" class="custom-home-btn">
            🏠 กลับสู่หน้าหลัก
        </a>
    </div>
""", unsafe_allow_html=True)