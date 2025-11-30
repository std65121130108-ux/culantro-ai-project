import streamlit as st
import test_gpu as tf
from PIL import Image, ImageOps
import numpy as np
import time
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

# --- 3. โหลดโมเดล (ปรับปรุงใหม่ ให้แจ้งเตือนชัดๆ) ---
@st.cache_resource
def load_model():
    filename = 'efficientnetb4_model.h5'
    
    # เช็คว่ามีไฟล์ไหม ถ้าไม่มีให้โหลด
    if not os.path.exists(filename):
        file_id = '1wQmgg_k45ymxx-sQJ33HdWVYlhVSqzuJ'
        url = f'https://drive.google.com/uc?id={file_id}'
        
        # 1. สร้าง Placeholder เพื่อจองพื้นที่แสดงข้อความ
        download_placeholder = st.empty()
        
        # 2. แสดงกล่องแจ้งเตือนขนาดใหญ่
        with download_placeholder.container():
            st.warning("""
                ⚠️ **กำลังดาวน์โหลดโมเดล AI (ครั้งแรกเท่านั้น)...**
                
                ไฟล์มีขนาดใหญ่ กรุณารอสักครู่ ระบบกำลังเตรียมความพร้อม...
            """)
            # แสดง Spinner หมุนๆ ให้รู้ว่าทำงานอยู่
            with st.spinner("🚀 กำลังดึงข้อมูลจาก Server... (ห้ามปิดหน้านี้)"):
                try:
                    import gdown
                    gdown.download(url, filename, quiet=False)
                    
                    if os.path.exists(filename):
                        # โหลดเสร็จ เปลี่ยนเป็นสีเขียวแจ้งเตือน
                        download_placeholder.success("✅ ดาวน์โหลดเสร็จสิ้น! พร้อมใช้งาน")
                        time.sleep(2) # โชว์ค้างไว้ 2 วินาที
                        download_placeholder.empty() # ลบกล่องทิ้งไปเลย เพื่อความสะอาด
                    else:
                        download_placeholder.error("❌ ดาวน์โหลดไม่สำเร็จ กรุณาลองใหม่")
                        return None
                except Exception as e:
                    download_placeholder.error(f"❌ เกิดข้อผิดพลาด: {e}")
                    return None
                    
    # โหลดโมเดลเข้า TensorFlow
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

# ⭐ 1. ตัวแปรสำหรับเปลี่ยน Key กล้อง (เพื่อรีเซ็ต)
if 'reset_count' not in st.session_state:
    st.session_state['reset_count'] = 0

# ⭐ 2. ตัวแปรสำหรับเก็บภาพถ่าย (เพื่อซ่อนกล้อง)
if 'cam_img_buffer' not in st.session_state:
    st.session_state['cam_img_buffer'] = None

# ส่วนหัว
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

# Tabs
tab_cam, tab_up = st.tabs(["📸 ถ่ายภาพใบพริก", "📂 อัปโหลดไฟล์รูป"])

img_file_buffer = None
camera_key = f"camera_{st.session_state['reset_count']}"
uploader_key = f"uploader_{st.session_state['reset_count']}"

# --- ส่วนกล้อง (แก้ไข Logic: ถ่ายแล้วซ่อน) ---
with tab_cam:
    # ถ้ายังไม่มีภาพในความจำ -> แสดงกล้อง
    if st.session_state['cam_img_buffer'] is None:
        camera_image = st.camera_input("กล้องถ่ายรูป", label_visibility="hidden", key=camera_key)
        
        # ข้อความแจ้งเตือน (แสดงเฉพาะตอนเปิดกล้อง)
        st.markdown("""
            <div style="text-align: center; margin-top: 20px;">
                <div style="
                    display: inline-block;
                    background: linear-gradient(90deg, #FF416C 0%, #FF4B2B 100%);
                    padding: 15px 30px;
                    border-radius: 50px;
                    box-shadow: 0 5px 15px rgba(255, 65, 108, 0.4);
                ">
                    <h4 style="
                        color: #ffffff !important; 
                        margin: 0 !important; 
                        padding: 0 !important;
                        font-weight: 600; 
                        font-size: 1.1rem;
                        -webkit-text-fill-color: #ffffff !important;
                    ">
                        📸 กดปุ่ม "Take Photo" ด้านบนเพื่อถ่ายรูป
                    </h4>
                </div>
            </div>
        """, unsafe_allow_html=True)

        if camera_image is not None:
            # ถ้ากดถ่ายปุ๊บ -> จำภาพไว้ -> สั่งโหลดหน้าใหม่เพื่อซ่อนกล้อง
            st.session_state['cam_img_buffer'] = camera_image
            st.rerun()
            
    else:
        # ถ้ามีภาพแล้ว -> ไม่ต้องโชว์กล้อง -> ส่งค่าภาพไปให้ตัวแปรหลัก
        img_file_buffer = st.session_state['cam_img_buffer']
        st.success("✅ บันทึกภาพเรียบร้อยแล้ว (กดปุ่ม 'ถ่ายรูปใหม่อีกครั้ง' หากต้องการถ่ายรูปภาพใหม่)")

# --- ส่วนอัปโหลด ---
with tab_up:
    uploaded_file = st.file_uploader("เลือกรูปภาพจากเครื่อง", type=["jpg", "png", "jpeg"], key=uploader_key)
    if uploaded_file is not None:
        img_file_buffer = uploaded_file

# 3. ส่วนแสดงผลและปุ่มกด
if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div style="border-radius: 15px; overflow: hidden; box-shadow: 0 5px 15px rgba(0,0,0,0.1); border: 3px solid rgba(255,255,255,0.8);">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)

    # ปุ่มกดคู่
    b1, b2 = st.columns(2, gap="medium")
    
    with b1:
        predict_click = st.button("🚀 วินิจฉัยโรคทันที", use_container_width=True)
        
    with b2:
        reset_click = st.button("🔄 ถ่ายรูปใหม่อีกครั้ง", use_container_width=True)

    # --- ⭐ แก้ไข Logic Reset: ต้องลบภาพในความจำด้วย ⭐ ---
    if reset_click:
        st.session_state['reset_count'] += 1
        st.session_state['cam_img_buffer'] = None # ล้างภาพที่จำไว้
        st.rerun()

    if predict_click:
        if model is None:
            st.error("❌ ไม่สามารถโหลดโมเดลได้")
        else:
            with st.spinner('AI กำลังประมวลผล...'):
                predictions = import_and_predict(image, model)
                
                # --- ⚠️ แก้ไขจุดที่ 1: อัปเดตรายชื่อ Class ให้ตรงกับโฟลเดอร์ (เรียงตามตัวอักษร A-Z) ---
                class_names = [
                    'Bacterial Spot', 
                    'Cercospora Leaf Spot', 
                    'Curl Virus', 
                    'Healthy Leaf', 
                    'Not leaf chilli', 
                    'Nutrition Deficiency', 
                    'White spot'
                ]
                
                class_index = np.argmax(predictions)
                result_class = class_names[class_index]
                confidence = np.max(predictions) * 100

            st.markdown("<div style='height: 1px; background-color: rgba(0,0,0,0.1); margin: 30px 0;'></div>", unsafe_allow_html=True)
            
            # แปลงชื่อแสดงผลเป็นภาษาไทยให้สวยงาม
            display_name = result_class
            if result_class == 'Bacterial Spot': display_name = "โรคจุดแบคทีเรีย (Bacterial Spot)"
            elif result_class == 'Cercospora Leaf Spot': display_name = "โรคใบจุดตากบ (Cercospora)"
            elif result_class == 'Curl Virus': display_name = "โรคใบหงิกไวรัส (Curl Virus)"
            elif result_class == 'Healthy Leaf': display_name = "ต้นพริกแข็งแรง (Healthy)"
            elif result_class == 'Not leaf chilli': display_name = "⚠️ ไม่ใช่รูปใบพริก"
            elif result_class == 'Nutrition Deficiency': display_name = "อาการขาดสารอาหาร (Deficiency)"
            elif result_class == 'White spot': display_name = "โรคจุดขาว (White Spot)"

            st.markdown(f"""
                <div style="text-align: center;">
                    <h3 style="color: #666; font-size: 1rem; margin-bottom: 5px;">ผลการวิเคราะห์</h3>
                    <h1 style="color: #FF4B2B !important; font-size: 2.2rem; margin: 0;">{display_name}</h1>
                    <div style="background: #fff0f0; color: #FF4B2B; display: inline-block; padding: 5px 15px; border-radius: 20px; font-weight: bold; font-size: 0.9rem; margin-top: 10px;">
                        ความมั่นใจ: {confidence:.2f}%
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # --- ⚠️ แก้ไขจุดที่ 2: ปรับคำแนะนำให้ตรงกับ 7 โรคใหม่ ---
            treatment_text = ""
            bg_color = "rgba(255, 248, 225, 0.8)"
            icon_bg = "#ffecb3"
            icon = "⚠️"
            
            if result_class == 'Healthy Leaf':
                treatment_text = "ยอดเยี่ยม! ต้นพริกของคุณแข็งแรงดี ไม่พบร่องรอยโรค หมั่นรดน้ำและใส่ปุ๋ยตามปกติเพื่อรักษาผลผลิต"
                bg_color = "rgba(232, 245, 233, 0.8)"
                icon_bg = "#c8e6c9"
                icon = "🌿"
                
            elif result_class == 'Bacterial Spot':
                treatment_text = "โรคจุดแบคทีเรีย: ระบาดได้ดีในหน้าฝน ให้เก็บใบที่เป็นโรคไปเผาทำลาย และฉีดพ่นสารประกอบทองแดง (Copper) หรือใช้เชื้อแบคทีเรียบาซิลลัส (BS) ในการควบคุม"
                icon = "🟤"
                
            elif result_class == 'Cercospora Leaf Spot':
                treatment_text = "โรคใบจุดตากบ (เชื้อรา): มักเกิดจุดกลมสีน้ำตาล ให้ตัดแต่งใบที่ระบาดออก เพื่อให้อากาศถ่ายเท และฉีดพ่นสารป้องกันกำจัดเชื้อรากลุ่มแมนโคเซบ หรือคาร์เบนดาซิม"
                icon = "🍂"
                
            elif result_class == 'Curl Virus':
                 treatment_text = "โรคใบหงิก (ไวรัส): เกิดจากแมลงพาหะ เช่น เพลี้ยไฟ/แมลงหวี่ขาว หากเป็นรุนแรงควรถอนทิ้งทันทีเพื่อป้องกันการลาม ป้องกันโดยการกำจัดแมลงพาหะอย่างสม่ำเสมอ"
                 icon = "🌀"
                 
            elif result_class == 'Nutrition Deficiency':
                 treatment_text = "อาการขาดสารอาหาร: ใบอาจมีสีเหลืองซีด หรือเส้นใบเขียวแต่เนื้อใบเหลือง ควรปรับปรุงดิน ตรวจวัดค่า pH และเติมปุ๋ยธาตุอาหารรอง/เสริม (เช่น แมกนีเซียม, เหล็ก, แคลเซียม)"
                 icon = "🟡"
            
            elif result_class == 'White spot':
                 treatment_text = "โรคจุดขาว: อาจเกิดจากเชื้อรา Alternaria หรือ Ramularia ให้หมั่นดูแลแปลงให้สะอาด ระบายอากาศให้ดี และใช้สารชีวภัณฑ์ไตรโคเดอร์มา หรือสารเคมีกลุ่ม azoxystrobin หากระบาดหนัก"
                 icon = "⚪"

            elif result_class == 'Not leaf chilli':
                 treatment_text = "ระบบตรวจจับว่าภาพนี้ **ไม่ใช่ใบพริก** หรือภาพไม่ชัดเจน กรุณาถ่ายภาพใบพริกใหม่อีกครั้ง เพื่อการวิเคราะห์ที่แม่นยำ"
                 bg_color = "rgba(255, 235, 238, 0.8)" 
                 icon_bg = "#ffcdd2"
                 icon = "❌"
            
            # แสดงกล่องคำแนะนำ
            st.markdown(f"""
                <div style="background-color: {bg_color}; padding: 25px; border-radius: 15px; margin-top: 25px; text-align: left; border: 1px solid rgba(0,0,0,0.05);">
                    <div style="display: flex; align-items: start;">
                        <div style="background: {icon_bg}; width: 45px; height: 45px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.4rem; margin-right: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); flex-shrink: 0;">
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
        <strong>วิจัยทางคอมพิวเตอร์  โดยสาขาวิชาคอมพิวเตอร์ศึกษา</strong> <br>
        <strong>คณะครุศาสตร์  มหาวิทยาลัยราชภัฏอุบลราชธานี</strong> <br>
        <span class="badge-custom">V.1.0 (Final Release)</span> <br>
        <div style="margin-top: 10px; font-size: 0.75rem; color: #aaa;">
            <strong>พัฒนาโดย: แมวใส่ชุดกบ และผองเพื่อน</strong>
        </div>
    </div>
""", unsafe_allow_html=True)