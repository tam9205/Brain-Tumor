import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import os

# ==========================================
# 1. CẤU HÌNH TRANG WEB
# ==========================================
st.set_page_config(page_title="Hệ thống Phân tích U Não", page_icon="🧠", layout="wide")

# ==========================================
# 2. HỆ THỐNG XÁC THỰC (ĐĂNG NHẬP / ĐĂNG KÝ)
# ==========================================
# Khởi tạo cơ sở dữ liệu người dùng tạm thời trong bộ nhớ
if 'users' not in st.session_state:
    st.session_state['users'] = {'admin': '123456'} # Tài khoản mặc định
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'current_user' not in st.session_state:
    st.session_state['current_user'] = ''

def login():
    st.title("🔐 Đăng nhập Hệ thống")
    with st.form("login_form"):
        username = st.text_input("Tên đăng nhập")
        password = st.text_input("Mật khẩu", type="password")
        submit = st.form_submit_button("Đăng nhập")
        
        if submit:
            if username in st.session_state['users'] and st.session_state['users'][username] == password:
                st.session_state['logged_in'] = True
                st.session_state['current_user'] = username
                st.success("Đăng nhập thành công! Đang chuyển hướng...")
                st.rerun()
            else:
                st.error("Sai tên đăng nhập hoặc mật khẩu!")

def register():
    st.title("📝 Đăng ký Tài khoản")
    with st.form("register_form"):
        new_user = st.text_input("Tên đăng nhập mới")
        new_pass = st.text_input("Mật khẩu", type="password")
        confirm_pass = st.text_input("Xác nhận mật khẩu", type="password")
        submit = st.form_submit_button("Đăng ký")
        
        if submit:
            if new_user in st.session_state['users']:
                st.error("Tên đăng nhập đã tồn tại!")
            elif new_pass != confirm_pass:
                st.error("Mật khẩu xác nhận không khớp!")
            elif len(new_pass) < 6:
                st.error("Mật khẩu phải có ít nhất 6 ký tự!")
            else:
                st.session_state['users'][new_user] = new_pass
                st.success("Đăng ký thành công! Vui lòng chuyển sang tab Đăng nhập.")

# Hiển thị màn hình chờ nếu chưa đăng nhập
if not st.session_state['logged_in']:
    st.markdown("<h1 style='text-align: center; color: #2e66ff;'>🧠 NỀN TẢNG AI CHẨN ĐOÁN U NÃO</h1>", unsafe_allow_html=True)
    st.markdown("---")
    tab1, tab2 = st.tabs(["Đăng nhập", "Đăng ký"])
    with tab1: login()
    with tab2: register()
    st.stop() # Dừng chạy code bên dưới nếu chưa đăng nhập

# ==========================================
# 3. GIAO DIỆN CHÍNH (SAU KHI ĐĂNG NHẬP)
# ==========================================
# Load model AI
@st.cache_resource
def load_model():
    model = smp.Unet(encoder_name="efficientnet-b0", encoder_weights=None, in_channels=3, classes=1, activation='sigmoid')
    model.load_state_dict(torch.load("best_efficientunet.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Sidebar Điều hướng
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3003/3003264.png", width=100)
    st.markdown(f"### Xin chào, **{st.session_state['current_user']}**! 👋")
    st.markdown("---")
    menu = st.radio("📌 Tùy chọn chức năng:", ["📊 Tổng quan Dữ liệu (Dataset)", "🔍 AI Chẩn đoán Hình ảnh"])
    st.markdown("---")
    if st.button("🚪 Đăng xuất"):
        st.session_state['logged_in'] = False
        st.session_state['current_user'] = ''
        st.rerun()

# --- TRANG 1: TỔNG QUAN DỮ LIỆU ---
if menu == "📊 Tổng quan Dữ liệu (Dataset)":
    st.title("📊 Khám phá Bộ dữ liệu Nghiên cứu")
    st.write("Dự án được huấn luyện trên một bộ dữ liệu quy mô lớn, được dán nhãn thủ công cẩn thận để đảm bảo mô hình nhận diện chính xác các khối u có kích thước và hình dạng đa dạng.")
    
    # Hiển thị số liệu tổng quan (Dùng các con số thực tế của bạn)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tổng số ảnh chụp MRI", "8,413")
    col2.metric("Tập Huấn luyện (Train)", "70 %")
    col3.metric("Tập Xác thực (Valid)", "20 %")
    col4.metric("Tập Kiểm thử (Test)", "10 %")
    
    st.markdown("---")
    st.markdown("### 🖼️ Thư viện ảnh mẫu (Gallery)")
    st.write("Dưới đây là một số hình ảnh lát cắt MRI sọ não tiêu biểu có trong bộ dữ liệu:")
    
    # Hiển thị ảnh mẫu (Mô phỏng)
    # Tự động tạo thư mục ảnh mẫu nếu chưa có
    sample_dir = "sample_images"
    os.makedirs(sample_dir, exist_ok=True)
    
    sample_files = [f for f in os.listdir(sample_dir) if f.endswith(('.jpg', '.png'))]
    
    if len(sample_files) > 0:
        cols = st.columns(4)
        for i, file_name in enumerate(sample_files[:8]): # Hiển thị tối đa 8 ảnh
            with cols[i % 4]:
                img_path = os.path.join(sample_dir, file_name)
                st.image(Image.open(img_path), use_column_width=True, caption=file_name)
    else:
        st.info(f"💡 Hướng dẫn: Bạn hãy tạo một thư mục tên là '{sample_dir}' nằm cùng chỗ với file app.py, sau đó copy khoảng 4-8 bức ảnh MRI vào đó. Trang web sẽ tự động hiển thị chúng lên đây làm mẫu!")

# --- TRANG 2: AI CHẨN ĐOÁN ---
elif menu == "🔍 AI Chẩn đoán Hình ảnh":
    st.title("🔍 Trợ lý AI Phân Vùng Khối U")
    st.markdown("Hệ thống sử dụng kiến trúc **EfficientNet-B0 + U-Net** để tự động nhận diện và khoanh vùng ranh giới khối u trên ảnh MRI sọ não.")
    
    uploaded_file = st.file_uploader("📂 Tải lên một bức ảnh MRI từ máy của bạn...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        image_np = np.array(image)
        img_resized = cv2.resize(image_np, (256, 256))
        
        input_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        
        with st.spinner("🧠 AI đang quét và tính toán..."):
            with torch.no_grad():
                output = model(input_tensor)
                pred_mask = (output.squeeze().numpy() > 0.5).astype(np.uint8)
                
        overlay = img_resized.copy()
        overlay[pred_mask == 1] = [255, 0, 0] 
        blended = cv2.addWeighted(img_resized, 0.6, overlay, 0.4, 0)
        
        st.success("Hoàn thành! Xem kết quả chi tiết bên dưới.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(img_resized, caption="1. Ảnh MRI Gốc")
        with col2:
            st.image(pred_mask * 255, caption="2. Vùng U do AI định vị")
        with col3:
            st.image(blended, caption="3. Chồng ảnh ranh giới (Overlay)")