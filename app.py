import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import os
import datetime # Thư viện để lấy thời gian thực

# ==========================================
# 1. CẤU HÌNH TRANG WEB
# ==========================================
st.set_page_config(page_title="Hệ thống Phân tích U Não", page_icon="🧠", layout="wide")

# ==========================================
# 2. HỆ THỐNG XÁC THỰC VÀ KHỞI TẠO BỘ NHỚ
# ==========================================
if 'users' not in st.session_state:
    st.session_state['users'] = {'admin': '123456'}
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'current_user' not in st.session_state:
    st.session_state['current_user'] = ''
    
# 🌟 KHỞI TẠO BỘ NHỚ LỊCH SỬ CHẨN ĐOÁN
if 'history' not in st.session_state:
    st.session_state['history'] = []

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
            elif new_pass != confirm_pass or len(new_pass) < 6:
                st.error("Mật khẩu không hợp lệ (tối thiểu 6 ký tự và phải khớp nhau)!")
            else:
                st.session_state['users'][new_user] = new_pass
                st.success("Đăng ký thành công! Vui lòng chuyển sang tab Đăng nhập.")

if not st.session_state['logged_in']:
    st.markdown("<h1 style='text-align: center; color: #2e66ff;'>🧠 NỀN TẢNG AI CHẨN ĐOÁN U NÃO</h1>", unsafe_allow_html=True)
    st.markdown("---")
    tab1, tab2 = st.tabs(["Đăng nhập", "Đăng ký"])
    with tab1: login()
    with tab2: register()
    st.stop()

# ==========================================
# 3. GIAO DIỆN CHÍNH (SAU KHI ĐĂNG NHẬP)
# ==========================================
@st.cache_resource
def load_model():
    model = smp.Unet(encoder_name="efficientnet-b0", encoder_weights=None, in_channels=3, classes=1, activation='sigmoid')
    model.load_state_dict(torch.load("best_efficientunet.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# --- SIDEBAR: THANH ĐIỀU HƯỚNG ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3003/3003264.png", width=100)
    st.markdown(f"### Xin chào, **{st.session_state['current_user']}**! 👋")
    st.markdown("---")
    
    # 🌟 Đã thêm Menu Lịch sử vào đây
    menu = st.radio("📌 Tùy chọn chức năng:", [
        "📊 Tổng quan Dữ liệu", 
        "🔍 AI Chẩn đoán Hình ảnh", 
        "🕒 Lịch sử Phiên khám"
    ])
    
    st.markdown("---")
    if st.button("🚪 Đăng xuất"):
        st.session_state['logged_in'] = False
        st.session_state['current_user'] = ''
        st.session_state['history'] = [] # Xóa lịch sử khi đăng xuất
        st.rerun()

# --- TRANG 1: TỔNG QUAN DỮ LIỆU ---
if menu == "📊 Tổng quan Dữ liệu":
    st.title("📊 Khám phá Bộ dữ liệu Nghiên cứu")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tổng số ảnh", "8,413")
    col2.metric("Train Set", "70 %")
    col3.metric("Valid Set", "20 %")
    col4.metric("Test Set", "10 %")
    st.markdown("---")
    st.markdown("### 🖼️ Thư viện ảnh mẫu")
    sample_dir = "sample_images"
    os.makedirs(sample_dir, exist_ok=True)
    sample_files = [f for f in os.listdir(sample_dir) if f.endswith(('.jpg', '.png'))]
    if len(sample_files) > 0:
        cols = st.columns(4)
        for i, file_name in enumerate(sample_files[:8]):
            with cols[i % 4]:
                st.image(Image.open(os.path.join(sample_dir, file_name)), use_column_width=True)
    else:
        st.info("Chưa có ảnh mẫu trong thư mục 'sample_images'.")

# --- TRANG 2: AI CHẨN ĐOÁN ---
elif menu == "🔍 AI Chẩn đoán Hình ảnh":
    st.title("🔍 Trợ lý AI Phân Vùng Khối U")
    
    # 🌟 Thanh trượt điều chỉnh độ nhạy
    threshold = st.slider("⚙️ Điều chỉnh độ nhạy của AI (Ngưỡng Threshold)", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
    
    uploaded_file = st.file_uploader("📂 Tải lên một bức ảnh MRI từ máy của bạn...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        image_np = np.array(image)
        img_resized = cv2.resize(image_np, (256, 256))
        input_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        
        with st.spinner("🧠 AI đang quét và tính toán..."):
            with torch.no_grad():
                output = model(input_tensor)
                pred_mask = (output.squeeze().numpy() > threshold).astype(np.uint8)
                
        overlay = img_resized.copy()
        overlay[pred_mask == 1] = [255, 0, 0] 
        blended = cv2.addWeighted(img_resized, 0.6, overlay, 0.4, 0)
        
        # 🌟 Tính toán diện tích khối u
        tumor_pixels = np.sum(pred_mask == 1)
        pixel_to_mm2 = 1.5 # Hệ số quy đổi giả định
        estimated_area = tumor_pixels * pixel_to_mm2
        
        if tumor_pixels > 0:
            st.warning(f"⚠️ **Phát hiện dấu hiệu bất thường!** Ước tính diện tích mặt cắt: **{estimated_area:.2f} mm²**.")
        else:
            st.success("✅ Tuyệt vời! AI không phát hiện dấu hiệu khối u trên mặt cắt này.")
        
        col1, col2, col3 = st.columns(3)
        with col1: st.image(img_resized, caption="1. Ảnh MRI Gốc")
        with col2: st.image(pred_mask * 255, caption="2. Vùng U do AI định vị")
        with col3: st.image(blended, caption="3. Chồng ảnh ranh giới")
        
        # 🌟 Nút Tải xuống Kết quả
        is_success, buffer = cv2.imencode(".png", cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
        if is_success:
            st.download_button(label="📥 Tải ảnh Kết quả về máy", data=buffer.tobytes(), file_name=f"Result_{uploaded_file.name}", mime="image/png")
            
        # 🌟 LƯU VÀO LỊCH SỬ (Chỉ lưu khi người dùng bấm một nút xác nhận để tránh lưu rác)
        if st.button("💾 Lưu ca khám này vào Lịch sử"):
            record = {
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "filename": uploaded_file.name,
                "area": estimated_area,
                "status": "Có khối u" if tumor_pixels > 0 else "Bình thường",
                "image": blended # Lưu thẳng ảnh blended để hiển thị cho lẹ
            }
            st.session_state['history'].append(record)
            st.success("Đã lưu vào lịch sử thành công!")

# --- TRANG 3: LỊCH SỬ CHẨN ĐOÁN ---
elif menu == "🕒 Lịch sử Phiên khám":
    st.title("🕒 Lịch sử Chẩn đoán Bệnh nhân")
    st.write("Dữ liệu các ca quét MRI đã được AI phân tích trong phiên làm việc hiện tại.")
    
    if len(st.session_state['history']) == 0:
        st.info("Chưa có hồ sơ chẩn đoán nào được lưu.")
    else:
        # Lặp ngược danh sách để hiển thị ca mới nhất lên đầu
        for record in reversed(st.session_state['history']):
            # Dùng st.expander để tạo các hộp thoại có thể đóng/mở cho gọn gàng
            with st.expander(f"📁 Hồ sơ: {record['filename']} | ⏰ {record['time']}"):
                col_info, col_img = st.columns([1, 2])
                
                with col_info:
                    st.markdown("### Thông tin lâm sàng")
                    st.markdown(f"**Trạng thái:** {record['status']}")
                    if record['area'] > 0:
                        st.markdown(f"**Diện tích ước tính:** {record['area']:.2f} mm²")
                    
                with col_img:
                    st.image(record['image'], caption="Ảnh ranh giới khối u", width=300)
