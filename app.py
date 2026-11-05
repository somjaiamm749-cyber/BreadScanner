import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="AI Bakery Tray Scanner", page_icon="🍞")
st.title("🍞 AI Bakery & Food Tray Scanner")
st.write("อัปโหลดรูปภาพถาดขนมปังเพื่อตรวจจับ นับจำนวน และคำนวณราคาอัตโนมัติ")

@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

price_dict = {
    "Bread with Pandan Custard": 35,
    "Shio pan": 45,
    "chocolate donute": 30,
    "croissant": 40,
    "glazed donut": 25,
    "honey butter toast": 50
}

uploaded_file = st.file_uploader("เลือกรูปภาพขนมปัง...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='รูปภาพที่อัปโหลด', use_container_width=True)

    if st.button("🔍 ตรวจจับและคำนวณราคา"):
        with st.spinner('กำลังประมวลผลด้วย YOLOv8...'):
            results = model.predict(image, conf=0.4)
            result = results[0]
            
            detected_list = [result.names[int(cls_id)] for cls_id in result.boxes.cls]
            counted_items = {item: detected_list.count(item) for item in set(detected_list)}

            res_plotted = result.plot()
            st.image(res_plotted[:, :, ::-1], caption='ผลการตรวจจับ AI', use_container_width=True)

            st.subheader("🧾 ใบเสร็จรับเงิน (AI Bakery & Food Tray Scanner)")
            total_price = 0
            
            if counted_items:
                for item_name, quantity in counted_items.items():
                    unit_price = price_dict.get(item_name, 0)
                    subtotal = quantity * unit_price
                    total_price += subtotal
                    st.write(f"- **{item_name}** : {quantity} ชิ้น x {unit_price} บาท = **{subtotal} บาท**")
                
                st.markdown("---")
                st.markdown(f"### 💰 ราคาสุทธิ (Total Price): {total_price} บาท")
            else:
                st.warning("❌ ไม่พบรายการขนมปังในภาพ")
