import cv2
import numpy as np
import streamlit as st

# Fungsi untuk menghitung operator Sobel pada gambar
def sobel_operator(image, kernel_x, kernel_y):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.filter2D(gray, -1, kernel_x)
    sobel_y = cv2.filter2D(gray, -1, kernel_y)
    sobel = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
    sobel_rgb = cv2.cvtColor(sobel, cv2.COLOR_GRAY2RGB)
    return sobel_rgb

# Tampilan Streamlit
st.title("Deteksi Tepi dengan Operator Sobel")

# Menu sidebar untuk memilih gambar
uploaded_file = st.file_uploader("Pilih gambar")

# Tampilan utama
if uploaded_file is not None:
    # Baca gambar dari file yang diunggah
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
    
    # Tampilkan gambar asli
    st.image(image, caption="Gambar asli", use_column_width=True)
    
    # Hitung operator Sobel pada gambar asli
    sobel = sobel_operator(image, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]), np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]))
    
    # Tampilkan hasil operator Sobel
    st.image(sobel, caption="Hasil operator Sobel", use_column_width=True)
else:
    st.empty()