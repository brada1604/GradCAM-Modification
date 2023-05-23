import streamlit as st
from PIL import Image

st.title("Aplikasi Gambar Streamlit")
st.write("Ini adalah contoh aplikasi untuk menampilkan gambar menggunakan Streamlit")

# Upload gambar dari pengguna
uploaded_file = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
  # Baca gambar menggunakan PIL
  image = Image.open(uploaded_file)
  st.image(image, caption="Gambar yang diunggah", use_column_width=True)