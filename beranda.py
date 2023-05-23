import streamlit as st
from PIL import Image
def main_page():
    # Original_Image = Image.open("img/kopma logo.png")
    # img = Original_Image
    st.title("PENGELOLAAN CITRA DIGITAL")
    st.title("Tugas Besar")
    
    st.header("Profile")
    with st.container():
        col1, col2 = st.columns([9,3])
        with col1:
            st.title('KELOMPOK MADEV 2.0')
            st.write('BAGUS NUGROHO | 211511034')
            st.write('MUHAMMAD RIVAN RIVALDI | 211511048')
            st.write('REYNA NUR RAHMAH SETIANA | 211511054')
            st.write('SALMAN ALFARISI | 211511059')
            st.write('PROGRAM STUDI', 'D3 TEKNIK INFORMATIKA')
            st.write('KELAS', '2B')
        # with col2:
        #     st.image(img)
if __name__ == '__main__':
    main_page()