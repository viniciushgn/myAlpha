import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="myAlpha",
    page_icon="💹",
)

imageLogo = Image.open('logo.png')

st.image(imageLogo)
st.write("""  
#### a GUI program for testing simple alpha strategies in finance.
""")