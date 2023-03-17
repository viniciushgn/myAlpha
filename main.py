import streamlit as st
from PIL import Image
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objects as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.set_page_config(
    page_title="Alpha",
    page_icon="💹",
)

imageLogo = Image.open('logo.png')

st.image(imageLogo)
st.write("""  
#### a GUI program for testing simple alpha strategies in finance.
""")

stocks = ("AAPL", "GOOG", "MSFT", "GME")
selected_stock = st.selectbox("Select stock for Prophet prediction", stocks)

n_years = st.slider("Years used for Prophet prediction:", 1 , 4)
period = n_years * 365

def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace = True)
    return data

data_load_state = st.text("↻Loading Yahoo stock data")
data = load_data(selected_stock)
data_load_state.text("Stock data ready ✔️")



st.write("""
 Learn more about the [_Prophet_ procedure for forecasting time series data](https://facebook.github.io/prophet/)
""")