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
st.markdown("""---""")
st.markdown("""# 1)Prophet Prediction""")
stocks = ("AAPL", "GOOG", "MSFT", "GME")
selected_stock = st.selectbox("Select stock for Prophet prediction", stocks)

n_years = st.slider("Years of historical data used for Prophet prediction:", 1 , 4)
period = n_years * 365

@st.cache_data
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

st.subheader('Raw data')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Stock Open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Stock Close'))
    fig.layout.update(title_text = "Time Series Data", xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)

plot_raw_data()

df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close":"y"})
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forecast data')
st.write(forecast.tail())  

st.write('forecast data')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('forecast components')
fig2 = m.plot_components(forecast)
st.write(fig2)