import streamlit as st
from PIL import Image
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objects as go

#preparo para funcao yfinance.download()---------------------------------------
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

#configurando pagina streamlit-------------------------------------------------
st.set_page_config(
    page_title="Alpha",
    page_icon="💹",
)

#acessando matriz da logo------------------------------------------------------
imageLogo = Image.open('logo.png')

#recebendo imput---------------------------------------------------------------
st.image(imageLogo)
st.write("""  
#### um programa GUI para auxiliar estratégias quantitativas em finanças.
""")
st.markdown("""---""")
st.markdown("""# Escolha uma ação""")
stocks = ("^BVSP","^DJI", "^GSPC","^CMC200", "NQ=F", "EURBRL=X", "TSLA", "AAPL", "GOOG", "MSFT", "GME", "NVDA", "FDX", "FRC", "CL=F", "BTC-USD","GS")
selected_stock = st.selectbox("Selecione uma ação para a análise", stocks)

#parametro necessario para funcao Prophet make_future_dataframe()--------------
n_years = st.slider("Anos gerados pela analise", 1 , 4)
period = n_years * 365

#salvando os estados das variaveis acima no cache-----------------------------
@st.cache_data 
#baixando dados do yahoo finance----------------------------------------------
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace = True)
    return data

#carregando dados------------------------------------------------------------
data_load_state = st.text("↻Carregando dados do Yahoo Finance")
data = load_data(selected_stock)#carregando dados do ticker escolhido
data_load_state.text("Dados carregados com sucesso ✔️")
st.markdown("""---""")
st.write("""
 # 1)Histórico
""")

st.subheader('Último dado carregado')
st.write(data.tail(1))

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Preço Abertura USD'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Preço Fechamento USD'))
    fig.layout.update(title_text = "Dados Yahoo! x Tempo", xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)

plot_raw_data()

#previsao Prophet--------------------------------------------------------------------------
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close":"y"})#para se adequar a funcao fit
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)# dataframe extendido ao futuro pelo numero de dias definido
forecast = m.predict(future)#https://facebook.github.io/prophet/docs/quick_start.html


st.markdown("""---""")
st.write("""
 # 2)Prophet
""")
st.write("""
 Entenda o [procedimento _Prophet_ para prever dados de séries temporais](https://facebook.github.io/prophet/)
""")
st.subheader('Últimos dados previstos pelo Prophet')
st.write(forecast.tail(3))  

st.subheader('Dados previstos pelo Prophet')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.subheader('Componentes técnicos da previsão')
fig2 = m.plot_components(forecast)
st.write(fig2)

st.write("""[[1]A piecewise linear or logistic growth curve trend. Prophet automatically detects changes in trends by selecting changepoints from the data.](https://research.facebook.com/blog/2017/2/prophet-forecasting-at-scale/)
\n[[2]A yearly seasonal component modeled using Fourier series.](https://research.facebook.com/blog/2017/2/prophet-forecasting-at-scale/)
\n[[3]A weekly seasonal component using dummy variables.](https://research.facebook.com/blog/2017/2/prophet-forecasting-at-scale/) """)