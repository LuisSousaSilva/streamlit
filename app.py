#%%
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import portfolyoulab as pl
from datetime import datetime

st.set_page_config(layout="wide")
#%%
st.title('Stock charts')

@st.cache
def download_yahoo_data(tickers, normalize_quotes=False,
                      start='1970-01-01', end='2030-12-31'):
    quotes=pd.DataFrame()
    for ticker in tickers:
        df = yf.download(ticker, start=start, end=end, progress=False)
        df = df[['Adj Close']]
        df.columns=[ticker]
        quotes = pl.merge_time_series(quotes, df)
    
    quotes = quotes.dropna().ffill()
     
    if normalize_quotes:
        quotes = normalize(quotes)

    return quotes

# Using "with" notation
with st.sidebar:
    tickers_str = st.text_input(label='Tickers')
    start_date =  datetime.strptime('2010-10-10', '%Y-%m-%d')
    start_date = st.date_input('Start Date', value=start_date)
    end_date = st.date_input('End Date')
    title_1 = st.text_input(label='Title 1')
    title_2 = st.text_input(label='Title 2')
    title_3 = st.text_input(label='Title 3')
    title_4 = st.text_input(label='Title 4')
    

#%%
tickers = tickers_str.split(",")

if tickers_str != '':
    data_load_state = st.text('Loading data...')
    data = download_yahoo_data(tickers)
    data = pl.normalize(data[start_date:end_date])
    data_load_state.text("")
    plot_1 = pl.ichart(data, title=title_1, yticksuffix='€')
    plot_2 = pl.ichart(data, title=title_2)
    plot_3 = pl.ichart(data, title=title_3)
    plot_4 = pl.ichart(data, title=title_4)

    col1, col2 = st.columns(2)

    col1.header("A cat")
    col1.plotly_chart(plot_1, use_container_width=True)

    col2.header("Grayscale")
    col2.plotly_chart(plot_2, use_container_width=True)

    col3, col4 = st.columns(2)

    col3.header("A cat")
    col3.plotly_chart(plot_3, use_container_width=True)

    col4.header("Grayscale")
    col4.plotly_chart(plot_4, use_container_width=True)



#%%
