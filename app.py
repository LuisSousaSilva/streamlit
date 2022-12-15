#%%
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import portfolyoulab as pl
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

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
    start_date =  datetime.strptime('1970-10-10', '%Y-%m-%d')
    start_date = st.date_input('Start Date', value=start_date)
    end_date = st.date_input('End Date')
    title_1 = st.text_input(label='Title 1')
    title_2 = st.text_input(label='Title 2')

    st.markdown('**Tamanho do heatmap de correlações**')
    width = st.sidebar.slider("Largura", 3., 10.0, 5.50)
    height = st.sidebar.slider("Altura", 0.5, 10.0, 4.00)

    eur_usd = st.radio(
    "EUR or USD?",
    ('EUR', 'USD'))
    # title_3 = st.text_input(label='Title 3')
    # title_4 = st.text_input(label='Title 4')


    

#%%
tickers = tickers_str.split(",")

if tickers_str != '':
    data_load_state = st.text('Loading data...')
    data = download_yahoo_data(tickers)
    data = pl.normalize(data[start_date:end_date])
    data_load_state.text("")

    # Correlation Matrix
    Corr_matrix = data.pct_change().corr()

    # Plotting the correlation matrix
    fig, ax = plt.subplots(figsize=(width, height))

    sns.heatmap(Corr_matrix, annot = True, cmap = "coolwarm", linewidths=.2, vmin = -1)
    plt.yticks(rotation=360)
    plt.title('Matrix de correlação')

    if eur_usd == 'EUR':
        plot_1 = pl.ichart(data, title=title_1, yticksuffix='€', yTitle='Valorização por cada 100 euros investidos')
    else:
        plot_1 = pl.ichart(data, title=title_1, yticksuffix='$', yTitle='Valorização por cada 100 dólares investidos')
    
    plot_2 = pl.ichart(pl.compute_drawdowns(data), title=title_2, yticksuffix='%')
    # plot_3 = pl.ichart(pl.compute_drawdowns(data), title=title_3, colors=['orange'])
    # plot_4 = pl.ichart(data, title=title_4)

    returns_table = pl.compute_performance_table(data)

    st.plotly_chart(plot_1, use_container_width=False)

    st.plotly_chart(plot_2, use_container_width=False)

    st.dataframe(returns_table)

    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.image(buf)

    # col1, col2 = st.columns(2)

    # col1.header("A cat")
    # col1.plotly_chart(plot_1, use_container_width=True)

    # col3, col4 = st.columns(2)

    # col3.header("A cat")
    # col3.plotly_chart(plot_3, use_container_width=True)

    # col4.header("Grayscale")
    # col4.plotly_chart(plot_4, use_container_width=True)



#%%
