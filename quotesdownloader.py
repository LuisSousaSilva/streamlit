import pandas as pd
import string
import random
import numpy as np
import random
import time

def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str
    
def normalize(df):
    df = df.dropna()
    return (df / df.iloc[0]) * 100

def merge_time_series(df_1, df_2, how='outer'):
    df = df_1.merge(df_2, how=how, left_index=True, right_index=True)
    return df
    
def compute_growth_index(dataframe, initial_value=100, initial_cost=0, ending_cost=0):
    initial_cost = initial_cost / 100
    ending_cost  = ending_cost / 100
    
    GR = ((1 + dataframe.pct_change()).cumprod()) * (initial_value * (1 - initial_cost))
    GR.iloc[0]  = initial_value * (1 - initial_cost)
    GR.iloc[-1] = GR.iloc[-1] * (1 * (1 - ending_cost))
    return GR 
    
def download_eod_data(tickers, key):
    Begin = '2000-03-10'
    ETFs = pd.DataFrame()
    
    # Download
    for ticker in tickers:
        try:
            url = "https://eodhistoricaldata.com/api/eod/" + str(ticker) + "?api_token=" + key + "&period=d."
            ETF = pd.read_csv(url, index_col = 'Date', parse_dates = True)[['Close']].iloc[:-1, :]
            ETFs = ETFs.merge(ETF, left_index = True, right_index = True, how='outer')
        except:
            print('Download of fund ' + ticker + ' failed')
            
    ETFs.columns = tickers
    ETFs = ETFs.fillna(method='ffill')
    ETFs = ETFs.replace(to_replace=0, method='ffill')
    
    return ETFs

def download_eod_data_single(tickers, key):
    Begin = '2000-03-10'
    ETFs = pd.DataFrame()
    
    # Download
    for ticker in tickers:
        try:
            url = "https://eodhistoricaldata.com/api/eod/" + str(ticker) + "?api_token=" + key + "&period=d."
            ETF = pd.read_csv(url, index_col = 'Date', parse_dates = True)
            ETFs = ETFs.merge(ETF, left_index = True, right_index = True, how='outer')
        except:
            print('Download of fund ' + ticker + ' failed')
            
    ETFs = ETFs.dropna().replace(to_replace=0, method='ffill')
    
    return ETFs

def download_ms(MSids, nomes, key):
    # Downloading funds and creating quotes and returns dataframes
    Begin = '2000-03-10'
    fundos = pd.DataFrame()

    # Download
    for i in np.arange(len(MSids)):
        try:
            url = "https://lt.morningstar.com/api/rest.svc/timeseries_price/" + key + "?id=" + MSids[i] + "&currencyId=BAS&idtype=Morningstar&frequency=daily&startDate=" + Begin + "&outputType=CSV"
            fundo = pd.read_csv(url, sep = ";" , index_col = 'date', parse_dates = True)
            fundo =  fundo.drop('Unnamed: 2', axis=1)
            fundo.columns = [nomes[i]]
            fundos = fundos.merge(fundo, left_index = True, right_index = True, how='outer')
        except:
            print('Download of fund ' + MSids[i] + ' failed')
        
        time.sleep(random.uniform(1, 2)) # Sleep for 1 to 2 seconds

    fundos.index.name = 'Date'
    return fundos

# Cria????o da fun????o para ler m??ltiplos ficheiros do investing.com. Inclui op????o de come??o e fim da an??lise.
def read_csv_investing(tickers, start='1900-01-01', stop='2100-01-01'):
    ETFs = pd.DataFrame()

    # Para cada valor na vari??vel tickers
    for ticker in tickers:
        # Ler o ficheiro .csv correspondente, ler as datas e seleccionar s?? a coluna de pre??os
        ETF = pd.read_csv(ticker + ' Historical Data.csv', index_col='Date', parse_dates=True)[['Price']]
        # Dar o nome do ticker ?? coluna para depois podermos distinguir na DataFrame
        ETF.columns = [ticker]
        # Usar a fun????o merge_time_series usando a op????o outer (quando s??o muitos ETFs aconselho
        # sempre a fun????o outer para n??o ir "perdendo" demasiadas cota????es simplesmente porque h??
        # um ETF sem cota????o nesse dia). Por outro lado a fun????o dropna() for??a a come??arem e 
        # acabarem no mesmo dia (para serem efectivamente compar??veis)
        ETFs = merge_time_series(ETFs, ETF, how='outer')

    # Ordenar as datas para que sejam ascendentes
    ETFs = ETFs.sort_index(ascending=True)
    
    # Acrescentar fun????o de "cortar" a s??rie temporal que por defeito est?? desde 1900 at?? 2100, 
    # o que basicamente ?? um "atalho" para dizer que n??o pretendo cortar pois esse periodo dever??o
    # apanhar quaisquer datas para as quais h?? ETFs do investing
    ETFs = ETFs[start:stop]

    return ETFs

def read_xls_MSCI(tickers, nomes, start='1990', end='2100'):
    MSCIs = pd.DataFrame()
        
    for ticker in tickers:
        # Read relevant information
        MSCI = pd.read_excel(ticker + '.xlsx').iloc[6:].dropna()
        # Rename columns
        MSCI.columns = ['Date', 'Price']
        # Convert the date column to datetime
        MSCI['Date'] = pd.to_datetime(MSCI['Date'])
        # Set date column as index
        MSCI.set_index('Date', inplace=True)
        # Merge
        MSCIs = merge_time_series(MSCIs, MSCI, how='outer').dropna()
        # Start / End
        MSCIs = MSCIs[start:end]
        # Growth Index
        MSCIs = compute_growth_index(MSCIs)
        
    MSCIs.columns = nomes
    
    return MSCIs

def read_xlsx_MSCI(file_name, nomes):
        
    MSCI = pd.read_excel(file_name + '.xlsx', sheet_name='Historical', index_col='As Of', parse_dates=True)[['Fund Return Series']]
    MSCI.columns = nomes
    MSCI.index.names = ['Date']
    
    return MSCI

def download_yahoo_data(tickers, normalize_quotes=True,
                      start='1970-01-01', end='2030-12-31'):
    quotes=pd.DataFrame()
    for ticker in tickers:
        # df = yf.download(ticker, start=start, end=end, progress=False)
        df = df[['Adj Close']]
        df.columns=[ticker]
        quotes = merge_time_series(quotes, df)
    
    quotes = quotes.ffill()
     
    if normalize_quotes:
        quotes = normalize(quotes)

    return quotes