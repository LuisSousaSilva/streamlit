# importing libraries
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import datetime as dt
import seaborn as sns
import pandas as pd
import numpy as np
import statsmodels.api as sm

from pandas.tseries.offsets import DateOffset
from matplotlib.ticker import FuncFormatter
from pandas.core.base import PandasObject
from datetime import datetime

# Setting pandas dataframe display options
pd.set_option("display.max_rows", 20)
pd.set_option('display.width', 800)
pd.set_option('max_colwidth', 800)

pd.options.display.float_format = '{:,.2f}'.format

# Set matplotlib style
plt.style.use('seaborn')

# Defining today's Date
from datetime import date
today = date.today()
epoch_year = date.today().year
last_year = epoch_year - 1

#### Functions ####
def compute_growth_index(dataframe, initial_value=100, initial_cost=0, ending_cost=0):
    initial_cost = initial_cost / 100
    ending_cost  = ending_cost / 100
    
    GR = ((1 + dataframe.pct_change()).cumprod()) * (initial_value * (1 - initial_cost))
    GR.iloc[0]  = initial_value * (1 - initial_cost)
    GR.iloc[-1] = GR.iloc[-1] * (1 * (1 - ending_cost))
    return GR 

def compute_drawdowns(dataframe):
    '''
    Function to compute drawdowns of a timeseries
    given a dataframe of prices
    '''
    return (dataframe / dataframe.cummax() -1) * 100

def compute_return(dataframe, years=''):
    '''
    Function to compute drawdowns of a timeseries
    given a dataframe of prices
    '''
    if isinstance(years, int):
        years = years
        dataframe = filter_by_date(dataframe, years=years)
        return (dataframe.iloc[-1] / dataframe.iloc[0] -1) * 100

    else:
        return (dataframe.iloc[-1] / dataframe.iloc[0] -1) * 100
    
def compute_max_DD(dataframe):
    return compute_drawdowns(dataframe).min()

def compute_cagr(dataframe, years=''):
    '''
    Function to calculate CAGR given a dataframe of prices
    '''
    if isinstance(years, int):
        years = years
        dataframe = filter_by_date(dataframe, years=years)
        return(dataframe.iloc[-1].div(dataframe.iloc[0])).pow(1 / years).sub(1).mul(100)
    
    else:
        years = len(pd.date_range(dataframe.index[0], dataframe.index[-1], freq='D')) / 365
        
    return(dataframe.iloc[-1].div(dataframe.iloc[0])).pow(1 / years).sub(1).mul(100)

def compute_mar(dataframe):
    '''
    Function to calculate mar: Return Over Maximum Drawdown
    given a dataframe of prices
    '''
    return compute_cagr(dataframe).div(compute_drawdowns(dataframe).min().abs())

def compute_StdDev(dataframe, freq='days'):    
    '''
    Function to calculate annualized standart deviation
    given a dataframe of prices. It takes into account the
    frequency of the data.
    '''    
    if freq == 'days':
        return dataframe.pct_change().std().mul((np.sqrt(252))).mul(100)
    if freq == 'months':
        return dataframe.pct_change().std().mul((np.sqrt(12))).mul(100)
    if freq == 'quarters':
        return dataframe.pct_change().std().mul((np.sqrt(4))).mul(100)

def compute_sharpe(dataframe, years='', freq='days'):   
    '''
    Function to calculate the sharpe ratio given a dataframe of prices.
    '''    
    return compute_cagr(dataframe, years).div(compute_StdDev(dataframe, freq))

def compute_performance_table(dataframe, years='si', freq='days'):    
    '''
    Function to calculate a performance table given a dataframe of prices.
    Takes into account the frequency of the data.
    ''' 
    
    if years == 'si':
        years = len(pd.date_range(dataframe.index[0], dataframe.index[-1], freq='D')) / 365.25
        
        df = pd.DataFrame([compute_cagr(dataframe, years),
                           compute_StdDev(dataframe, freq),
                           compute_sharpe(dataframe, years, freq), compute_max_DD(dataframe), compute_mar(dataframe)])
        df.index = ['CAGR', 'StdDev', 'Sharpe', 'Max DD', 'MAR']
        
        df = round(df.transpose(), 2)
        
        # Colocar percentagens
        df['CAGR'] = (df['CAGR'] / 100).apply('{:.2%}'.format)
        df['StdDev'] = (df['StdDev'] / 100).apply('{:.2%}'.format)
        df['Max DD'] = (df['Max DD'] / 100).apply('{:.2%}'.format)
        
        start = str(dataframe.index[0])[0:10]
        end   = str(dataframe.index[-1])[0:10]
        # print_title('Performance from ' + start + ' to ' + end + ' (≈ ' + str(round(years, 1)) + ' years)')
        
        # Return object
        return df

    if years == 'ytd':
        last_year_end = dataframe.loc[str(last_year)].iloc[-1].name
        dataframe = dataframe[last_year_end:]

        df = pd.DataFrame([compute_cagr(dataframe, years=years),
                    compute_StdDev(dataframe), compute_sharpe(dataframe),
                    compute_max_DD(dataframe), compute_mar(dataframe)])
        df.index = ['CAGR', 'StdDev', 'Sharpe', 'Max DD', 'MAR']

        df = round(df.transpose(), 2)

        # Colocar percentagens
        df['CAGR'] = (df['CAGR'] / 100).apply('{:.2%}'.format)
        df['StdDev'] = (df['StdDev'] / 100).apply('{:.2%}'.format)
        df['Max DD'] = (df['Max DD'] / 100).apply('{:.2%}'.format)

        return df

    else:
        dataframe = filter_by_date(dataframe, years)
        df = pd.DataFrame([compute_cagr(dataframe, years=years),
                           compute_StdDev(dataframe), compute_sharpe(dataframe),
                           compute_max_DD(dataframe), compute_mar(dataframe)])
        df.index = ['CAGR', 'StdDev', 'Sharpe', 'Max DD', 'MAR']

        df = round(df.transpose(), 2)

        # Colocar percentagens
        df['CAGR'] = (df['CAGR'] / 100).apply('{:.2%}'.format)
        df['StdDev'] = (df['StdDev'] / 100).apply('{:.2%}'.format)
        df['Max DD'] = (df['Max DD'] / 100).apply('{:.2%}'.format)
        
        start = str(dataframe.index[0])[0:10]
        end   = str(dataframe.index[-1])[0:10]
        
        if years == 1:
            print_title('Performance from ' + start + ' to ' + end + ' (' + str(years) + ' year)')
        else:
            print_title('Performance from ' + start + ' to ' + end + ' (' + str(years) + ' years)')
            
        return df

def compute_performance_table_no_title(dataframe, years='si', freq='days'):    
    '''
    Function to calculate a performance table given a dataframe of prices.
    Takes into account the frequency of the data.
    ''' 
    
    if years == 'si':
        years = len(pd.date_range(dataframe.index[0], dataframe.index[-1], freq='D')) / 365.25
        
        df = pd.DataFrame([compute_cagr(dataframe, years),
                           compute_StdDev(dataframe, freq),
                           compute_sharpe(dataframe, years, freq), compute_max_DD(dataframe), compute_mar(dataframe)])
        df.index = ['CAGR', 'StdDev', 'Sharpe', 'Max DD', 'MAR']
        
        df = round(df.transpose(), 2)
        
        # Colocar percentagens
        df['CAGR'] = (df['CAGR'] / 100).apply('{:.2%}'.format)
        df['StdDev'] = (df['StdDev'] / 100).apply('{:.2%}'.format)
        df['Max DD'] = (df['Max DD'] / 100).apply('{:.2%}'.format)
        
        start = str(dataframe.index[0])[0:10]
        end   = str(dataframe.index[-1])[0:10]
        
        # Return object
        return df

    if years == 'ytd':
        last_year_end = dataframe.loc[str(last_year)].iloc[-1].name
        dataframe = dataframe[last_year_end:]

        df = pd.DataFrame([compute_cagr(dataframe, years=years),
                    compute_StdDev(dataframe), compute_sharpe(dataframe),
                    compute_max_DD(dataframe), compute_mar(dataframe)])
        df.index = ['CAGR', 'StdDev', 'Sharpe', 'Max DD', 'MAR']

        df = round(df.transpose(), 2)

        # Colocar percentagens
        df['CAGR'] = (df['CAGR'] / 100).apply('{:.2%}'.format)
        df['StdDev'] = (df['StdDev'] / 100).apply('{:.2%}'.format)
        df['Max DD'] = (df['Max DD'] / 100).apply('{:.2%}'.format)

        return df

    else:
        dataframe = filter_by_date(dataframe, years)
        df = pd.DataFrame([compute_cagr(dataframe, years=years),
                           compute_StdDev(dataframe), compute_sharpe(dataframe),
                           compute_max_DD(dataframe), compute_mar(dataframe)])
        df.index = ['CAGR', 'StdDev', 'Sharpe', 'Max DD', 'MAR']

        df = round(df.transpose(), 2)

        # Colocar percentagens
        df['CAGR'] = (df['CAGR'] / 100).apply('{:.2%}'.format)
        df['StdDev'] = (df['StdDev'] / 100).apply('{:.2%}'.format)
        df['Max DD'] = (df['Max DD'] / 100).apply('{:.2%}'.format)
        
        start = str(dataframe.index[0])[0:10]
        end   = str(dataframe.index[-1])[0:10]
            
        return df

def compute_time_period(timestamp_1, timestamp_2):
    
    year = timestamp_1.year - timestamp_2.year
    month = timestamp_1.month - timestamp_2.month
    day = timestamp_1.day - timestamp_2.day
    
    if month < 0:
        year = year - 1
        month = 12 + month
    
    if day < 0:
        day = - day
        
    # Returns datetime object in years, month, days
    return(str(year) + ' Years ' + str(month) + ' Months ' + str(day) + ' Days')
    
def get(quotes):

    # resample quotes to business month
    monthly_quotes = quotes.resample('BM').last()
    
    # get monthly returns
    returns = monthly_quotes.pct_change()

    # get close / first column if given DataFrame
    if isinstance(returns, pd.DataFrame):
        returns.columns = map(str.lower, returns.columns)
        if len(returns.columns) > 1 and 'close' in returns.columns:
            returns = returns['close']
        else:
            returns = returns[returns.columns[0]]

    # get returnsframe
    returns = pd.DataFrame(data={'Retornos': returns})
    returns['Ano'] = returns.index.strftime('%Y')
    returns['Mês'] = returns.index.strftime('%b')

    # make pivot table
    returns = returns.pivot('Ano', 'Mês', 'Retornos').fillna(0)

    # order columns by month
    returns = returns[['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']]

    return returns

def plot(returns,
         title="Monthly Returns (%)",
         title_color="black",
         title_size=12,
         annot_size=10,
         figsize=None,
         cmap='RdYlGn',
         cbar=False,
         square=False):

    returns = get(returns)
    returns *= 100
    
    if figsize is None:
        size = list(plt.gcf().get_size_inches()) 
        figsize = (size[0], size[0] // 2)
        plt.close()

    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.heatmap(returns, ax=ax, annot=True,
                     annot_kws={"size": annot_size}, fmt="0.2f", linewidths=0.4, center=0,
                     square=square, cbar=cbar, cmap=cmap)
    ax.set_title(title, fontsize=title_size, color=title_color, fontweight="bold")

    fig.subplots_adjust(hspace=0)
    plt.yticks(rotation=0)
    plt.show()
    plt.close()


PandasObject.get_returns_heatmap = get
PandasObject.plot_returns_heatmap = plot

def calendarize(returns):
    
    '''
    The calendarize function is an slight adaption of ranaroussi's monthly-returns-heatmap 
    You can find it here: https://github.com/ranaroussi/monthly-returns-heatmap/
    
    It turns monthly data into a 12 columns(months) and yearly row seaborn heatmap
    '''
    
    # get close / first column if given DataFrame
    if isinstance(returns, pd.DataFrame):
        returns.columns = map(str.lower, returns.columns)
        if len(returns.columns) > 1 and 'close' in returns.columns:
            returns = returns['close']
        else:
            returns = returns[returns.columns[0]]

    # get returnsframe
    returns = pd.DataFrame(data={'Retornos': returns})
    returns['Ano'] = returns.index.strftime('%Y')
    returns['Mês'] = returns.index.strftime('%b')

    # make pivot table
    returns = returns.pivot('Ano', 'Mês', 'Retornos').fillna(0)

    # order columns by month
    returns = returns[['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']]

    return returns

def plotly_table(df, width=990, height=500, columnwidth=[25], title=None , index=True, header=True,
                 header_alignment=['center'],  header_line_color='rgb(100, 100, 100)', header_font_size=[12],
                 header_font_color=['rgb(45, 45, 45)'], header_fill_color=['rgb(200, 200, 200)'],
                 cells_alignment=['center'], cells_line_color=['rgb(200, 200, 200)'], cells_font_size=[11], 
                 cells_font_color=['rgb(45, 45, 45)'], cells_fill_color=['rgb(245, 245, 245)','white' ]):
    
    # Making the header bold and conditional  
        if (header == False and index == False):            
            lst = list(df.columns[0 + i] for i in range(len(df.columns)))  
            header = [[i] for i in lst]
            header =  list([str( '<b>' + header[0 + i][0] + '</b>') for i in range(len(df.columns))])
            header = [[i] for i in header]
            header.pop(0)
            header =  [[]] + header
            
            trace = go.Table(
                columnwidth = columnwidth,
                    header=dict(values=header,
                              line = dict(color=header_line_color),
                              align = header_alignment,
                              font = dict(color=header_font_color, size=header_font_size),
                              height = 22,
                              fill = dict(color=header_fill_color)),
            
            cells=dict(values=df.transpose().values.tolist(),                       
                       line=dict(color=cells_line_color),
                       align = cells_alignment,
                       height = 22,
                       font = dict(color=cells_font_color, size=cells_font_size),
                       fill = dict(color = [cells_fill_color * len(df.index)]),
                      ),      
        )
              
        # Making the header bold and conditional  
        if (header == True and index == True):            
            lst = list(df.columns[0 + i] for i in range(len(df.columns)))  
            header = [[i] for i in lst]
            header =  list([str( '<b>' + header[0 + i][0] + '</b>') for i in range(len(df.columns))])
            header = [[i] for i in header]
            header = [['']] + header
            
            # Making the index Bold
            lst_i = list(df.index[0 + i] for i in range(len(df.index)))
            index = [[i] for i in lst_i]
            index =  list([[ '<b>' + str(index[0 + i][0]) + '</b>' for i in range(len(df.index))]])
            
            trace = go.Table(
                columnwidth = columnwidth,
                    header=dict(values=header,
                              line = dict(color=header_line_color),
                              align = header_alignment,
                              font = dict(color=header_font_color, size=header_font_size),
                              height = 22,
                              fill = dict(color=header_fill_color)),
            
            cells=dict(values=index + df.transpose().values.tolist(),                       
                       line=dict(color=cells_line_color),
                       align = cells_alignment,
                       height = 22,
                       font = dict(color=cells_font_color, size=cells_font_size),
                       fill = dict(color = [cells_fill_color * len(df.index)]),
                      ),      
        )
            
        # Making the header bold and conditional  
        if (header == False and index == True):            
            lst = list(df.columns[0 + i] for i in range(len(df.columns)))  
            header = [[i] for i in lst]
            header =  list([str( '<b>' + header[0 + i][0] + '</b>') for i in range(len(df.columns))])
            header = [[i] for i in header]
            header = [[]] + header
            
            lst_i = list(df.index[0 + i] for i in range(len(df.index)))
            index = [[i] for i in lst_i]
            index =  list([[ '<b>' + str(index[0 + i][0]) + '</b>' for i in range(len(df.index))]])
            
            trace = go.Table(
                columnwidth = columnwidth,
                    header=dict(values=header,
                              line = dict(color=header_line_color),
                              align = header_alignment,
                              font = dict(color=header_font_color, size=header_font_size),
                              height = 22,
                              fill = dict(color=header_fill_color)),
            
            cells=dict(values=index + df.transpose().values.tolist(),                       
                       line=dict(color=cells_line_color),
                       align = cells_alignment,
                       height = 22,
                       font = dict(color=cells_font_color, size=cells_font_size),
                       fill = dict(color = [cells_fill_color * len(df.index)]),
                      ),      
        )
            
        # Making the header bold and conditional  
        if (header == True and index == False):            
            lst = list(df.columns[0 + i] for i in range(len(df.columns)))  
            header = [[i] for i in lst]
            header =  list([str( '<b>' + header[0 + i][0] + '</b>') for i in range(len(df.columns))])
            header = [[i] for i in header]
            header = header
            
            trace = go.Table(
                columnwidth = columnwidth,
                    header=dict(values=header,
                              line = dict(color=header_line_color),
                              align = header_alignment,
                              font = dict(color=header_font_color, size=header_font_size),
                              height = 22,
                              fill = dict(color=header_fill_color)),
            
            cells=dict(values=df.transpose().values.tolist(),                       
                       line=dict(color=cells_line_color),
                       align = cells_alignment,
                       height = 22,
                       font = dict(color=cells_font_color, size=cells_font_size),
                       fill = dict(color = [cells_fill_color * len(df.index)]),
                      ),      
        )
        
        if title == None:
            layout = go.Layout(
                autosize=False,
                height=height,
                width=width,
                margin=dict (l=0, r=0, b=0, t=0, pad=0),
            )
        else:
                layout = go.Layout(
                    autosize=False,
                    height=height,
                    width=width,
                    title=title,
                    margin=dict( l=0, r=0, b=0, t=25, pad=0),
                )

        data = [trace]
        fig = go.Figure(data=data, layout=layout)

def get_quotes(tickers, names, normalize=False):
    Quotes = pd.read_csv('D:/GDrive/_GitHub/Backtester/Data/Cotacoes_diarias_all.csv', index_col='Date', parse_dates=True)[tickers].dropna()
    Quotes.columns=names
    
    if normalize==True:
        return normalize(Quotes)
    else:
        return Quotes

def compute_portfolio(quotes, weights):
    
    Nomes=quotes.columns
    
    # Anos do Portfolio
    Years = quotes.index.year.unique()

    # Dicionário com Dataframes anuais das cotações dos quotes
    Years_dict = {}
    k = 0

    for Year in Years:
        # Dynamically create key
        key = Year
        # Calculate value
        value = quotes.loc[str(Year)]
        # Insert in dictionary
        Years_dict[key] = value
        # Counter
        k += 1

    # Dicionário com Dataframes anuais das cotações dos quotes
    Quotes_dict = {}
    Portfolio_dict = {}

    k = 0    
    
    for Year in Years:
        
        n = 0
        
        #Setting Portfolio to be a Global Variable
        global Portfolio
        
        # Dynamically create key
        key = Year

        # Calculate value
        if (Year-1) in Years:
            value = Years_dict[Year].append(Years_dict[Year-1].iloc[[-1]]).sort_index()
        else:
            value = Years_dict[Year].append(Years_dict[Year].iloc[[-1]]).sort_index()

        # Set beginning value to 100
        value = (value / value.iloc[0]) * 100
        # 
        for column in value.columns:
            value[column] = value[column] * weights[n]
            n +=1
        
        # Get Returns
        Returns = value.pct_change()
        # Calculating Portfolio Value
        value['Portfolio'] = value.sum(axis=1)

        # Creating Weights_EOP empty DataFrame
        Weights_EOP = pd.DataFrame()
        # Calculating End Of Period weights
        for Name in Nomes:
            Weights_EOP[Name] = value[Name] / value['Portfolio']
        # Calculating Beginning Of Period weights
        Weights_BOP = Weights_EOP.shift(periods=1)

        # Calculatins Portfolio Value
        Portfolio = pd.DataFrame(Weights_BOP.multiply(Returns).sum(axis=1))
        Portfolio.columns=['Simple']
        # Transformar os simple returns em log returns 
        Portfolio['Log'] = np.log(Portfolio['Simple'] + 1)
        # Cumsum() dos log returns para obter o preço do Portfolio 
        Portfolio['Price'] = 100*np.exp(np.nan_to_num(Portfolio['Log'].cumsum()))
        Portfolio['Price'] = Portfolio['Price']   

        # Insert in dictionaries
        Quotes_dict[key] = value
        Portfolio_dict[key] = Portfolio
        # Counter
        k += 1

    # Making an empty Dataframe for Portfolio data
    Portfolio = pd.DataFrame()

    for Year in Years:
        Portfolio = pd.concat([Portfolio, Portfolio_dict[Year]['Log']])

    # Delete repeated index values in Portfolio    
    Portfolio.drop_duplicates(keep='last')

    # Naming the column of log returns 'Log'
    Portfolio.columns= ['Log']

    # Cumsum() dos log returns para obter o preço do Portfolio 
    Portfolio['Price'] = 100*np.exp(np.nan_to_num(Portfolio['Log'].cumsum()))
        
    # Round Portfolio to 2 decimals and eliminate returns
    Portfolio = pd.DataFrame(round(Portfolio['Price'], 2))

    # Naming the column of Portfolio as 'Portfolio'
    Portfolio.columns= ['Portfolio']

    # Delete repeated days
    Portfolio = Portfolio.loc[~Portfolio.index.duplicated(keep='first')]

    return Portfolio
    
# Multi_period_return (in CAGR)
def multi_period_return(df, years = 1, days=252):
    shifted = df.shift(days * years)
    One_year = (((1 + (df - shifted) / shifted) ** (1 / years))-1)  * 100
    return One_year

def compute_drawdowns_i(dataframe):
    '''
    Function to compute drawdowns based on 
    the inicial value of a timeseries
    given a dataframe of prices
    '''
    return (dataframe / 100 -1) * 100
        
def print_title(string):
    display(Markdown('**' + string + '**'))

def print_italics(string):
    display(Markdown('*' + string + '*'))
    
def all_percent(df, rounding_value=2):
    return round(df, rounding_value).astype(str) + '%'

def preview(df):
    df = pd.concat([df.head(3), df.tail(4)])
    df.iloc[3] = '...'
    return df

def normalize(df):
    df = df.dropna()
    return (df / df.iloc[0]) * 100
    
dimensions=(950, 500)

colorz = ['royalblue', 'orange', 'dimgrey', 'darkorchid']

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

### print(color.BOLD + 'Hello World !' + color.END)

##################################################
### Begin of compute_drawdowns_table function ####
##################################################

### Função auxiliar 1
def compute_time_period(timestamp_1, timestamp_2):
    
    year = timestamp_1.year - timestamp_2.year
    month = timestamp_1.month - timestamp_2.month
    day = timestamp_1.day - timestamp_2.day
    
    if month < 0:
        year = year - 1
        month = 12 + month
    
    if day == 0:
        day = - day
        
    if day < 0:
        month =  month - 1
        if timestamp_1.month not in [1, 3, 5, 7, 8, 10, 12]:
            day = 31 + day
        else:
            day = 30 + day        
        
    # Returns datetime object in years, month, days
    return(str(year) + ' Years, ' + str(month) + ' Months, ' + str(day) + ' Days')

### Função auxiliar 2
def compute_drawdowns_periods(df):
    
    # Input: df of max points in drawdowns (where dd == 0)
    
    drawdown_periods = list()

    for i in range(0, len(df.index)):
      
        drawdown_periods.append(compute_time_period(df.index[i], df.index[i - 1]))
    
    drawdown_periods = pd.DataFrame(drawdown_periods)
    
    return (drawdown_periods)


### Função auxiliar 3
def compute_max_drawdown_in_period(prices, timestamp_1, timestamp_2):
    
    df = prices[timestamp_1:timestamp_2]
    
    max_dd = compute_max_DD(df)
    
    return max_dd

### Função auxiliar 4
def compute_drawdowns_min(df, prices):
    
    # Input: df of max points in drawdowns (where dd == 0)
    
    drawdowns_min = list()

    for i in range(0, len(df.index) - 1):
      
        drawdowns_min.append(compute_max_drawdown_in_period(prices, df.index[i], df.index[i + 1]))
    
    drawdowns_min = pd.DataFrame(drawdowns_min)
    
    return(drawdowns_min)

### Função principal
def compute_drawdowns_table(prices, number=5):

    # input: df of prices
    dd = compute_drawdowns(prices)
    
    max_points = dd[dd == 0].dropna()
        
    data = [0.0] 
  
    # Create the pandas DataFrame
    new_data = pd.DataFrame(data, columns = ['New_data'])

    new_data['Date'] = prices.index.max()

    new_data.set_index('Date', inplace=True)
    
    max_points = max_points.loc[~max_points.index.duplicated(keep='first')]

    max_points = pd.DataFrame(pd.concat([max_points, new_data], axis=1).iloc[:, 0])
    
    dp = compute_drawdowns_periods(max_points)
        
    dp.set_index(max_points.index, inplace=True)
    
    df = pd.concat([max_points, dp], axis=1)
    
    df.index.name = 'Date'
    
    df.reset_index(inplace=True)
    
    df['End'] = df['Date'].shift(-1)
    
    df[0] = df[0].shift(-1)
    
    df['values'] = round(compute_drawdowns_min(max_points, prices), 2)
    
    df = df.sort_values(by='values')
    
    df['Number'] = range(1, len(df) + 1)
    
    df.reset_index(inplace=True)
    
    df.columns = ['index', 'Begin', 'point', 'Length', 'End', 'Depth', 'Number']
    
    df = df[['Begin', 'End', 'Depth', 'Length']].head(number)
    
    df.iloc[:, 2] = df.iloc[:, 2].apply( lambda x : str(x) + '%')
    
    df.set_index(np.arange(1, number + 1), inplace=True)

    df['End'] = df['End'].astype(str)

    df['Begin'] = df['Begin'].astype(str)

    for i in range(0, len(df['End'])):
        if df['End'].iloc[i] == str(prices.iloc[-1].name)[0:10]:
            df['End'].iloc[i] = str('N/A')

    return(df)

################################################
### End of compute_drawdowns_table function ####
################################################

def compute_r2(x, y, k=1):
    xpoly = np.column_stack([x**i for i in range(k+1)])    
    return sm.OLS(y, xpoly).fit().rsquared

def compute_r2_table(df, benchmark):

# df of prices

    lista = []

    for i in np.arange(0, len(df.columns)):
        lista.append(compute_r2(benchmark, df.iloc[: , i]))
             
    Dataframe = pd.DataFrame(lista)
    
    Dataframe.index = df.columns
    
    Dataframe.columns = benchmark.columns
    
    return(round(Dataframe.transpose(), 3))

colors = ['royalblue',            # 1 - royalblue
          'dimgrey',              # 2 - dimgrey
          'rgb(255, 153, 51)',    # 3 - orange
          'indigo',               # 4 - Indigo
          'rgb(219, 64, 82)',     # 5 - Red
          'rgb(0, 128, 128)',     # 6 - Teal
          '#191970',              # 7 - Navy
          'rgb(128, 128, 0)',     # 8 - Olive
          '#00BFFF',              # 9 - Water Blue
          'rgb(128, 177, 211)']   # 10 - Blueish

def compute_costs(DataFrame, percentage, sessions_per_year=365, Nome='Price'):
    DataFrame = pd.DataFrame(DataFrame.copy())
    DataFrame['Custos'] = (percentage/sessions_per_year) / 100
    DataFrame['Custos_shifted'] = DataFrame['Custos'].shift(1)
    DataFrame['Custos_acumulados'] = DataFrame['Custos_shifted'].cumsum()
    DataFrame[Nome] = DataFrame.iloc[ : ,0] * (1-DataFrame['Custos_acumulados'])
    DataFrame = DataFrame[[Nome]]
    DataFrame = DataFrame.fillna(100)
    return DataFrame

def compute_ms_performance_table(DataFrame, freq='days'):
    nr_of_days = int(str(DataFrame.index[-1] - DataFrame.index[0])[0:4])

    if nr_of_days < 365:
        df = compute_performance_table(DataFrame, freq=freq)
        df.index = ['S.I.']
        df = df[['CAGR', 'StdDev', 'Sharpe', 'Max DD', 'MAR']]

    elif nr_of_days >= 365 and nr_of_days < 365*3:
        df0 = compute_performance_table(DataFrame)
        df_ytd = compute_performance_table_no_title(DataFrame, years='ytd')
        df1 = compute_performance_table_no_title(DataFrame, years=1)
        df = pd.concat([df0, df_ytd, df1])
        df.index = ['S.I.', 'YTD', '1 Year']
        df = df[['CAGR', 'StdDev', 'Sharpe', 'Max DD', 'MAR']]

    elif nr_of_days >= 365*3 and nr_of_days < 365*5:
        df0 = compute_performance_table(DataFrame)
        df_ytd = compute_performance_table_no_title(DataFrame, years='ytd')
        df1 = compute_performance_table_no_title(DataFrame, years=1)
        df3 = compute_performance_table_no_title(DataFrame, years=3)
        df = pd.concat([df0, df_ytd, df1, df3])
        df.index = ['S.I.', 'YTD', '1 Year', '3 Years']
        df = df[['CAGR', 'StdDev', 'Sharpe', 'Max DD', 'MAR']]

    if nr_of_days >= 365*5 and nr_of_days < 365*10:
        df0 = compute_performance_table(DataFrame)
        df_ytd = compute_performance_table_no_title(DataFrame, years='ytd')
        df1 = compute_performance_table_no_title(DataFrame, years=1)
        df3 = compute_performance_table_no_title(DataFrame, years=3)
        df5 = compute_performance_table_no_title(DataFrame, years=5)
        df = pd.concat([df0, df_ytd, df1, df3, df5])
        df.index = ['S.I.', 'YTD', '1 Year', '3 Years', '5 Years']
        df = df[['CAGR', 'StdDev', 'Sharpe', 'Max DD', 'MAR']]

    elif nr_of_days >= 365*10 and nr_of_days < 365*15:
        df0 = compute_performance_table(DataFrame)
        df_ytd = compute_performance_table_no_title(DataFrame, years='ytd')
        df1 = compute_performance_table_no_title(DataFrame, years=1)
        df3 = compute_performance_table_no_title(DataFrame, years=3)
        df5 = compute_performance_table_no_title(DataFrame, years=5)
        df10 = compute_performance_table_no_title(DataFrame, years=10)
        df = pd.concat([df0, df_ytd, df1, df3, df5, df10])
        df.index = ['S.I.', 'YTD', '1 Year', '3 Years', '5 Years', '10 Years']
        df = df[['CAGR', 'StdDev', 'Sharpe', 'Max DD', 'MAR']]

    # elif nr_of_days >= 365*15 and nr_of_days < 365*20:
    else:
        df0 = compute_performance_table(DataFrame)
        df_ytd = compute_performance_table_no_title(DataFrame, years='ytd')
        df1 = compute_performance_table_no_title(DataFrame, years=1)
        df3 = compute_performance_table_no_title(DataFrame, years=3)
        df5 = compute_performance_table_no_title(DataFrame, years=5)
        df10 = compute_performance_table_no_title(DataFrame, years=10)
        df15= compute_performance_table_no_title(DataFrame, years=15)
        df = pd.concat([df0, df_ytd, df1, df3, df5, df10, df15])
        df.index = ['S.I.', 'YTD', '1 Year', '3 Years', '5 Years', '10 Years', '15 Years']
        df = df[['CAGR', 'Return', 'StdDev', 'Sharpe', 'Max DD', 'MAR']]

    return df

def compute_log_returns(prices):
    """
    Compute log returns for each ticker.
    
    INPUT
    ----------
    prices
    
    OUTPUT
    -------
    log_returns
    """
    
    return np.log(prices) - np.log(prices.shift())

def merge_time_series(df_1, df_2, on='', how='outer'):
    '''
    on = 'index'
    '''
    if on=='index':
        df = pd.concat([df_1, df_2], axis=0).sort_index().drop_duplicates()

        return df.sort_index()
    else:
        df = df_1.merge(df_2, how=how, left_index=True, right_index=True)
        return df
        
colors_list=['royalblue', 'darkorange',
           'dimgrey', 'rgb(86, 53, 171)',  'rgb(44, 160, 44)',
           'rgb(214, 39, 40)', '#ffd166', '#62959c', '#b5179e',
           'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
           'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
           'rgb(188, 189, 34)', 'rgb(23, 190, 207)']

lightcolors = [
    'royalblue',
   'rgb(111, 231, 219)',
   'rgb(131, 90, 241)',
              
              ]

colors_list=['royalblue', 'darkorange',
           'dimgrey', 'rgb(86, 53, 171)',  'rgb(44, 160, 44)',
           'rgb(214, 39, 40)', '#ffd166', '#62959c', '#b5179e',
           'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
           'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
           'rgb(188, 189, 34)', 'rgb(23, 190, 207)'] * 10

def ichart(data, title='', colors=colors_list, yTitle='', xTitle='', style='normal',
        width=990, height=500, hovermode='x', yticksuffix='', ytickprefix='',
        ytickformat="", source_text='', y_position_source='-0.125', xticksuffix='',
        xtickprefix='', xtickformat="", dd_range=[-50, 0], y_axis_range_range=None,
        log_y=False, image=''):

    '''
    style = normal, area, drawdowns_histogram
    colors = color_list or lightcolors
    hovermode = 'x', 'x unified', 'closest'
    y_position_source = -0.125 or bellow
    dd_range = [-50, 0]
    ytickformat =  ".1%"
    image: 'forum' ou 'fp
    
    '''
    
    if image=='fp':
        image='https://raw.githubusercontent.com/LuisSousaSilva/Articles_and_studies/master/FP-cor-positivo.png'
    elif image=='forum':
        image='https://raw.githubusercontent.com/LuisSousaSilva/Articles_and_studies/master/logo_forum.png'
        
    fig = go.Figure()

    fig.update_layout(
        paper_bgcolor='#F5F6F9',
        plot_bgcolor='#F5F6F9',
        width=width,
        height=height,
        hovermode=hovermode,
        title=title,
        title_x=0.5,
        yaxis = dict(
            ticksuffix=yticksuffix,
            tickprefix=ytickprefix,
            tickfont=dict(color='#4D5663'),
            gridcolor='#E1E5ED',
            range=y_axis_range_range,
            titlefont=dict(color='#4D5663'),
            zerolinecolor='#E1E5ED',
            title=yTitle,
            showgrid=True,
            tickformat=ytickformat,
                    ),
        xaxis = dict(
            title=xTitle,
            tickfont=dict(color='#4D5663'),
            gridcolor='#E1E5ED',
            titlefont=dict(color='#4D5663'),
            zerolinecolor='#E1E5ED',
            showgrid=True,
            tickformat=xtickformat,
            ticksuffix=xticksuffix,
            tickprefix=xtickprefix,
                    ),
        images= [dict(
            name= "watermark_1",
            source= image,
            xref= "paper",
            yref= "paper",
            x= -0.05500,
            y= 1.250,
            sizey= 0.20,
            sizex= 0.20,
            opacity= 1,
            layer= "below"
        )],
        annotations=[dict(
            xref="paper",
            yref="paper",
            x= 0.5,
            y= y_position_source,
            xanchor="center",
            yanchor="top",
            text=source_text,
            showarrow= False,
            font= dict(
                family="Arial",
                size=12,
                color="rgb(150,150,150)"
                )
        )
    ]

    ), # end

    if log_y:

        fig.update_yaxes(type="log")

    if style=='normal':
        z = -1
        
        for i in data:
            z = z + 1
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[i],
                mode='lines',
                name=i,
                line=dict(width=1.3,
                        color=colors[z]),
            ))

    if style=='area':
        z = -1
        
        for i in data:
            z = z + 1
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[i],
                hoverinfo='x+y',
                mode='lines',
                name=i,
                line=dict(width=0.7,
                        color=colors[z]),
                stackgroup='one' # define stack group
            ))

    if style=='drawdowns_histogram':
        fig.add_trace(go.Histogram(x=data.iloc[:, 0],
                     histnorm='probability',
                     marker=dict(colorscale='RdBu',
                                 reversescale=False,
                                 cmin=-24,
                                 cmax=0,
                                 color=np.arange(start=dd_range[0], stop=dd_range[1]),
                                 line=dict(color='white', width=0.2)),
                     opacity=0.75,
                     cumulative=dict(enabled=True)))

    return fig

def compute_rolling_cagr(dataframe, years):
    index = dataframe.index + pd.DateOffset(years=years)

    start = dataframe.index[0]
    end   = dataframe.index[-1]

    portfolio = dataframe.copy()
    portfolio.set_index(index, inplace=True)

    portfolio = portfolio[start:]

    rr = (dataframe. iloc[:, 0] / portfolio. iloc[:, 0]  - 1) * 100
    rr = rr.loc[:end]
    
    return pd.DataFrame(((((rr/100) + 1))**(1/years))-1)

def filter_by_years(dataframe, years=0):
    
    last_date = dataframe.tail(1).index
    year_nr = last_date.year.values[0]
    month_nr = last_date.month.values[0]
    day_nr = last_date.day.values[0]
    
    if month_nr == 2 and day_nr == 29 and years % 4 != 0:
        new_date = str(year_nr - years) + '-' + str(month_nr) + '-' + str(day_nr-1)        
    else:
        new_date = str(year_nr - years) + '-' + str(month_nr) + '-' + str(day_nr)
    
    df = dataframe.loc[new_date:]
    
    dataframe = pd.concat([dataframe.loc[:new_date].tail(1), dataframe.loc[new_date:]])
    # Delete repeated days
    dataframe = dataframe.loc[~dataframe.index.duplicated(keep='first')]

    return dataframe

def filter_by_date(dataframe, years=0):

    '''
    Legacy function
    '''
    
    last_date = dataframe.tail(1).index
    year_nr = last_date.year.values[0]
    month_nr = last_date.month.values[0]
    day_nr = last_date.day.values[0]
    
    if month_nr == 2 and day_nr == 29 and years % 4 != 0:
        new_date = str(year_nr - years) + '-' + str(month_nr) + '-' + str(day_nr-1)        
    else:
        new_date = str(year_nr - years) + '-' + str(month_nr) + '-' + str(day_nr)
    
    df = dataframe.loc[new_date:]
    
    dataframe = pd.concat([dataframe.loc[:new_date].tail(1), dataframe.loc[new_date:]])
    # Delete repeated days
    dataframe = dataframe.loc[~dataframe.index.duplicated(keep='first')]

    return dataframe

def color_negative_red(value):
  """
  Colors elements in a dateframe
  green if positive and red if
  negative. Does not color NaN
  values.
  """

  if value < 0:
    color = 'red'
  elif value > 0:
    color = 'green'
  else:
    color = 'black'

  return 'color: %s' % color

def compute_yearly_returns(dataframe, start='1900', end='2100', style='table',
                        title='Yearly Returns', color=False, warning=True): 
    '''
    Style: table // string // chart
    '''
    # Getting start date
    start = str(dataframe.index[0])[0:10]

    # Resampling to yearly (business year)
    yearly_quotes = dataframe.resample('BA').last()

    # Adding first quote (only if start is in the middle of the year)
    yearly_quotes = pd.concat([dataframe.iloc[:1], yearly_quotes])
    first_year = dataframe.index[0].year - 1
    last_year = dataframe.index[-1].year + 1

    # Returns
    yearly_returns = ((yearly_quotes / yearly_quotes.shift(1)) - 1) * 100
    yearly_returns = yearly_returns.set_index([list(range(first_year, last_year))])

    #### Inverter o sentido das rows no dataframe ####
    yearly_returns = yearly_returns.loc[first_year + 1:last_year].transpose()
    yearly_returns = round(yearly_returns, 2)

    # As strings and percentages
    yearly_returns.columns = yearly_returns.columns.map(str)    
    yearly_returns_numeric = yearly_returns.copy()

    if style=='table'and color==False:
        yearly_returns = yearly_returns / 100
        yearly_returns = yearly_returns.style.format("{:.2%}")
        print_title(title)

    
    elif style=='table':
        yearly_returns = yearly_returns / 100
        yearly_returns = yearly_returns.style.applymap(color_negative_red).format("{:.2%}")
        print_title(title)

    elif style=='numeric':
        yearly_returns = yearly_returns_numeric.copy()


    elif style=='string':
        for column in yearly_returns:
            yearly_returns[column] = yearly_returns[column].apply( lambda x : str(x) + '%')

        
    elif style=='chart':
        fig, ax = plt.subplots()
        fig.set_size_inches(yearly_returns_numeric.shape[1] * 1.25, yearly_returns_numeric.shape[0] + 0.5)
        yearly_returns = sns.heatmap(yearly_returns_numeric, annot=True, cmap="RdYlGn", linewidths=.2, fmt=".2f", cbar=False, center=0)
        for t in yearly_returns.texts: t.set_text(t.get_text() + "%")
        plt.title(title)
    
    else:
        print('At least one parameter has a wrong input')

    return yearly_returns

def beautify_columns(dataframe, column_numbers, symbol):
    for column_number in column_numbers:
        # Transformar em string
        for i in np.arange(0, len(dataframe.index)): # Talvez faz um as.type(str) ao dataframe todo
            dataframe.iloc[i , column_number] = \
            str(round(dataframe.iloc[i , column_number], 2))

            # Se for 0, passar a se 0.00 + symbol
            if dataframe.iloc[i , column_number] == '0':
                dataframe.iloc[i , column_number] = '0.00' + symbol
                
            # Se só tem 1 número a seguir ao ponto acrescentar um zero
            # (para ter duas casa decimais) e o símbolo do euro
            if len(dataframe.iloc[i , column_number].partition('.')[2]) < 2:
                dataframe.iloc[i , column_number] = \
                dataframe.iloc[i , column_number].partition('.')[0] \
                + dataframe.iloc[i , column_number].partition('.')[1] \
                + dataframe.iloc[i , column_number].partition('.')[2][0:1] \
                + '0' + symbol
                
            # Se já tem 2 duas casas decimais acrescentar só o símbolo de euro
            if len(dataframe.iloc[i , column_number].partition('.')[2]) >= 2 \
            and symbol not in dataframe.iloc[i , column_number]:
                dataframe.iloc[i , column_number] =\
                dataframe.iloc[i , column_number] + symbol
                
            # Se tem mais de 3 casas antes do ponto acrescentar uma vírgula
            if len(dataframe.iloc[i , column_number].partition('.')[0]) > 3:
                dataframe.iloc[i , column_number] = \
                dataframe.iloc[i , column_number].partition('.')[0][:-3] \
                + ',' \
                + dataframe.iloc[i , column_number].partition('.')[0][-3:] \
                + dataframe.iloc[i , column_number].partition('.')[1] \
                + dataframe.iloc[i , column_number].partition('.')[2]
                
            # Se tem mais de 6 casas antes do ponto fazer uma virgula de milhões
            if len(dataframe.iloc[i , column_number].partition('.')[0]) > 7:
                dataframe.iloc[i , column_number] =\
                dataframe.iloc[i , column_number].partition(',')[0][:-3] \
                + ',' \
                + dataframe.iloc[i , column_number].partition(',')[0][-3:] \
                + dataframe.iloc[i , column_number].partition(',')[1] \
                + dataframe.iloc[i , column_number].partition(',')[2]
    
    return dataframe

def clean_dataframe(dataframe, values_to_clean):
    for value in values_to_clean:
        dataframe = dataframe.replace({value: '-'}, regex=True)
        
    return dataframe


def ints_to_floats(dataframe):
    # Para cada coluna da dataframe
    for column in dataframe.columns:
        # Se a coluna for int
        if dataframe[column].dtype == 'int64':
            # transforma-la em float
            dataframe[column] = dataframe[column].astype('float')

    return dataframe

def compute_time_series(dataframe, start_value=100):

#    INPUT: Dataframe of returns
#    OUTPUT: Growth time series starting in 100

    return (np.exp(np.log1p(dataframe).cumsum())) * start_value

def compute_yearly_returns_warning(dataframe):
    start = str(dataframe.index[0])[0:10]
    print_italics('Note: First Year only has performance since ' + start)

def deflate(data, inflation_rate='0.02'):
    '''
    DATA = Dataframe with nominal values to deflate at
    the given inflation rate inflation rate
    '''
    df = pd.DataFrame(0, index=data.index, columns=['Deflator', 'Inflation'])
    df['Inflation'] = 1 + 0.02
    df['Inflation'].iloc[0] = 1
    df['Deflator'] = df['Inflation'].cumprod() * 100
    data_deflated = data.div(df['Deflator'], axis=0).mul(100)
    
    return data_deflated

def mini_chart(df, size=[0.5, 0.15], periods=13):
    # changing the rc parameters and plotting a line plot
    plt.rcParams['figure.figsize'] = size

    prices_list = list(df.resample('M').last().tail(periods).iloc[:, 0])

    # Marks of RAM in different subjects out of 100.
    x = np.arange(periods)    
    y = prices_list

    if prices_list[0] > prices_list[-1]:
        line_color='crimson'
    else:
        line_color='green'

    plt.xlabel("Subject")
    plt.ylabel("Ram's marks out of 100")
    plt.plot(x, y, color=line_color)
    plt.axis('off')  # command for hiding the axis.