import pandas as pd
import time
import urllib.request, json 
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
import datetime
import calendar
import csv
import pandas as pd
import os


path = os.environ["heuristik_data_path"]
path = os.path.abspath(path) + '/'
nltk.download("punkt",path)
nltk.download('stopwords',path)
symbols_nyse = pd.read_csv(path+'nyse-listed_csv.csv')['ACT Symbol'].tolist()
symbols_nasdaq = pd.read_csv(path+'nasdaq-listed-symbols_csv.csv')['Symbol'].tolist()


def name_extraction(symbol):
    """
    Resolve name from symbol.
    """
    df_nyse = pd.read_csv(path+'nyse-listed_csv.csv')
    df_nyse = df_nyse.rename(columns={"ACT Symbol": "Symbol"})
    df_nasdaq = pd.read_csv(path+'nasdaq-listed-symbols_csv.csv')
    dfs = df_nyse.append(df_nasdaq)
    results = dfs[dfs['Symbol'].str.match(symbol)]
    if len(results) == 0:
        print('No relevant symbol found.')
        symbol = ''
    elif len(results) >= 1:
        symbol = list(results['Company Name'])[0].partition(' ')[0]
        
    return symbol


def clean_raw_text(raw, 
                   max_length = 200, 
                   remove_tags = True, 
                   remove_punctuation = True, 
                   remove_stopwords = True,
                   remove_numbers = True,
                   remove_name = ''):
    """
    Clean text by removing html, tags, punctuation, company name, stopwords...
    """
    
    clean = BeautifulSoup(raw, 'lxml') # remove script
    if remove_tags:
        clean = clean.text.lower() # remove tags, lower case
        
    if remove_punctuation:
        tokenizer = nltk.RegexpTokenizer(r"\w+") # remove punctuation
        
    clean = tokenizer.tokenize(clean) #tokenize
    for i, word in enumerate(clean):
        if word == remove_name.lower():
            clean[i] = 'company' #remove clear name
            
        if remove_numbers and any(character.isdigit() for character in word):
            clean[i] = ''
            
        if remove_stopwords and (word in stopwords.words('english')):
            clean[i] = ''
            
    if len(clean)>max_length: #limit length
        clean = " ".join(clean[0:max_length])
    else:
        clean = " ".join(clean)
        
    clean = " ".join(clean.split())
    return clean


def apply_sentiment_to_text(text, price_data):
    """
    Labels text items according to price sentiment.
    """
    
    text['sentiment'] = ''
    for text_time in list(text.index):
        nearest_index = price_data.index.get_loc(text_time, method='ffill')
        text['sentiment'][text_time] = price_data['sentiment'][nearest_index]
        
    return text


def get_sentiment(df,start_time,timedelta,barriers):
    """
    Extracting stock sentiment from stock price data.
    """
    end_time = start_time + timedelta

    if end_time > df.index[-1]:
        end_time = df.index[-1]
        
    if start_time == end_time:
        sentiment = 0
    else:
        nearest_start_index = df.index.get_loc(start_time, method='bfill')
        nearest_end_index = df.index.get_loc(end_time, method='bfill')
        interval_data = df._slice(slice(nearest_start_index, nearest_end_index))
        start_price = interval_data['price'][0]
        end_price = interval_data['price'][-1]
        horizontal_barriers = start_price * pd.Series([1+barriers,1-barriers])
        upper = (interval_data['price']>horizontal_barriers[0])
        lower = (interval_data['price']<horizontal_barriers[1])
        upper_any = upper.any()
        lower_any = lower.any()
        if lower_any:
            if upper.any():
                upper_first_index = interval_data[upper].index[0]
                lower_first_index = interval_data[lower].index[0]
                if upper_first_index > lower_first_index:
                    sentiment = -1
                else: 
                    sentiment = 1
            else:
                sentiment = -1
        else:
            if upper.any():
                sentiment = 1
            else:
                sentiment = 0
                
    return sentiment


def apply_sentiment(df,timedelta = pd.Timedelta('7 days'),barriers = 0.05):
    """
    Adding sentiment to a stock price dataframe
    """
    print('Extract price sentiment. Timeframe: '+ str(timedelta)+'. Barriers : '+ str(100*barriers)+'%.')
    return df.index.map(lambda s: get_sentiment(df,s,timedelta,barriers))


def download_price(symbol, # name of the equity
             function = 'SMA',  # Values: 'SMA', TIME_SERIES_INTRADAY' , TIME_SERIES_DAILY
             outputsize = 'compact',  # Values: compact, full
             apikey = os.environ["heuristik_alphavantage"], # Docs https://www.alphavantage.co/documentation/
             timedelta =  pd.Timedelta('7 days'), # Time window for tripple barrier method
             barriers = 0.05, # Vertical window for tripple barrier method
             force_download = False # False means use cached files
             ):
    """
    Downloading stock prices from AlphaVantage. Various options for different time-resolutions.
    """
    
    print('Getting prices for '+symbol+'.')
    query = ''.join(('https://www.alphavantage.co/query?&',
                     'datatype=','csv','&',
                    'function=',function,'&',
                    'outputsize=',outputsize,'&', 
                    'symbol=',symbol,'&',
                    'apikey=',apikey,
                     ''
                    ))
    
    str_timedelta = str(timedelta).replace(' ','_').replace(':','_')
    save_file = path+'price_data/'+function +'_'+ str_timedelta+'_'+str(barriers) + '_' +symbol + '.csv'
    if os.path.exists(save_file) and not force_download:
        print('Loading prices from file.')
        data = (pd.read_csv(save_file))
        data['time'] = data['time'].map(lambda s: pd.to_datetime(s))
        data = data.set_index('time')
    else:
        print('Downloading prices from AlphaVantage.')
        if function == 'TIME_SERIES_INTRADAY':
            query += ('&interval=' + '5min')
            df = pd.read_csv(query)
            if 'high' in df:
                df = df.drop(columns = ['high','low','close','volume'])
                df['timestamp'] = df['timestamp'].map(lambda s: pd.to_datetime(s).tz_localize('US/Eastern'))
                df = df.rename(columns={'open':'price','timestamp': 'time'})
                df = df.set_index('time')
            else:
                print('Error: Did not retrieve data.')
                
        elif function == 'SMA':
            query += ('&interval=' + '60min')
            query += ('&time_period='+'5')
            query += ('&series_type='+'open')
            df = pd.read_csv(query)
            if 'time' in df:
                df['time'] = df['time'].map(lambda s: pd.to_datetime(s).tz_localize('US/Eastern'))
                df = df.rename(columns={'SMA':'price'})
                df = df.set_index('time')
            else:
                print('Error: Did not retrieve data.')
                
        elif function == 'TIME_SERIES_DAILY':
            df = pd.read_csv(query)
            if 'timestamp' in df:
                df = df.drop(columns = ['high','low','close','volume'])
                df['timestamp'] = df['timestamp'].map(lambda s: pd.to_datetime(s).tz_localize('US/Eastern'))
                df = df.rename(columns={'open':'price','timestamp': 'time'})
                df = df.set_index('time')
            else:
                print('Error: Did not retrieve data.')
                
        df = df.iloc[::-1]
        data = df
        data['sentiment'] = apply_sentiment(data,timedelta = timedelta,barriers =barriers)
        data.to_csv(save_file)
    return save_file, data


def retrieve_symbols(symbol):
    """
    Resolving company symbol from AlphaVantage keyword search
    """
    
    query = 'https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords=' + symbol + '&apikey=' + os.environ["heuristik_alphavantage"]
    with urllib.request.urlopen(query) as url:
        data = json.loads(url.read().decode())
        
    if not 'bestMatches' in data.keys():
        output = []
    else:
        output = data['bestMatches']
        
    return output


def download_stocknews_page(name = 'TSLA', page = 1,api_key = os.environ["heuristik_stocknews"], 
                            page_size = 50,print_query = False):
    """
    Download one page from StockNews API.
    """
    if name == 'all':
        query_dict = {'section': 'alltickers',
                          'items':str(page_size),
                          'token':api_key,
                         'page': str(page)}
        query = 'https://stocknewsapi.com/api/v1/category?'
        for key in list(query_dict.keys()):
            query = query + key + '=' + query_dict[key]+'&'
        query = query[0:-1]
    else:
        query_dict = {'tickers': name,
                      'items':str(page_size),
                      'token':api_key,
                     'page': str(page)}
        #Assemble query:
        query = 'https://stocknewsapi.com/api/v1?'
        for key in list(query_dict.keys()):
            query = query + key + '=' + query_dict[key]+'&'
            
        query = query[0:-1]
        
    with urllib.request.urlopen(query) as url:
        data = json.loads(url.read().decode())

    if data['data'] ==[]:
        pages = 0
    else:
        pages = data['total_pages']
        
    if print_query:
        print(query)

    return pages, data['data']


def download_stocknews(name = 'TSLA', pages = 'all', api_key = os.environ["heuristik_stocknews"], 
                       download_path = '', save = True,print_query = False):
    """
    Download and save multiple pages from StockNews API.
    """
    
    assert pages == 'all' or (isinstance(pages,int) and pages>0), "Option pages should be 'all' or positive integer."
    start_time = time.time()
    number_of_pages, page_0 = download_stocknews_page(name = name, page = 1,api_key = api_key, print_query = print_query)
    if number_of_pages == 0:
        return []
    
    all_pages = convert_stocknews_data(page_0,name = name)
    if pages == 'all':
        pages = number_of_pages
        
    if number_of_pages < pages:
        pages = number_of_pages
        
    print('Downloading '+ str(pages) +' pages from StockNews. This may take a minute...')
    for i in range(2,pages):
        if time.time() < start_time + 1/12:
            time.sleep(1/12)
        _, page_i = download_stocknews_page(name = name, page = i,api_key = api_key)
        current_page = convert_stocknews_data(page_i,name = name)
        all_pages = all_pages.append(current_page)
        start_time = time.time()
        
    if save:
        all_pages.to_csv(download_path)
        
    return all_pages


def convert_stocknews_data(data,name = ''):
    """
    Cleans text contained in a dataframe.
    """
    
    df = pd.DataFrame({'time' : [],'symbol':[],'text' : [],'raw_text' : [],'url': [],'src_sentiment' : []})
    if name != 'all':
        clear_name = name_extraction(name)
    
    length_of_page = len(data)
    for i in range(0,length_of_page):
        if data[i]['text'] == None:
            text_body = data[i]['title']
        elif data[i]['title'] == None:
            text_body = data[i]['text']
        else:
            text_body = data[i]['title']+' '+ data[i]['text']
            
        if name != 'all':
            text_body = clean_raw_text(text_body,remove_name = clear_name)
            
        df = df.append({'time' : pd.to_datetime(data[i]['date']).tz_convert('US/Eastern'),
                        'symbol': ', '.join(data[i]['tickers']),
                        'source_name': data[i]['source_name'],
                        'text':text_body,
                        'raw_title':data[i]['title'], 
                        'raw_text': data[i]['text'],
                        'url': data[i]['news_url'],
                        'src_sentiment':data[i]['sentiment']}, ignore_index=True)
        
    df = df.set_index('time')
    return df

def process_company(asset_name,pages = 200, force_download = False, text_src = 'stocknews'
                   ,timewindow = '3 days', barriers = 0.05, data_version = '4'):
    """
    To process a company: downloads stock prices and news, and saves labeled dataframe to disk.
    """
    success = False
    timedelta = pd.Timedelta(timewindow)
    _, price_data = download_price(symbol = asset_name, function = 'TIME_SERIES_DAILY',
                                   outputsize ='full',barriers = barriers,timedelta = timedelta,
                                   force_download = force_download )
    text_name = name_extraction(asset_name)
    if text_src == 'stocknews':
        symbol = asset_name
        text_path = path+'text_data/' + 'stocknews_text_v'+data_version +'_'+ asset_name +'.csv'
        if os.path.exists(text_path) and force_download == False:
            print('Loading text from file: '+text_path)
            text = pd.read_csv(text_path)
            text['time'] = text['time'].map(lambda s: pd.to_datetime(s).tz_convert('US/Eastern'))
            text = text.set_index('time')
            text = text.iloc[::-1]
        else:
            text = download_stocknews(name = symbol, pages = pages, download_path = text_path )
            if len(text) == 0:
                print('No news found.')
                return
            else:
                text.to_csv(text_path)
    else:
        print('Error. Source not found.')
    if len(text) != 0:
        text = apply_sentiment_to_text(text, price_data)
        print('Save data.')
        str_timedelta = str(timedelta).replace(' ','_').replace(':','_')
        text.to_csv(path+'text_data/' +'stocknews_labeled_text_v'+data_version+'_' + asset_name +'_'+str_timedelta+'_'+str(barriers)+'.csv')
        success = True
    else:
        success = False
        
    return success


class data:
    def __init__(self,
                 timeframe = '3 days', 
                 data_version = '4', 
                 barriers =  '5%',
                 binary_sentiment = True
                 ):
        """
        Data manager class to load data from files or initiate download.
        
        Example use:
        
        data = heuristik.data(
                    timeframe = args.timeframe, 
                    data_version = args.data_version, 
                    barriers =  args.barriers,
                    binary_sentiment = args.binary_sentiment)
        df = data.retrieve(symbols = ['TWTR','AMD'],download=True)
        """
        
        self.path = path
        self.timeframe = timeframe
        self.data_version = data_version
        self.barriers = barriers
        self.data_version = data_version
        self.binary_sentiment = binary_sentiment
    
    def percent_to_float(self,s):
        assert isinstance(s,str) or isinstance(s,float) or isinstance(s,int ), 'Please provide str or float as input for barrier.'
        assert not s.startswith("-"), 'Provide positive barrier percentage.'
        if isinstance(s,float)or isinstance(s,int ):
            barrier = float(s/ 100)
        else:
            s = str(float(s.rstrip("%")))
            i = s.find(".")
            if i == -1:
                barrier =  int(s) / 100
            s = s.replace(".", "")
            i -= 2
            if i < 0:
                barrier =  float("." + "0" * abs(i) + s)
            else:
                barrier = float(s[:i] + "." + s[i:])
        return barrier

    def to_numerical_sentiment(self,rating):
        if isinstance(rating,str):
            if rating == 'Negative':
                if self.binary_sentiment:
                    return 1
                else:
                    print('nooo')
                    return -1
            elif rating == 'Positive':
                return 1
            else:
                return 0
        if (rating == rating) : #check for 'NaN'
            rating = int(rating)
            if rating == -1:
                if self.binary_sentiment:
                    return 1
                else:
                    return -1
            elif rating == 0:
                return 0
            else:
                return 1

    def human_format(self,num):
        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0
        # add more suffixes if you need them
        return '%.1f%s' % (num, ['', 'k', 'M', 'G', 'T', 'P'][magnitude])


    def retrieve(self,
                symbols,
                download = False):
        
        path = self.path
        timeframe = self.timeframe
        data_version = self.data_version
        barriers = self.barriers
        data_version = self.data_version
        
        pd_timeframe = pd.Timedelta(timeframe)
        greater_than_zero = pd_timeframe.round('1s')>pd.Timedelta('0')
        assert greater_than_zero, 'Please provide valid timeframe > 1 second.' 
        if not path[-1]=='/':
            path = path+'/'
            
        if isinstance(symbols,str):
            symbols = symbols.replace(' ','').split(',')

        str_timedelta = str(pd_timeframe).replace(' ','_').replace(':','_')
        str_barriers = str(self.percent_to_float(barriers))
        file_name = 'stocknews_labeled_text_v'+data_version+'_'
        data_paths = []
        if path[0]== '~':
            print('Warning: expanding ~ to home directory.')
            path = os.path.expanduser(path)
            print('Path: '+ path)

        for symbol in symbols:
            data_paths.append(path+'text_data/'+file_name+symbol+'_'+str_timedelta+'_'+str_barriers+'.csv')

        data_available = []
        for i, data_path in enumerate(data_paths):
            if not os.path.exists(data_path):
                if download == True:
                    print('Downloading data.')
                    success = process_company(symbols[i],pages = 200, force_download = False, text_src = 'stocknews',timewindow = self.timeframe, barriers = float(str_barriers),data_version = self.data_version)
                    print('Data downloaded for ', symbols[i],'.')
                    data_available.append(success)
                else:
                    print('Data unavailable for symbol '+symbols[i]+'. Skipping.')
                    data_available.append(False)
            else:
                data_available.append(True)

        if not any(data_available):
            return ''
        
        if data_available[0]:
            df = pd.read_csv(data_paths[0])
        else:
            df = pd.DataFrame([])

        for i, data_path in enumerate(data_paths[1:]):
            if data_available[i+1]:
                df = df.append(pd.read_csv(data_path))

        df['sentiment'] = df.sentiment.apply(self.to_numerical_sentiment)
        df= df.rename(columns={'sentiment': 'price_sentiment'})
        df['src_sentiment'] = df.src_sentiment.apply(self.to_numerical_sentiment)
        #df = df.reset_index()
        #df = df.drop(columns=['index'])
        print('Successfully retrieved',self.human_format(len(df)),'samples.')
        return df