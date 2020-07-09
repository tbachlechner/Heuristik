
import sys
import os
import re
import json
sys.path.insert(1, '/home/ubuntu/ai/Heuristik/src')
os.environ["heuristik_data_path"] = '/home/ubuntu/ai/Heuristik/data'

import heuristik
import time
import os
import pandas as pd
import streamlit as st

from PIL import Image



hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

time_window = 60*60*12
non_general_factor = 24
parse_samples = 200

items = 20

@st.cache(suppress_st_warning=True, show_spinner=False,allow_output_mutation=True)
def call_rn():
    rn = heuristik.recent_news_class(parse_samples = parse_samples)
    return rn

def symbol_resolution(symbol):
    symbol = re.sub('[^a-zA-Z]+', '',symbol)
    symbol = symbol.replace(" ", "%20")

    # Capitalize:
    symbol = ''.join([letter.capitalize() for letter in symbol])

    if symbol == '':
        resolved_symbol_list = []
    else:
        resolved_symbol_list = heuristik.retrieve_symbols(symbol)

    if len(resolved_symbol_list) == 0 or symbol == 'GENERAL':
        resolved_symbol = 'GENERAL'
    else:
        resolved_symbol = resolved_symbol_list[0]['1. symbol']

    if len(resolved_symbol_list)>1 and not symbol == 'GENERAL':
        if not resolved_symbol_list[0]['9. matchScore'] ==  '1.0000':
            st.warning("Input may be ambiguous. Using symbol: "+resolved_symbol_list[0]['1. symbol']+', '+resolved_symbol_list[0]['2. name'])

    if len(resolved_symbol_list)==0 and not symbol == 'GENERAL':
        st.warning("Symbol not found. Showing general news.")
    
    if resolved_symbol == 'GENERAL':
        resolved_symbol = 'all'
    return resolved_symbol


rn = call_rn()



#Title
#st.title('Heuristik.')
image = Image.open(os.environ["heuristik_data_path"]+'/heuristik_logo.jpeg')
st.image(image,width = 300)
st.header('News indicative of imminent stock price volatility.')

#symbol = st.selectbox(   'Retrieve news for (e.g. AAPL, TSLA, ...)', ('all', 'AAPL', 'TSLA'))
symbol = st.text_input('Retrieve news for... (e.g. AAPL, TSLA, ...)','General')

symbol = symbol_resolution(symbol)


history_path = os.environ["heuristik_data_path"] + '/call_history'
if not os.path.exists(history_path):
    os.mkdir(history_path)

filename = history_path+'/'+symbol+'.pkl'


with st.spinner('Downloading and processing '+ str(parse_samples) + ' news items...'):
    if not os.path.exists(filename):
        df = rn.recent_news(name = symbol)
        df.to_pickle(filename)
    elif time.time() - os.stat(filename).st_mtime > time_window and symbol == 'all':
        df = rn.recent_news(name = symbol)
        df.to_pickle(filename)
    elif time.time() - os.stat(filename).st_mtime > non_general_factor * time_window:
        df = rn.recent_news(name = symbol)
        df.to_pickle(filename)
    else:
        df = pd.read_pickle(filename) 

indices = list(df.index)
symbols =  list(df['symbol'])
raw_titles = list(df['raw_title'])
raw_texts = list(df['raw_text'])
urls = list(df['url'])
scores =  list(df['prediction_score'])
num_items = len(df)
if items > num_items:
    items = num_items


for i in range(0,items):
    if raw_titles[i] != raw_texts[i]:
        print_text = raw_texts[i]
    else:
        print_text = ''
    body = '### {number} | {symbol} | Date: {date} \n  **[{title}]({url})** \n {text}'.format(number = str(i+1),symbol = symbols[i], title = raw_titles[i],date= indices[i],url = urls[i],text=print_text)
    st.markdown(body)
    st.write('Score:  ' +str(round(100 * scores[i] ))+'/100' )
    
