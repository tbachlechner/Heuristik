import sys
sys.path.insert(1, '../src')
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

rn = heuristik.recent_news_class(parse_samples = 200)

#Title
#st.title('Heuristik.')
image = Image.open('heuristik_logo.jpeg')
st.image(image,width = 300)
st.header('News indicative of imminent stock price volatility.')

#symbol = st.selectbox(   'Retrieve news for (e.g. AAPL, TSLA, ...)', ('all', 'AAPL', 'TSLA'))
symbol = st.text_input('Retrieve news for... (e.g. AAPL, TSLA, ...)','General')
filename = './'+symbol+'.pkl'

# Capitalize:
symbol = ''.join([letter.capitalize() for letter in symbol])


if symbol == 'GENERAL':
    symbol = 'all'

if not os.path.exists(filename):
    df = rn.recent_news(name = symbol)
    df.to_pickle(filename)
elif time.time() - os.stat(filename).st_mtime >3600:
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



for i in range(0,20):
    if raw_titles[i] != raw_texts[i]:
        print_text = raw_texts[i]
    else:
        print_text = ''
    body = '### {number} | {symbol} | Date: {date} \n  **[{title}]({url})** \n {text}'.format(number = str(i+1),symbol = symbols[i], title = raw_titles[i],date= indices[i],url = urls[i],text=print_text)
    st.markdown(body)
    st.write('Score:  ' +str(round(100 * scores[i] ))+'/100' )
    

    
