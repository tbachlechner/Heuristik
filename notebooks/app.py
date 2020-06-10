import sys
sys.path.insert(1, '../src')
import heuristik
import time
import os
import pandas as pd
import streamlit as st

import threading
from threading import Thread
from threading import Event

class RepeatingTimer(Thread):
    def __init__(self, interval_seconds, callback):
        super().__init__()
        self.stop_event = Event()
        self.interval_seconds = interval_seconds
        self.callback = callback

    def run(self):
        while not self.stop_event.wait(self.interval_seconds):
            self.callback()

    def stop(self):
        self.stop_event.set()

def dothis():
    filename = './'+'all'+'.pkl'
    print('Helooo')
    #rn = heuristik.recent_news_class()
    #df_all = rn.recent_news(name = 'all')
    #df_all.to_pickle(filename)

r1.stop()
r1 = RepeatingTimer(120, dothis)


r1.start()

rn = heuristik.recent_news_class()

#Title
st.title('Heuristik.')
st.header('News indicative of imminent stock price volatility.')

symbol = st.selectbox(   'Retrieve news for (e.g. AAPL, TSLA, ...)', ('all', 'AAPL', 'TSLA'))
filename = './'+symbol+'.pkl'

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
    

    