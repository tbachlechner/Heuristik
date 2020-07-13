""" Provide periodic inference on general news for server latency """

# Import Packages
import sys
import os
import json



# Path to files is set in ./config.py
sys.path.insert(1, '/home/ubuntu/ai/Heuristik/src')
os.environ["heuristik_data_path"] = '/home/ubuntu/ai/Heuristik/data'


import heuristik
import time
import os
import pandas as pd

#Interval in s for updates (1/hr)
interval = 60*60
# number of news items to parse
parse_samples = 2000
history_path = os.environ["heuristik_data_path"] + '/call_history'
if not os.path.exists(history_path):
    os.mkdir(history_path)

general_filename = history_path+'/'+'all'+'.pkl'
rn = heuristik.recent_news_class(parse_samples = parse_samples)
while True:
    df = rn.recent_news(name = 'all')
    df.to_pickle(general_filename)
    time.sleep(interval)
