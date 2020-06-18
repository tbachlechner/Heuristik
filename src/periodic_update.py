""" Provide periodic inference on general news for server latency """

# Import Packages
import sys
import os
import json

<<<<<<< HEAD
sys.path.insert(1, '/home/ubuntu/ai/Heuristik/src')
os.environ["heuristik_data_path"] = '/home/ubuntu/ai/Heuristik/data'
=======
# Path to files is set in ./config.py
sys.path.insert(1, '/home/thomas/ai/Heuristik/src')
os.environ["heuristik_data_path"] = '/home/thomas/ai/Heuristik/data'
>>>>>>> 180993bb1db25289a40f80d00f5a6ef63840d791

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
    os.mkdir(history_pazth)

general_filename = history_path+'/'+'all'+'.pkl'
rn = heuristik.recent_news_class(parse_samples = parse_samples)
while True:
    df = rn.recent_news(name = 'all')
    df.to_pickle(general_filename)
    time.sleep(interval)
