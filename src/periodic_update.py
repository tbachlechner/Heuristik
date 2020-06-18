import sys
import os
import json

sys.path.insert(1, '/home/ubuntu/ai/Heuristik/src')
os.environ["heuristik_data_path"] = '/home/ubuntu/ai/Heuristik/data'

import heuristik
import time
import os
import pandas as pd

interval = 60*60
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
