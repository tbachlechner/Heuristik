
import os
import json

def config(data_path = os.environ["heuristik_data_path"], stocknews_key = None, alphavantage_key = None):
    global config_dict
    


    data_path = os.path.expanduser(data_path)
    
    keys_path = data_path + '/keys.json'
    keys_file_exists = os.path.exists(keys_path)

    if keys_file_exists:
        keys_file = open(keys_path,'r')
        keys_dict = json.load(keys_file)

    if stocknews_key == None and keys_file_exists:
        stocknews_key = keys_dict['stocknews']
    else:
        print('Enter StockNews API key:')
        stocknews_key = input()
    
    if alphavantage_key == None and keys_file_exists:
        alphavantage_key = keys_dict['alphavantage']
    else:
        print('Enter AlphaVantage API key:')
        alphavantage_key = input()
    

    if not keys_file_exists:
        print('Save keys? (y/n)')
        save_keys = input()
        if save_keys == 'y':
            with open(keys_path, 'w') as outfile:
                json.dump({'alphavantage': alphavantage_key, 'stocknews': stocknews_key}, outfile)
            print('Saved keys at '+ keys_path)

        
    config_dict = {'data_path': data_path,
                   'stocknews': stocknews_key,
                   'alphavantage': alphavantage_key}
    os.environ["heuristik_stocknews"] = stocknews_key
    os.environ["heuristik_alphavantage"] = alphavantage_key
    os.environ["heuristik_data_path"] = data_path

    
    return config_dict
