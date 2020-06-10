from data_manager import *
from training import *

def recent_news(items = 20,
                parse_samples = 200,
                name = 'all',
                model_name = 'bert-base-cased',
                pretrained = 'BaseModel_large',
                path = '../data/',
                device = 'cpu',
                max_len = 100,
                batch_size = 32,
                seed = 1):
    model = load_model(model_name = model_name, n_classes = 2, pretrained = pretrained,path = path)
    model = model.to(device)
    pages = round(parse_samples/50)
    if pages == 0:
        pages = 1
    if items > 50 * pages:
        pages = round(items/50) + 1
        
    df_eval = download_stocknews(name = name, pages = pages, path = '', save = False ,print_query=False)

    eval_loaders = prepare_loaders(df_eval,
                              bert_model_name = model_name, 
                              max_len = max_len, 
                              batch_size = batch_size ,
                              seed = seed,
                            test_size = 0)

    dl_eval, _, _ = eval_loaders.train_val_test()
    print('Analyzing relevance...')
    df_pred, _ = get_predictions(model, dl_eval, device,use_targets= False)


    df_eval_pred = df_eval
    df_pred.index = df_eval_pred.index
    df_eval_pred['predictions']=df_pred['predictions']
    df_eval_pred['prediction_score']=df_pred['prediction_score']
    df_eval_pred = df_eval_pred.sort_values(by = 'prediction_score', ascending=False)
    print('-'*100)
    for i in range(0,items):
        print('Date: ', list(df_eval_pred.index)[i])
        print('Company: ', list(df_eval_pred['symbol'])[i])
        print('News:')
        print(list(df_eval_pred['raw_title'])[i])
        print(list(df_eval_pred['raw_text'])[i])
        print('URL:')
        print(list(df_eval_pred['url'])[i])
        print('Relevance score (0-100):')
        print('{:2.0f}'.format(100*list(df_eval_pred['prediction_score'])[i]))
        print('-'*100)
    return df_eval_pred


class recent_news_class:
    def __init__(self,
                 items = 20,
                 parse_samples = 200,
                 model_name = 'bert-base-cased',
                 pretrained = 'BaseModel_large',
                 path = '../data/',
                 device = 'cpu',
                 max_len = 100,
                 batch_size = 32,
                 seed = 1
                 ):
        self.model_name = model_name
        self.max_len = max_len
        self.batch_size = batch_size
        self.seed = seed
        self.model = load_model(model_name = model_name, n_classes = 2, pretrained = pretrained,path = path)
        self.model = self.model.to(device)
        self.pages = round(parse_samples/50)
        self.items = items
        self.device = device
        self.items = items
        if self.pages == 0:
            self.pages = 1
        if self.items > 50 * self.pages:
            self.pages = round(self.items/50) + 1
    
    def recent_news(self,name):
        df_eval = download_stocknews(name = name, pages = self.pages, path = '', save = False , print_query=False)

        eval_loaders = prepare_loaders(df_eval,
                                  bert_model_name = self.model_name, 
                                  max_len = self.max_len, 
                                  batch_size = self.batch_size ,
                                  seed = self.seed,
                                  test_size = 0)

        dl_eval, _, _ = eval_loaders.train_val_test()
        print('Analyzing relevance...')
        df_pred, _ = get_predictions(self.model, dl_eval, self.device,use_targets= False)


        df_eval_pred = df_eval
        df_pred.index = df_eval_pred.index
        df_eval_pred['predictions']=df_pred['predictions']
        df_eval_pred['prediction_score']=df_pred['prediction_score']
        df_eval_pred = df_eval_pred.sort_values(by = 'prediction_score', ascending=False)
        print('-'*100)
        for i in range(0,self.items):
            print('Date: ', list(df_eval_pred.index)[i])
            print('Company: ', list(df_eval_pred['symbol'])[i])
            print('News:')
            print(list(df_eval_pred['raw_title'])[i])
            print(list(df_eval_pred['raw_text'])[i])
            print('URL:')
            print(list(df_eval_pred['url'])[i])
            print('Relevance score (0-100):')
            print('{:2.0f}'.format(100*list(df_eval_pred['prediction_score'])[i]))
            print('-'*100)
        return df_eval_pred