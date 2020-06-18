""" PyTorch Deep Learning for Stock Sentiment """

# Import Packages
import time
import os
import argparse

import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from focal_loss import FocalLoss

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import wandb


# Path to files is set in ./config.py
model_path = os.environ['heuristik_data_path']


# Define Data loader utility
class TextDataset(Dataset):
    """ Converts text items to dictionary callable by torch.utils.data.DataLoader
    """
    
    def __init__(self, 
                 reviews, 
                 targets, 
                 tokenizer, 
                 max_len, 
                 batch_size, 
                 seed):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.batch_size = batch_size
        self.seed = seed

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]
        encoding = self.tokenizer.encode_plus(
          review,
          add_special_tokens=True,
          max_length=self.max_len,
          return_token_type_ids=False,
          pad_to_max_length=True,
          return_attention_mask=True,
          return_tensors='pt',
        )
        return {
          'review_text': review,
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          'targets': torch.tensor(target, dtype=torch.long)
        }


# Utility to convert pandas dataframe to train/val/test dataloaders
class prepare_loaders():
    
    def __init__(self, df, max_len, batch_size, seed, bert_model_name,test_size=0):
        """ Converts  pandas dataframe to train/val/test dataloaders. 
        Parameters: 
        arg1 (pd.Dataframe): Dataframe containing the columns 'text' and 'price_sentiment'. 
        arg2 (int): Number of words to keep in each row of 'text'
        arg3 (int): Batch size. Fit to your system.
        arg4 (int): Set random seed for reproducibility
        arg5 (str): String of Huggingface BERT model for tokenizer, see https://github.com/google-research/bert
        arg6 (float): relative train / test&validation set split proportion
        """
        
        self.batch_size = batch_size
        self.seed = seed
        self.test_size = test_size
        if test_size == 0:
            self.df_train = df
        else:
            self.df_train, self.df_test = train_test_split(df, test_size=test_size, random_state=self.seed)
            self.df_val, self.df_test = train_test_split(self.df_test, test_size=0.5, random_state=self.seed)
            
        tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        if 'price_sentiment' not in self.df_train:
            self.df_train['price_sentiment'] = 0
            
        self.ds_train = TextDataset(
            reviews=self.df_train.text.to_numpy(),
            targets=self.df_train.price_sentiment.to_numpy(),
            tokenizer=tokenizer,
            max_len=max_len,
            seed = seed,
            batch_size = batch_size
            )
        if test_size != 0:
            self.ds_val = TextDataset(
                reviews=self.df_val.text.to_numpy(),
                targets=self.df_val.price_sentiment.to_numpy(),
                tokenizer=tokenizer,
                max_len=max_len,
                seed = seed,
                batch_size = batch_size
                )
            self.ds_test = TextDataset(
                reviews=self.df_test.text.to_numpy(),
                targets=self.df_test.price_sentiment.to_numpy(),
                tokenizer=tokenizer,
                max_len=max_len,
                seed = seed,
                batch_size = batch_size
                )

    def train_val_test(self):
        
        train_data_loader = DataLoader(self.ds_train,batch_size=self.batch_size,num_workers=4)
        if self.test_size != 0:
            val_data_loader = DataLoader(self.ds_val,batch_size=self.batch_size,num_workers=4)
            test_data_loader = DataLoader(self.ds_test,batch_size=self.batch_size,num_workers=4)
            return train_data_loader, val_data_loader, test_data_loader
        else:
            return train_data_loader, [], []


# Define and load the model
def load_model(model_name, n_classes, pretrained ='None', path = model_path,rezero = False):
    """ Define and load the SentimentClassifier model. By default a pre-trained Huggingface BERT is loaded.
    Model consists of BERT encoder + deep ReZero linear network, see https://arxiv.org/pdf/2003.04887.pdf
    
    Parameters: 
    model_name (str): String of Huggingface BERT model for BertModel, see https://github.com/google-research/bert
    n_classes (int): Integer defining the number of classes for sentiment analysis.
    pretrained (str): Which local pretrained model to load. Default 'None' does not load anything.
    path (str): Path to directory where pretrained models are stored.
    """ 

    class SentimentClassifier(nn.Module):
        
        def __init__(self, n_classes):
            super(SentimentClassifier, self).__init__()
            self.bert = BertModel.from_pretrained(model_name)
            self.drop = nn.Dropout(p=0.3)
            self.width = self.bert.config.hidden_size
            if rezero:
                depth = 5
                self.resweight = nn.Parameter(torch.zeros(depth), requires_grad=True)
                self.linear_layers = nn.ModuleList([nn.Linear(self.width, self.width) for i in range(depth)])
            
            self.out = nn.Linear(self.width, n_classes)

        def forward(self, input_ids, attention_mask):
            _, pooled_output = self.bert(
              input_ids=input_ids,
              attention_mask=attention_mask
            )
            output = self.drop(pooled_output)
            if rezero:
                for i, j in enumerate(self.linear_layers):
                    # ReZero architecture, see https://arxiv.org/pdf/2003.04887.pdf
                    output = output + self.resweight[i] *  torch.relu(self.linear_layers[i](output))
            
            output = self.out(output)
            return output

    model = SentimentClassifier(n_classes)

    if pretrained != 'None':
        print('Loading pre-trained model: '+path+'/'+pretrained+'.pth')
        if not torch.cuda.is_available():
            map_location='cpu'
            model.load_state_dict(torch.load(path+'/'+pretrained+'.pth', map_location=map_location))
        else:
            model.load_state_dict(torch.load(path+'/'+pretrained+'.pth'))
    
    return model


# Train the model for one epoch
def train_epoch(
    epoch,
    model, 
    data_loader, 
    loss_fn, 
    optimizer, 
    device, 
    scheduler,
    print_freq = 50
    ):
    """ Train the model of a single epoch, and log progress with wandb.
    Parameters: 
    epoch (int): Which epoch number is being trained (for logging).
    model (PyTorch model): Provide a trainable model.
    data_loader (PyTorch Dataloader): Provide a data loader
    loss_fn (function): Loss function
    optimizer (PyTorch optimizer): Which optimizer to use
    scheduler (PyTorch scheduler): Update optimizer parameters at each iteration
    print_freq (int): Log frequency for wandb
    """ 
    
    model = model.train()
    train_loss = 0
    total = 0
    correct_predictions = 0
    start_time = time.time()
    total_batches = len(data_loader)
    for batch_idx, d  in enumerate(data_loader): 
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        outputs = model(input_ids=input_ids,attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total += targets.size(0)
        correct_predictions += torch.sum(preds == targets)
        train_loss += loss.item()
        if (1+ batch_idx) % print_freq == 0:
            acc = 100.*float(correct_predictions)/float(total)
            print('Batch: {:3.0f}/{:3.0f}. Train Loss: {:0.3f}  Acc: {:2.0f} '.format(
                  batch_idx+1,total_batches,train_loss/(batch_idx+1),acc))
            lr = list(optimizer.param_groups)[0]['lr']
            wandb.log({'epoch': epoch, 'lr':lr , 'train_loss':train_loss/(batch_idx+1), 'train_acc':acc})
            
    print('Time/epoch: {:3.0f}'.format(time.time()-start_time))
    acc = 100.*float(correct_predictions)/float(total)
    lr = list(optimizer.param_groups)[0]['lr']
    wandb.log({'epoch': epoch, 'lr':lr , 'train_loss':train_loss/(batch_idx+1), 'train_acc':acc})
    print('-----------')
    return acc, train_loss / (1+batch_idx)


# Define function to compute F1 score for binary classification
def f1_loss(y_true:torch.Tensor, y_pred:torch.Tensor, is_training=False) -> torch.Tensor:
    ''' Calculate F1 score.
    '''
    
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2
    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    epsilon = 1e-7
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    f1 = f1.cpu()
    f1.numpy()
    return float(f1)


# Evaluate and log the model
def eval_model(epoch, model, data_loader, loss_fn, device):
    """ Evaluates the model: outputs accuracy, validation loss and f1 score.
    """
    
    model = model.eval()
    val_loss = 0
    total = 0
    correct_predictions = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_idx, d in enumerate(data_loader):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            outputs = model(input_ids=input_ids,attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            all_preds.append(preds)
            all_targets.append(targets)
            loss = loss_fn(outputs, targets)
            total += targets.size(0)
            val_loss += loss.item()
            correct_predictions += torch.sum(preds == targets)
            
    acc = 100.*float(correct_predictions)/float(total)
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    f1 = f1_loss(all_targets,all_preds)
    print('Epoch: {:2.0f} Val. Loss: {:0.3f}.  Acc: {:2.0f}. F1:  {:0.3f}'.format(epoch,val_loss/(batch_idx+1),acc,f1))
    wandb.log({'epoch': epoch, 'val_loss': val_loss/(batch_idx+1), 'val_acc':acc, 'val_F1': f1})
    return acc, val_loss, f1


# Return predictions from model
def get_predictions(model, data_loader, device, n_examples = 50,force_n_examples = False, use_targets = False):
    """ Runs inference on a given dataloader to return n_examples high confidence predictions.
    """
    
    model = model.eval()
    all_preds = []
    if use_targets == True:
        correct_predictions = 0
        all_targets = []
        
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            if use_targets == True:
                targets = d["targets"].to(device)
            outputs = model(input_ids=input_ids,attention_mask=attention_mask)
            preds = torch.nn.functional.softmax(outputs,dim = 1)[:,1]
            all_preds.extend(preds.to('cpu').tolist())
            if use_targets == True:
                all_targets.extend(targets.to('cpu').tolist())
    
    strongest_preds_index = sorted(range(len(all_preds)), key=lambda x: all_preds[x])[-n_examples:]
    strongest_preds = all_preds
    df = pd.DataFrame({'prediction_score': all_preds}).round(decimals=2)
    for i in range(len(strongest_preds)):
        if i not in strongest_preds_index:
            strongest_preds[i] = 0
        elif strongest_preds[i]<0.5 and force_n_examples == False:
            strongest_preds[i] = 0
        else:
            strongest_preds[i] = 1
            
    if use_targets == True:
        correct_predictions = sum([(strongest_preds[i]==1 and strongest_preds[i] == all_targets[i]) for i in strongest_preds_index])
        number_of_predictions = sum(strongest_preds)
        correct_ratio = 0
        if number_of_predictions > 0:
            correct_ratio = correct_predictions/number_of_predictions
        print('Correct ratio:', correct_ratio)
        
    df['predictions'] = pd.DataFrame({'prediction_score':strongest_preds})
    if use_targets == True:
        filename = str(n_examples)+'_best_predictions_acc'
        wandb.log({filename: correct_ratio})
        return df,correct_ratio
    else:
        return df, None
    
    
# Train the model for several epochs.
def train_model(epochs,model,dl_train,device, dl_val = None, path = model_path, file_name = '', print_freq = 50):
    """ Trains the model for several epochs and saves the model that has best strong confidence predictions
    """
    
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(dl_train) * epochs
    if dl_val != None:
        evaluate = True
    else:
        evaluate = False
        
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0,num_training_steps=total_steps)
    loss_fn = FocalLoss(gamma=4,class_num =2).to(device)#FocalLoss().to(device)
    best_correct_ratio = 0
    correct_ratio = 0
    if evaluate:
        _, correct_ratio = get_predictions(model,dl_val,device, n_examples = 50,force_n_examples = True, use_targets = True)
        eval_model(0,model,dl_val,loss_fn, device)
        
    epoch = 0
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        print('-' * 10)
        train_acc, train_loss = train_epoch(epoch,model,dl_train,loss_fn, optimizer, 
                                            device, scheduler,print_freq = print_freq)
        if evaluate:
            val_acc, val_loss, f1 = eval_model(epoch,model,dl_val,loss_fn, device)
            _, correct_ratio = get_predictions(model,dl_val,device, n_examples = 50,
                                               force_n_examples = True, use_targets = True)

        if correct_ratio > best_correct_ratio and file_name != '':
            print('Save. Epoch: ',epoch+1)
            torch.save(model.state_dict(), path+'/'+file_name)
            best_correct_ratio = correct_ratio
            
            