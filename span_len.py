import numpy as np 
import pandas as pd 
import torch
import torch.nn.functional as F
import datetime 
import json
import os
import sys
import tqdm

# LSTM for sequence classification in the IMDB dataset
from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer)
from keras.saving import load_model

np.random.seed(37)

def calc_metrics(method='MöbiusHEDGE', dataset='SST-2'):
    path = f'experiments/{method}/{dataset}/'

    dirs = [x[0] for x in os.walk(path)][1:]
    nsamples = len(dirs)

    span_len = np.zeros((nsamples, 1))

    for ndir in tqdm.tqdm(range(nsamples)):
        folder = dirs[ndir]
        metadata = json.load(open(f'{folder}/metadata.json', 'r'))
        max_inter_fea_set = metadata['max interaction set']
        span_len[ndir] = len(max_inter_fea_set)

    span_len = pd.DataFrame(span_len, columns=['span_len'])
    pd.DataFrame({'mean': span_len.mean(), 'std': span_len.std()}).to_csv(f'{path}/span_len.csv', index=False)

if __name__ == '__main__':
    calc_metrics(method='MöbiusHEDGE')