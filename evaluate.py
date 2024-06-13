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
    # load the model
    model = BertForSequenceClassification.from_pretrained('HEDGE\output\IMDB')
    tokenizer = BertTokenizer.from_pretrained('HEDGE\output\IMDB', do_lower_case=True)

    def model_predict(x: np.ndarray) -> np.ndarray:
        probs = np.zeros((x.shape[0], 2))
        for i in range(x.shape[0]):
            input_ids = torch.tensor(np.hstack([101, x[i, :, 0], 102]).reshape(1,-1)) # cls, x, sep
            output = model(**{'input_ids': input_ids})[0][0]
            probs[i, :] = F.softmax(output, dim=0).detach().numpy()
        
        return np.array([probs[:, 1]])

    f = lambda x: model_predict(x)

    dirs = [x[0] for x in os.walk(path)][1:]
    nsamples = len(dirs)
    aopc = np.zeros((nsamples, 10))
    log_odds = np.zeros((nsamples, 10))
    coherence = np.zeros((nsamples, 1))
    span_len = np.zeros((nsamples, 1))

    for ndir in tqdm.tqdm(range(nsamples)):
        folder = dirs[ndir]
        # load metadata
        metadata = json.load(open(f'{folder}/metadata.json', 'r'))
        # load shap values
        importance = pd.read_csv(f'{folder}/importance.csv')['importance'][::-1]
        # sort the importance with reversed order since timeshap uses the opposite to HEDGE 
        inds = sorted(range(len(importance)), key=lambda i: importance[i], reverse=True)
        # prepare the data 
        data = metadata['sentence']
        data = tokenizer.encode(data, add_special_tokens=False)
        data = np.expand_dims(np.array(data), axis=[0,2])
        pred = float(metadata['prediction'])
        label = round(pred, 0)

        # calculate the AOPC and log odds
        for i, k in enumerate(np.arange(0.05, 0.55, 0.05)):
            del_inds = inds[:int(np.ceil(k*data.shape[1]))]
            data[0, del_inds, 0] = tokenizer.mask_token_id
            data[0, del_inds, 0] = tokenizer.encode(metadata['mask_token'])
            perturbed_pred = f(data)[0][0]
            
            aopc[ndir, i] = pred - perturbed_pred if label == 1 else (1-pred) - (1-perturbed_pred)
            log_odds[ndir, i] = np.log10(perturbed_pred/pred) if label == 1 else np.log((1-perturbed_pred)/(1-pred))
        
        if method in ['HEDGE', 'MöbiusHEDGE']:
            # calculate coherence score
            max_inter_fea_set = metadata['max interaction set']
            span_len[ndir] = len(max_inter_fea_set)

            Q = 100
            coherence[ndir] = 0
            for i in range(Q):
                # perturb the data
                data = metadata['sentence']
                data = tokenizer.encode(data, add_special_tokens=False)
                data = np.expand_dims(np.array(data), axis=[0,2])
                shuffled_set = np.random.choice(max_inter_fea_set, len(max_inter_fea_set), replace=False)
                data[0, max_inter_fea_set, 0] = data[0, shuffled_set, 0]

                # calculate difference in prediction
                perturbed_pred = f(data)[0][0]
                coherence[ndir] += abs(pred-perturbed_pred)
            
            coherence[ndir] /= Q
    
    # save results to file
    columns = [f'{i}%' for i in range(5, 55, 5)]
    aopc = pd.DataFrame(aopc, columns=columns)
    aopc.mean().to_csv(f'{path}/aopc.csv', index=False, header=False)
    log_odds = pd.DataFrame(log_odds, columns=columns)
    log_odds.mean().to_csv(f'{path}/log_odds.csv', index=False, header=False)
    if method in ['HEDGE', 'MöbiusHEDGE']:
        coherence = pd.DataFrame(coherence, columns=['coherence'])
        coherence.mean().to_csv(f'{path}/coherence.csv', index=False, header=False)
        span_len = pd.DataFrame(span_len, columns=['span_len'])
        pd.DataFrame({'mean': span_len.mean(), 'std': span_len.std()}).to_csv(f'{path}/span_len.csv', index=False)

if __name__ == '__main__':
    calc_metrics(method='TimeSHAP')