import numpy as np 
import pandas as pd 
import torch
import torch.nn.functional as F
import datetime 
import json
import os
import sys

# LSTM for sequence classification in the IMDB dataset
from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer)
from keras.saving import load_model

np.random.seed(37)

def calc_metrics(model_name='BERT', dataset='IMDB'):
    path = 'experiments/HEDGE/bert/'
    # load the model
    if model_name == 'BERT':
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
    elif model_name == 'LSTM':
        print('LSTM not implemented yet')
        sys.exit()
        model = load_model("models/LSTM.h5")
        f = lambda x: model.predict(x)
    elif model_name == 'CNN':
        print('CNN not implemented yet')
        sys.exit()
    else:
        raise ValueError('Model not supported')

    nsamples = len(os.listdir(path))
    aopc = np.zeros((nsamples, 10))
    log_odds = np.zeros((nsamples, 10))

    # load the results from file
    for entry in os.scandir(path):
        if entry.is_dir():
            folder = entry.name
            # load metadata
            metadata = json.load(open(f'{path}{folder}/metadata.json', 'r'))
            # load phi
            phi = pd.read_csv(f'{path}{folder}/shap_values.csv')
            phi = np.array(phi['shapley value'])
            # sort the indices by importance
            inds = sorted(range(len(phi)), key=lambda i: abs(phi[::-1][i]), reverse=True)
            # prepare the data
            sentence_tag = metadata['sentence tag']  
            data = metadata['input sentence']
            data = tokenizer.encode(data, add_special_tokens=False)
            data = np.expand_dims(np.array(data), axis=[0,2])
            pred = float(metadata['prediction'])

            # calculate the AOPC and log odds
            for i, k in enumerate(np.arange(0.05, 0.55, 0.05)):
                del_inds = inds[:int(np.ceil(k*data.shape[1]))]
                data[0, del_inds, 0] = tokenizer.mask_token_id
                perturbed_pred = f(data)[0][0]
                aopc[sentence_tag, i] = abs(pred-perturbed_pred)
                log_odds[sentence_tag, i] = np.log(perturbed_pred/pred) if round(pred, 0) == 1 else np.log((1-perturbed_pred)/(1-pred))
    
    # save results to file
    columns = [f'{i}%' for i in range(5, 55, 5)]
    aopc = pd.DataFrame(aopc, columns=columns)
    aopc.mean().to_csv(f'experiments/HEDGE/bert/aopc_shapley.csv', index=False, header=False)
    log_odds = pd.DataFrame(log_odds, columns=columns)
    log_odds.mean().to_csv(f'experiments/HEDGE/bert/log_odds_shapley.csv', index=False, header=False)

if __name__ == '__main__':
    calc_metrics()
        
        

