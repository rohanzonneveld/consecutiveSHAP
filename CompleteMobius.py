import numpy as np 
import pandas as pd 
import torch
import torch.nn.functional as F
import datetime 
import json
import os

# LSTM for sequence classification in the IMDB dataset
from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer)

from utils import CompleteMobiusKernel

np.random.seed(37)

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

# load the data
dataset = 'SST-2'
if dataset == 'IMDB':
    X_test = pd.read_csv('HEDGE\dataset\IMDB/test.tsv', sep='\t', header=0)
elif dataset == 'SST-2':
    X_test = pd.read_parquet('HEDGE\dataset\SST-2/test-00000-of-00001.parquet')
    

for i in range(831):#len(X_test['sentence'])):
    
    start = datetime.datetime.now()
    sentence_tag = i
    input_sentence = X_test['sentence'][sentence_tag]
    data = tokenizer.encode(input_sentence, add_special_tokens=False)
    if len(data) > 15:
        continue
    baseline = [tokenizer.mask_token_id]*len(data)

    # convert to timeshap format
    data = np.expand_dims(np.array(data), axis=[0,2])
    baseline = np.expand_dims(np.array(baseline), axis=[0,2])

    nsamples = 256
    
    print(f'Processing sentence {i} of {len(X_test["sentence"])}')
    explainer = CompleteMobiusKernel(f, baseline, 37, 'event')
    phi = explainer.shap_values(data, nsamples=nsamples, pruning_idx=0)
    runtime = (datetime.datetime.now()-start).seconds
    print(f"Processing sentence {i} took {runtime} seconds", end='\n\n')

    folder = f'experiments/CompleteMöbius/{dataset}/{sentence_tag}'

    # save the results
    os.makedirs(folder, exist_ok=True)
    pd.DataFrame(phi, columns=['importance']).to_csv(f'{folder}/importance.csv', index=False)
    metadata = {'time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
                'label': str(X_test['label'][sentence_tag]),
                'prediction': str(f(data)[0][0]),
                'baseline model': str(explainer.fnull),
                'sentence': input_sentence,
                'model': 'BERT',
                'dataset': dataset,
                'mask_token': tokenizer.mask_token,
                'sentence tag': sentence_tag,
                'nsamples': nsamples,
                'runtime': runtime,
                }
    json.dump(metadata, open(f'{folder}/metadata.json', 'w'), indent=4)
    