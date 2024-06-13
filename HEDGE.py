import numpy as np 
import pandas as pd 
import torch
import torch.nn.functional as F
import datetime 
import json

# LSTM for sequence classification in the IMDB dataset
from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer)

from utils import HEDGE

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


for i in range(1004,1005):#len(X_test['sentence'])):
    
    start = datetime.datetime.now()
    sentence_tag = i
    input_sentence = X_test['sentence'][sentence_tag]
    data = tokenizer.encode(input_sentence, add_special_tokens=False)
    # if len(data) <= 9 or len(data) > 15:
    #     continue
    baseline = [tokenizer.mask_token_id]*len(data)

    print(f'Processing sentence {i} of {len(X_test["sentence"])}')

    # convert to timeshap format
    data = np.expand_dims(np.array(data), axis=[0,2])
    baseline = np.expand_dims(np.array(baseline), axis=[0,2])

    hedge = HEDGE(f, data, baseline=baseline)
    hedge.shapley_topdown_tree()

    folder = f'experiments/HEDGE/{dataset}/{sentence_tag}'
    hedge.visualize_tree(tokenizer.ids_to_tokens, folder=folder, tag=sentence_tag)
    phrase_list, score_list = hedge.get_importance_phrase()
    phrase_list = [phrase for phrase in phrase_list if len(phrase) > 1]
    max_inter_set = phrase_list[0]
    importance = [fea[1] for fea in hedge.hier_tree[hedge.max_level]]
    importance.reverse() # other importance metrics are in order of timeshap so reversed
    runtime = (datetime.datetime.now()-start).seconds
    print(f"Processing sentence {i} took {runtime} seconds", end='\n\n')

    # save the results
    pd.DataFrame(importance, columns=['importance']).to_csv(f'{folder}/importance.csv', index=False)
    metadata = {'time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
                'label': str(X_test['label'][sentence_tag]),
                'prediction': str(f(data)[0][0]),
                'baseline model': str(hedge.bias),
                'sentence': input_sentence,
                'model': dataset,
                'dataset': 'IMDB',
                'mask_token': tokenizer.mask_token,
                'sentence tag': sentence_tag,
                'max interaction set': max_inter_set,
                'runtime': runtime,}
    json.dump(metadata, open(f'{folder}/metadata.json', 'w'), indent=4)
    