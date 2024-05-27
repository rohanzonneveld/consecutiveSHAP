import numpy as np 
import pandas as pd 
import datetime
import json

# LSTM for sequence classification in the IMDB dataset
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.saving import load_model

from timeshap.explainer import local_pruning

from utils import HEDGE, MobiusHEDGE

np.random.seed(37)

# load the dataset but only keep the top n words, <UNK> the rest?
top_words = 10000
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=top_words)

# truncate and/or pad input sequences
max_review_length = 600
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
X_train = X_train[:, :, np.newaxis]
X_test = X_test[:, :, np.newaxis]

# load the model
model = load_model("models/LSTM.h5")

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

f = lambda x: model.predict(x)
mask_token = 0
baseline = np.expand_dims(np.array([mask_token]*600), axis=[0,2])
baseline[0, np.where(X_test == 1)[1][np.argmax(np.where(X_test == 1)[1])], 0] = 1

sentence_tag = 2104 # TODO: this does not correspond to bert dataset, use same dataset -> tokenizer raw input to imdb token ids
data = np.expand_dims(X_test[sentence_tag,:].copy(), axis=0)

pruning_idx = np.where(data == 1)[1][0]+1
hedge = MobiusHEDGE(f, data, baseline=baseline, pruning_idx=pruning_idx, win_size=2)
hedge.compute_shapley_hier_tree()

word_to_id = imdb.get_word_index()
word_to_id = {k:(v+3) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2
word_to_id["<UNUSED>"] = 3
id_to_word = {value:key for key,value in word_to_id.items()}

folder = f'lstm/{1}'
hedge.visualize_tree(id_to_word, folder=folder, tag=1)
pd.DataFrame(hedge.m.items(), columns=['term', 'value']).to_csv(f'experiments/HEDGE/{folder}/mobius_transforms.csv', index=False)
metadata = {'time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
            'label': str(y_test[sentence_tag]),
            'prediction': str(f(data)[0][0]),
            'baseline model': str(hedge.bias),
            'baseline game': str(hedge.bias+hedge.m0),
            'mobius sum': str(hedge.bias+sum(hedge.m.values())),
            'sentence': ' '.join([str(id_to_word[word[0]]) for word in X_test[sentence_tag,:]]),
            'model': 'LSTM',
            'dataset': 'IMDB',
            'mask_token': id_to_word[mask_token]}
json.dump(metadata, open(f'experiments/HEDGE/{folder}/metadata.json', 'w'), indent=4)