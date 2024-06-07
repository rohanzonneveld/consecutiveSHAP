import numpy as np 
import pandas as pd 

# LSTM for sequence classification in the IMDB dataset
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.saving import load_model

from timeshap.explainer import local_pruning

from utils import run_experiment

np.random.seed(37)

# load the dataset but only keep the top n words, zero the rest
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

baseline = np.expand_dims(np.array([0]*600), axis=1)

x = np.expand_dims(X_test[0,:].copy(), axis=0)

pruning_dict = {'tol': 0.1}

# coal_plot_data, coal_prun_idx = local_pruning(f, x, pruning_dict, baseline, False)
# # coal_prun_idx is in negative terms
# pruning_idx = x.shape[1] + coal_prun_idx
pruning_idx = x.shape[1] + -18
print(f'#features={x.shape[1] - pruning_idx + 1}')

results = run_experiment(f, x, baseline, pruning_idx, show_plot=False, output_path='experiments/text')