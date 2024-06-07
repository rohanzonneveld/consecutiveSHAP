import numpy as np 

# LSTM for sequence classification in the IMDB dataset
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.preprocessing import sequence
np.random.seed(7)

top_words = 10000
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=top_words)

max_review_length = 600
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
X_train = X_train[:, :, np.newaxis]
X_test = X_test[:, :, np.newaxis]

# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words+1, embedding_vecor_length, input_shape=(max_review_length,)))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y_train, epochs=10, batch_size=64)

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

model.save("models/LSTM.keras")
