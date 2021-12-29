# ===================================================================================================
# PROBLEM B4
#
# Build and train a classifier for the BBC-text dataset.
# This is a multiclass classification problem.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in: http://mlg.ucd.ie/datasets/bbc.html.
#
# Desired accuracy and validation_accuracy > 91%
# ===================================================================================================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))


def solution_B4():
    bbc = pd.read_csv('https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/bbc-text.csv')
    
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"
    training_portion = .8

    train_size = int(len(bbc) * training_portion)

    text = bbc['text']
    label = bbc['category']

    text_clean = []
    for t in text:
        for word in STOPWORDS:
            token = ' ' + word + ' '
            t = t.replace(token, ' ')
            t = t.replace(' ', ' ')
        text_clean.append(t)

    training_text = text_clean[:train_size]
    testing_text = text_clean[train_size:]
    training_label = label[:train_size]
    testing_label = label[train_size:]

    tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_text)

    sequences = tokenizer.texts_to_sequences(training_text)
    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    testing_sequences = tokenizer.texts_to_sequences(testing_text)
    testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(label)

    training_label_seq = np.array(label_tokenizer.texts_to_sequences(training_label))
    testing_label_seq = np.array(label_tokenizer.texts_to_sequences(testing_label))

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
        tf.keras.layers.Dense(embedding_dim, activation='relu'),
        tf.keras.layers.Dense(6, activation='softmax')
    ])

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('accuracy') > 0.91 and logs.get('val_accuracy') > 0.91 and epoch > 50):
                self.model.stop_training = True

    callbacks = myCallback()

    model.fit(
        padded,
        training_label_seq,
        epochs=200,
        validation_data=(testing_padded, testing_label_seq),
        callbacks=[callbacks],
        verbose=2
    )

    return model

    # The code below is to save your model as a .h5 file.
    # It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    model = solution_B4()
    model.save("model_B4.h5")
