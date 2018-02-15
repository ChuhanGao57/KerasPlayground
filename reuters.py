from keras.datasets import reuters
import numpy as np

def vectorize(train_data, dimension=10000):
    result = np.zeros((len(train_data), dimension))
    for i, sample in enumerate(train_data):
        result[i, sample] = 1.0
    
    return result





(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

word_index = reuters.get_word_index()
reverse_word_index = dict([(val, key) for (key, val) in word_index.items()])
review = ''.join([reverse_word_index.get(i-3, '?') for i in train_data[0]])

x_train = vectorize(train_data)
x_test = vectorize(test_data)
y_train = vectorize(train_labels, dimension=max(train_labels)+1)
y_test = vectorize(test_labels, dimension=max(train_labels)+1)

