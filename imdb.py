import numpy as np
from keras.datasets import imdb
from keras import models
from keras import layers
from keras import optimizers
import Tkinter
import matplotlib.pyplot as plt


#test

def vectorize_seq(train_data, dimension = 10000):
    result = np.zeros((len(train_data), dimension))
    for i, seq in enumerate(train_data):
        result[i, seq] = 1

    return result



(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
word_index = imdb.get_word_index()
reverse_word_index = dict([(val, key) for (key, val) in word_index.items()])
review = ''.join([reverse_word_index.get(i-3, '?') for i in train_data[0]])
X_train = vectorize_seq(train_data)
X_test = vectorize_seq(test_data)

y_train = train_labels.astype('float32')
y_test = test_labels.astype('float32')



model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape = (10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

x_val = X_train[0:10000]
x_train_partial = X_train[10000:]
y_val = y_train[0:10000]
y_train_partial = y_train[10000:]

history = model.fit(x_train_partial, y_train_partial, epochs=4, batch_size=512, validation_data=(x_val, y_val))

train_acc = history.history['acc']
train_loss = history.history['loss']
val_acc = history.history['val_acc']
val_loss = history.history['val_loss']
epoch = np.arange(len(train_acc)) + 1

plt.figure(0)
plt.plot(epoch, train_acc, 'bo', label = 'Training Acc')
plt.plot(epoch, val_acc, 'b', label = 'Val Acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()


plt.figure(1)
plt.plot(epoch, train_loss, 'bo', label = 'Training Loss')
plt.plot(epoch, val_loss, 'b', label = 'Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
