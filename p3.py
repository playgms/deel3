import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
# from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import reuters

import matplotlib.pyplot as plt
(train_data,train_label),(test_data,test_label)=reuters.load_data(num_words=10000)
word_index=reuters.get_word_index()
rev_word_index=dict([(value,key) for (key,value) in word_index.items()])
decoded_newswire=''.join(rev_word_index.get(i-3,'?') for i in train_data[0])
def vectorize_sequences(sequences,dimensions=10000):
    results=np.zeros((len(sequences),dimensions))
    for i, sequence in enumerate(sequences):
      for word in set(sequence):
        if word < dimensions:
          results[i,word]+=1
    return results
x_train=vectorize_sequences(train_data,dimensions=10000)
x_test=vectorize_sequences(test_data,dimensions=10000)
from tensorflow.keras.utils import to_categorical
onehot_train_labels=to_categorical(train_label)
onehot_test_labels=to_categorical(test_label)
model=Sequential([
      layers.Dense(64,activation='relu'),
      layers.Dense(64,activation='relu'),
      layers.Dense(46,activation='softmax'),
      
])
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
x_val=x_train[:1000]
partial_x_train=x_train[1000:]
y_val=onehot_train_labels[:1000]
partial_y_val=onehot_train
history=model.fit(partial_x_train,partial_y_val,epochs=10,batch_size=10,validation_data=(x_val,y_val))
result=model.evaluate(x_test,onehot_test_labels)
print('accuracy=',result[1])
plt.plot(history.history['loss'],label='Test loss')
plt.plot(history.history['val_loss'],label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
plt.show()
