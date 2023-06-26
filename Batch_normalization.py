#----------------------------------------------------
# Пример использования алгоритма Batch Normalization
#----------------------------------------------------
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt
from tensorflow import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten, BatchNormalization

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train/255
x_test = x_test/255

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)


# берем только 5К значений из тестовой выборки (что бы эмитировать переобучение)
limit = 5000
x_train_data = x_train[:limit]
y_train_data = y_train_cat[:limit]

x_valid = x_train[limit:limit*2]
y_valid = y_train_cat[limit:limit*2]


model = keras.Sequential([
    Flatten(input_shape=(28,28,1)),
    Dense(300, activation='relu'),
    BatchNormalization(),
    Dense(10, activation='softmax')
])
# print(model.summary())

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
his = model.fit(x_train_data, y_train_data, batch_size=32, epochs=50, validation_data=(x_valid, y_valid))


plt.plot(his.history['loss'])
plt.plot(his.history['val_loss'])
plt.grid()
plt.show()