#----------------------------------------------------
# Обучаем простую НС различать цифры от 0 до 9 (черно-белые, 28*28 пикселей)
#----------------------------------------------------
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten


# загружаем данные для обучения и теста
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# нормализация входных данных от 0 до 1
x_train = x_train/255
x_test = x_test/255

# преобразование выходных значений в векторы (по категориям)
# вида [0,0,0,1,0,0,0,0,0,0] - тут в выводе цифра 3
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)


# создание структуры НС
model = keras.Sequential([
    Flatten(input_shape=(28,28,1)),
    Dense(130, activation='relu'),
    Dense(10, activation='softmax')
])
print(model.summary())

# myOpt = keras.optimizers.Adam(learning_rate=0.01) - задать вручную параметры для Adam
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.1)
print()
model.evaluate(x_test, y_test_cat) # проверка на тестовой выборке (выводится в консоль)

# делаем предсказание обученной НС (берем из тестовой выборки)
n = 0
x = np.expand_dims(x_test[n], axis=0) # расширяем, тк на входе нужен 3-х мерный массив
res = model.predict(x)
# print('\n', f' res = {res}')
print('\n', f'Распознанная цифра: {np.argmax(res)}')
# plt.imshow(x_test[n], cmap=plt.cm.binary)
# plt.show()

# распознавание всей тестовой выборки
pred = model.predict(x_test)
pred = np.argmax(pred, axis=1)

# Выделение неверных вариантов
mask = pred == y_test
x_false = x_test[~mask]
y_false = pred[~mask]
print('\n', x_false.shape)