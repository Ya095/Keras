#--------------------------------
# Простой вариант автоэнкодера.
# Пример восстановления изображения.
#--------------------------------
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Flatten, Input, Reshape


(x_train, y_train), (x_test, y_test) = mnist.load_data()

# стандартизация данных (от 0 до 1)
x_train = x_train / 255
x_test = x_test / 255

# приведение к нужному размеру для подачи на вход НС
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

# построение модели НС
# (sigmoid - тк на выходе надо получить числа от 0 до 1 = входным данным)
input_img = Input((28, 28, 1))
x = Flatten()(input_img)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
encoded = Dense(49, activation='relu')(x) # вектор скрытого состояния
d = Dense(64, activation='relu')(encoded)
d = Dense(28*28, activation='sigmoid')(d)
decoded = Reshape((28, 28, 1))(d) # изменение размерности

# формирование и обучение модели
autoencoder = Model(input_img, decoded, name='autoencoder')
autoencoder.compile(optimizer='Adam', loss='mean_squared_error')
autoencoder.fit(x_train, x_train, epochs=20, batch_size=100, shuffle=True)

# просмотр результата
n = 10
imgs = x_test[:n]
decoded_imgs = autoencoder.predict(x_test[:n], batch_size=n)

plt.figure(figsize=(n, 2))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(imgs[i].squeeze(), cmap='gray') # удаление оси цветового канала
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax2 = plt.subplot(2, n, i+n+1)
    plt.imshow(decoded_imgs[i].squeeze(), cmap='gray')
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)

plt.show()