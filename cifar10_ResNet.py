#---------------------------------------------------
# НС для распознавания цветных изображений cifar10
#---------------------------------------------------

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.layers import Conv2D, Input, Dropout, BatchNormalization, Dense, GlobalAveragePooling2D
from keras.layers import add
from keras import Model
from keras.utils import to_categorical
from keras.datasets import cifar10
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train / 255
x_test = x_test / 255

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

def dropoutAndBatch(x):
    return Dropout(0.3)(BatchNormalization()(x))


""" Структура НС """
inputs = Input(shape=(32, 32, 3), name='img')
x = Conv2D(32, (3,3), activation='relu')(inputs)
x = Conv2D(64, (3,3), activation='relu')(x)
x = dropoutAndBatch(x)
block_1_output = Conv2D(64, (3,3), activation='relu', padding='same', strides=3)(x)

x = Conv2D(64, (3,3), activation='relu', padding='same')(block_1_output)
x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = dropoutAndBatch(x)
block_2_output = add([block_1_output, x])

x = Conv2D(64, (3,3), activation='relu', padding='same')(block_2_output)
x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = dropoutAndBatch(x)
block_3_output = add([block_2_output, x])

x = Conv2D(128, (3,3), activation='relu', padding='same')(block_3_output)
x = dropoutAndBatch(x)
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
outputs = Dense(10, activation='softmax')(x)


""" Обучение модели """
model = Model(inputs, outputs, name='toy_resnet')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=(['accuracy']))
hist = model.fit(x_train, y_train, batch_size=64, epochs=15, validation_split=0.15)
print(model.evaluate(x_test, y_test))


""" Ошибки на графике """
plt.plot(hist.history['loss'], label='loss')
plt.plot(hist.history['val_los'], label='val_loss')
plt.legend()
plt.grid()
plt.show()