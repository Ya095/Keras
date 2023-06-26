#--------------------------
# Раскраска изображений
#--------------------------
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.layers import Conv2D, UpSampling2D, InputLayer
from keras.models import Sequential
from skimage.color import rgb2lab, lab2rgb
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


img = Image.open('images/cats400.jpeg')

def processed_image(img):
    """ преобразование изображения в формат lab
        'Image.BILINEAR' -> для изменения размерности
     """
    image = img.resize((256,256), Image.BILINEAR)
    image = np.array(image, dtype=float)
    size = image.shape
    lab = rgb2lab(1.0 / 255 * image)
    X, Y = lab[:, :, 0], lab[:, :, 1:]

    Y /= 128  # нормируем выходные значение в диапазон от -1 до 1
    X = X.reshape(1, size[0], size[1], 1)
    Y = Y.reshape(1, size[0], size[1], 2)
    return X, Y, size


X, Y, size = processed_image(img)

model = Sequential()
model.add(InputLayer(input_shape=(None, None, 1)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
model.add(UpSampling2D((2, 2)))

model.compile(optimizer='adam', loss='mse')
model.fit(x=X, y=Y, batch_size=1, epochs=50) # batch_size=1, тк всего 1 изображение

image_2 = Image.open('images/cats400_2.jpeg')
X, Y, size = processed_image(image_2)

output = model.predict(X)
output *= 128

min_vals, max_vals = -128, 127
ab = np.clip(output[0], min_vals, max_vals)

cur = np.zeros((size[0], size[1], 3))
cur[:,:,0] = np.clip(X[0][:,:,0], 0, 100)
cur[:,:,1:] = ab
plt.subplot(1, 2, 1)
plt.imshow(image_2)
plt.subplot(1, 2, 2)
plt.imshow(lab2rgb(cur))
plt.show()