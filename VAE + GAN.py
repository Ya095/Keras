#-----------------------------------------------------
# Генеративно-состязательная НС (с VAE как генератор)
#-----------------------------------------------------
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import time
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras.datasets import mnist
import keras.backend as K
from keras.layers import Dense, Flatten, Reshape, Input, Lambda, BatchNormalization, Dropout
from keras.layers import Conv2D, LeakyReLU


(x_train, y_train), (x_test, y_test) = mnist.load_data()

""" Подготовка данных """
x_train = x_train[y_train == 7]
y_train = y_train[y_train == 7]

BUFFER_SIZE = x_train.shape[0]
BATCH_SIZE = 100
length = BUFFER_SIZE // BATCH_SIZE * BATCH_SIZE

x_train = x_train[:length]
y_train = y_train[:length]

# стандартизация данных
x_train = x_train / 255
x_test = x_test / 255

x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


""" Формирование НС """
hidden_size = 2

def dropoutAndBatch(x):
    return Dropout(0.3)(BatchNormalization()(x))


def noiser(args):
    global z_mean, z_ln_var
    z_mean, z_ln_var = args
    N = K.random_normal(shape=(BATCH_SIZE, hidden_size), mean=0., stddev=1.0)
    return K.exp(z_ln_var / 2) * N + z_mean


# генератор
input_img = Input((28,28,1))
x = Flatten()(input_img)
x = Dense(256, activation='relu')(x)
x = dropoutAndBatch(x)

z_mean = Dense(hidden_size)(x)
z_ln_var = Dense(hidden_size)(x)

h = Lambda(noiser, output_shape=(hidden_size,))([z_mean, z_ln_var])

input_dec = Input((hidden_size,))
d = Dense(256, activation='relu')(input_dec)
d = dropoutAndBatch(d)
d = Dense(28*28, activation='sigmoid')(d)
decoded = Reshape((28,28,1))(d)

encoder = keras.Model(input_img, h, name='encoder')
decoder = keras.Model(input_dec, decoded, name='decoder')
generator = keras.Model(input_img, decoder(encoder(input_img)), name='generator')


# дискриминатор
discriminator = keras.Sequential([
    Conv2D(64, (5,5), strides=(2,2), padding='same', input_shape=[28,28,1]),
    LeakyReLU(),
    Dropout(0.3),

    Conv2D(128, (5,5), strides=(2,2), padding='same'),
    LeakyReLU(),
    Dropout(0.3),

    Flatten(),
    Dense(1)
])


""" Вычисляем потерь """
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# потери для генератора || cross_entropy(желаемый отклик, реальный отклик)
def generator_loss(fake_output):
    loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    kl_loss = -0.5 * K.sum(1 + z_ln_var - K.square(z_mean) - K.exp(z_ln_var), axis=-1)
    return loss + kl_loss*0.1


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fale_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fale_loss
    return total_loss

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


""" Обучение """
@tf.function
def trainStep(images):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(images, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss


def train(dataset, epochs):
    history = []
    MAX_PRINT_LABEL = 10
    th = BUFFER_SIZE // (BATCH_SIZE * MAX_PRINT_LABEL)
    n_epoch = 1

    for epoch in range(epochs):
        print(f'{n_epoch}/{EPOCHS}: ', end='')

        start = time.time()
        n = 0
        gen_loss_epoch = 0

        for image_batch in dataset:
            gen_loss, disc_loss = trainStep(image_batch)
            gen_loss_epoch += K.mean(gen_loss)
            if (n % th == 0):
                print('=', end='')
            n += 1

        history += [gen_loss_epoch / n]
        print(': ' + str(history[-1]))
        print('Время эпохи {0} в {1:.2f} секундах'.format(epoch+1, time.time() - start))

        n_epoch += 1

    return history


""" Запуск процесса обучения """
EPOCHS = 20
history = train(train_dataset, EPOCHS)

h = encoder.predict(x_test[:6000], batch_size=BATCH_SIZE)
plt.scatter(h[:, 0], h[:, 1])
plt.show()

plt.plot(history)
plt.grid()
plt.show()


""" Отображение результатов генерации """
n = 2
total = 2*n+1
plt.figure(figsize=(total, total))
num = 1

for i in range(-n, n +1):
    for j in range(-n, n+1):
        ax = plt.subplot(total, total, num)
        num += 1
        img = decoder.predict(np.expand_dims([0.5*i/n, 0.5*j/n], axis=0))
        plt.imshow(img.squeeze(), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
plt.show()