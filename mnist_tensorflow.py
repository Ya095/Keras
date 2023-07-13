import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from keras.datasets import mnist
from keras.utils import to_categorical


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

x_train = tf.reshape(tf.cast(x_train, dtype=tf.float32), [-1, 28*28])
x_test = tf.reshape(tf.cast(x_test, dtype=tf.float32), [-1, 28*28])
y_train = to_categorical(y_train, 10)


class DenseNN(tf.Module):
    def __init__(self, outputs, activate='relu'):
        super().__init__()
        self.outputs = outputs
        self.activate = activate
        self.fl_flag = False


    def __call__(self, x):
        if not self.fl_flag:
            self.w = tf.random.truncated_normal((x.shape[-1], self.outputs), stddev=0.1, name='w')
            self.b = tf.zeros([self.outputs], dtype=tf.float32, name='b')

            self.w = tf.Variable(self.w)
            self.b = tf.Variable(self.b)
            self.fl_flag = True

        y = x @ self.w + self.b

        if self.activate == 'relu':
            return tf.nn.relu(y)
        elif self.activate == 'softmax':
            return tf.nn.softmax(y)

        return y


class SeqModule(tf.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = DenseNN(128)
        self.layer_2 = DenseNN(10, activate='softmax')

    def __call__(self, x):
        return (self.layer_2(self.layer_1(x)))


model = SeqModule()

cross_entropy = lambda y_true, y_pred: tf.reduce_mean(tf.losses.categorical_crossentropy(y_true, y_pred))
opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

batch_size = 32
epochs = 10
total = x_train.shape[0]
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size=1024).batch(batch_size)

@tf.function
def trainBatch(x_batch, y_batch):
    with tf.GradientTape() as tape:
        f_loss = cross_entropy(y_batch, model(x_batch))

    grads = tape.gradient(f_loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return f_loss


for n in range(epochs):
    loss = 0
    for x_batch, y_batch in train_dataset:
       loss += trainBatch(x_batch, y_batch)

    print(f'loss = %.2f' % loss.numpy())


y = model(x_test)
y2 = tf.argmax(y, axis=1).numpy()
# acc = len(y_test[y_test == y2]) / y_test.shape[0] * 100 # кол-во верно распознанных делим на общее кол-во
# print(acc)

acc = tf.metrics.Accuracy()
acc.update_state(y_test, y2)
print(f'accuracy = %.2f' % (acc.result().numpy() * 100))