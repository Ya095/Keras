#------------------------------------
# Предсказываем цену зданий в Бостоне
#------------------------------------
from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

# нормировка данных
mean = x_train.mean(axis=0)
std = x_train.std(axis=0)
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

# создание модели
model = Sequential()
model.add(Dense(128, input_shape=x_train[1].shape, activation='relu'))
model.add(Dense(1))

# компиляция и обучение модели
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
hist = model.fit(x_train, y_train, batch_size=1, epochs=80)

model.evaluate(x_test, y_test) # [13.9684 - mae: 2.4350]
pred = model.predict(x_test)

print('\n', 'Примеры предсказаний')
print(pred[1], y_test[1])
print(pred[50], y_test[50])
print(pred[100], y_test[100])

plt.plot(hist.history['loss'], label='loss')
plt.plot(hist.history['val_loss'], label='val_loss')
plt.yscale('log')
plt.legend()
plt.show()