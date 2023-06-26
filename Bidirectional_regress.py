#----------------------------------------
# Двунаправленная RNN. Апроксимация графика (регрессия)
#----------------------------------------
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, GRU, Input, Bidirectional
from keras.models import Sequential
from keras.optimizers import Adam


N = 10000
data = np.array([np.sin(x/20) for x in range(N)]) + 0.1 * np.random.randn(N)
# plt.plot(data[:100])
# plt.show()

# формирование обучающей выборки
off = 3 # сколько берем отчетов до и после
length = off * 2 + 1 # всего отчетов
x = np.array([np.diag(np.hstack((data[i:i+off], data[i+off+1:i+length]))) for i in range(N - length)])
y = data[off:N-off-1]
# print(x.shape, y.shape) # -> (9993, 6, 6) (9993,)

# создание модели НС
model = Sequential()
model.add(Input((length-1, length-1)))
model.add(Bidirectional(GRU(2)))
model.add(Dense(1, activation='linear'))
model.summary()

model.compile(loss='mean_squared_error', optimizer=Adam(0.01))
hist = model.fit(x, y, batch_size=32, epochs=10)

# делаем прогноз
m = 200 # кол-во прогнозов
xx = np.zeros(m)
xx[:off] = data[:off] # первые эл-ты просто копируем
for i in range(m - off - 1):
    x = np.diag(np.hstack((xx[i:i+off], data[i+off+1:i+length])))
    x = np.expand_dims(x, axis=0) # подгоняем под размер
    y = model.predict(x)
    xx[i+off+1] = y # добавляем зн. на нужный индекс

plt.plot(xx[:m])
plt.plot(data[:m])
plt.show()
