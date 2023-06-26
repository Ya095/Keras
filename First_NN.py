#------------------------------------
# Обучаем простую НС с 1 нейроном
#------------------------------------
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import keras
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
from keras.optimizers import Adam

c = np.array([-40, -10, 0, 8, 15, 22, 38])
f = np.array([-40, 14, 32, 46, 59, 72, 100])

model = keras.models.Sequential() # Создание объекта последовательной многослойной НС.
model.add(Dense(units=1, input_shape=(1,), activation='linear'))

# задаем функцию потерь (критерий качества) и способ оптимизации град. спуска [после этого можем обучать НС]
model.compile(loss='mean_squared_error', optimizer=Adam(0.1))
print(model.summary())

history = model.fit(c, f, epochs=500, verbose=True) # обучение модели (НС)

print(model.predict([100])) # предсказание НС (в идеале = 212)
print(model.get_weights()) # получаем итоговые коэф. модели

plt.plot(history.history['loss'])
plt.grid()
plt.show()